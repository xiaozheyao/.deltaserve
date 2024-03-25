# pylint: disable=unused-argument
import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Tuple, Optional, List, Any
from transformers.configuration_utils import PretrainedConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_gather,
)
from .delta import DeltaLayerWeights, PackedDeltaLayerWeights
from .quant_linear import QuantLinear
from .config import DeltaConfig
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)

logger = init_logger(__name__)

if TYPE_CHECKING:
    pass


def _apply_delta(
    x: torch.Tensor,
    quant_linears: List[QuantLinear],
    base_output: torch.Tensor
):
    """Applies multiple delta to the input tensor"""
    # todo(xiaozhe): checkout when quant_linear
    # and x are on different devices
    outputs = []
    for ql in quant_linears:
        outputs.append(ql(x))
    return torch.stack(outputs, dim=0) + base_output


@dataclass
class DeltaMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)


class BaseLayerWithDelta(nn.Module):

    def create_delta_weights(self, max_deltas: int, delta_config: DeltaConfig,
                             model_config: PretrainedConfig) -> None:
        """Initializes lora matrices."""
        ...

    def reset_delta(self, index: int):
        """Resets the delta weights at index back to 0."""
        ...

    def set_delta(
        self,
        index: int,
        delta: Any,
    ):
        """Overwrites delta tensors at index."""
        ...

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        """Sets the mapping indices."""
        ...


class VocabParallelEmbeddingWithDelta(BaseLayerWithDelta):

    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer

    def reset_delta(self, index: int):
        pass

    def set_delta(
        self,
        int: int,
        delta: List[QuantLinear],
    ):
        pass

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.embeddings_indices = embeddings_indices
        self.indices_len = indices_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x)


class ColumnParallelLinearWithDelta(BaseLayerWithDelta):

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

    def reset_delta(self, index: int):
        pass
    
    def create_delta_weights(
        self,
        max_deltas: int,
        # let's pretend all quantization is done to the same bit width
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None
    ) -> None:
        self.qls: List[QuantLinear] = [None] * max_deltas
        
    def set_delta(self, index: int, delta: DeltaLayerWeights):
        self.qls[index] = QuantLinear.from_tensors(
            delta.qweight[0],
            delta.qzeros[0],
            delta.scales[0],
            delta.g_idx[0],
            bias=None,)
        
    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply_weights(self, x:torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias
        )
        output = _apply_delta(x, self.qls)
        return output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"ColumnParallelLinearWithDelta")
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)
        output_parallel = self.apply_weights(x, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias

    @property
    def linear_weights(self):
        return self.base_layer.linear_weights

class MergedColumnParallelLinearWithDelta(ColumnParallelLinearWithDelta):

    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"MergedColumnParallelLinearWithDelta")
        return self.base_layer(x)


class QKVParallelLinearWithDelta(ColumnParallelLinearWithDelta):

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def create_delta_weights(self, max_deltas: int, delta_config: DeltaConfig, model_config: Optional[PretrainedConfig] = None) -> None:
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = tp_rank
        self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"QKVParallelLinearWithDelta")
        return self.base_layer(x)


class RowParallelLinearWithDelta(BaseLayerWithDelta):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"RowParallelLinearWithDelta")
        return self.base_layer(x)


class SamplerWithDelta(BaseLayerWithDelta):

    def __init__(
        self,
        base_layer: Sampler,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

    @property
    def logits_as_hidden_states(self):
        return self.base_layer.logits_as_hidden_states

    @property
    def vocab_size(self):
        return self.base_layer.vocab_size

    @property
    def org_vocab_size(self):
        return self.base_layer.org_vocab_size

    @property
    def include_gpu_probs_tensor(self):
        return self.base_layer.include_gpu_probs_tensor

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_gather(logits)
        if logits is None:
            return None
        # Remove paddings in vocab (if any).
        logits = logits[:, :self.base_layer.vocab_size]
        return logits

    def forward(self, *args, **kwargs):
        return type(self.base_layer).forward(self, *args, **kwargs)


def from_layer(
        layer: nn.Module,
        max_loras: int,
        lora_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None) -> BaseLayerWithDelta:
    supported_layer_types = {
        VocabParallelEmbedding: VocabParallelEmbeddingWithDelta,
        ColumnParallelLinear: ColumnParallelLinearWithDelta,
        QKVParallelLinear: QKVParallelLinearWithDelta,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithDelta,
        RowParallelLinear: RowParallelLinearWithDelta,
    }
    for src_layer_type, delta_layer_type in supported_layer_types.items():
        if type(layer) is src_layer_type:  # pylint: disable=unidiomatic-typecheck
            ret = delta_layer_type(layer)
            ret.create_delta_weights(max_loras, lora_config, model_config)
            return ret
    return layer


def from_layer_sampler(
    layer: Sampler,
    lm_head: ParallelLMHead,
    max_loras: int,
    lora_config: DeltaConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> SamplerWithDelta:
    ret = SamplerWithDelta(layer, lm_head.embedding_dim, lm_head.weight.dtype,
                           lm_head.weight.device)
    # ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret
