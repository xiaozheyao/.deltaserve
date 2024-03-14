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
    VocabParallelEmbedding, ParallelLMHead
)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_gather,
)

from .quant_linear import QuantLinear
from .config import DeltaConfig

if TYPE_CHECKING:
    pass

def _apply_delta(
    x: torch.Tensor,
    quant_linears: List[QuantLinear],
):
    """Applies multiple delta to the input tensor"""
    # todo(xiaozhe): checkout when quant_linear 
    # and x are on different devices
    outputs = []
    for ql in quant_linears:
        outputs.append(ql(x))
    return torch.stack(outputs, dim=0)

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
        self, int: int,
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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.base_layer(x)
    
class ColumnParallelLinearWithDelta(BaseLayerWithDelta):
    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
    
    def reset_delta(self, index: int):
        pass
    
    def set_delta(self, index: int, delta: Any):
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
        self.indices_len = indices_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"ColumnParallelLinearWithDelta")
        pass
class MergedColumnParallelLinearWithDelta(ColumnParallelLinearWithDelta):
    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"MergedColumnParallelLinearWithDelta")
        return self.base_layer(x)

class QKVParallelLinearWithDelta(ColumnParallelLinearWithDelta):
    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__(base_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"QKVParallelLinearWithDelta")
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
