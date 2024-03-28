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
from .quant_linear import QuantLinear
from .config import DeltaConfig
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)

logger = init_logger(__name__)

TEST_BITS = 4
TEST_GROUPSIZE = 128

if TYPE_CHECKING:
    pass


def _apply_delta(x: torch.Tensor, quant_linears: List[QuantLinear],
                 base_output: torch.Tensor):
    """Applies multiple delta to the input tensor"""
    # todo(xiaozhe): checkout when quant_linear
    # and x are on different devices
    x = x.view(-1, x.shape[-1])
    base_output = base_output.view(-1, base_output.shape[-1])
    outputs = []
    for ql in quant_linears:
        if ql:
            # for qkv packed proj, ql is a list of 3 QuantLinear
            # we need to merge them in this case
            if isinstance(ql, list):
                outputs.append(torch.cat([ql[i](x) for i in range(3)], dim=-1))
            else:
                outputs.append(ql(x))
        else:
            # todo(xiaozhe): This is a hack to make sure the profile runs smoothly, later we should remove this
            # and update the profile to handle this case
            outputs.append(torch.zeros_like(base_output))
    output = torch.stack(outputs, dim=0)
    return (output + base_output).view_as(base_output)


@dataclass
class DeltaMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)

    def __str__(self):
        return f"index_mapping: {self.index_mapping}, prompt_mapping: {self.prompt_mapping}"


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
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
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
        index: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
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
        self.tp_size = get_tensor_model_parallel_world_size()

    def reset_delta(self, index: int):
        pass

    def create_delta_weights(
            self,
            max_deltas: int,
            # let's pretend all quantization is done to the same bit width
            delta_config: DeltaConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:

        self.qls: List[QuantLinear] = [None] * max_deltas
        scale_and_zero_size = self.base_layer.weight.shape[1] // TEST_GROUPSIZE
        # (without considering pack factor)
        # input size: self.base_layer.weight.shape[1]
        # output size: self.base_layer.weight.shape[0]
        self.qweight_stacked = torch.zeros(
            max_deltas,
            1,
            self.base_layer.weight.shape[1] // delta_config.pack_factor,
            self.output_size_per_partition,
            dtype=delta_config.delta_dtype,
            device=self.base_layer.weight.device
        )
        self.qzeros_stacked = torch.zeros(
            max_deltas,
            1,
            scale_and_zero_size,
            self.output_size_per_partition // delta_config.pack_factor,
            dtype=torch.int32
        )
        self.scales_stacked = torch.zeros(
            max_deltas,
            1,
            scale_and_zero_size,
            self.output_size_per_partition,
            dtype=delta_config.delta_dtype,
            device=self.base_layer.weight.device
        )
        self.g_idx = torch.tensor(
            [
                i // self.quant_config.group_size
                for i in range(self.input_size)
            ],
            dtype=torch.int32,
        )

    def set_delta(self,
                  index: int,
                  qweight: torch.Tensor,
                  qzeros: torch.Tensor,
                  scales: torch.Tensor,
                  g_idx: torch.Tensor,):
        self.reset_delta(index)
        if self.tp_size > 1:
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.output_dim
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = start_idx + shard_size
            # here we are only considering the quantized linear for current tp rank
            pass
        else:
            pass
            # self.qls[index] = QuantLinear.from_tensors(
            #     qweight=qweight,
            #     qzeros=qzeros,
            #     scales=scales,
            #     g_idx=g_idx,
            #     bias=None,
            # )

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

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        output = _apply_delta(x, self.qls, output)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"MergedColumnParallelLinearWithDelta")
        return self.base_layer(x)

    def reset_delta(self, index: int):
        self.qweight_stacked[0][index] = 0
        self.qweight_stacked[1][index] = 0

    def create_delta_weights(self, max_deltas: int, delta_config: DeltaConfig, model_config: PretrainedConfig | None = None) -> None:
        n_slices = 2
        if not (len(self.base_layer.output_sizes) == n_slices
                and self.base_layer.output_sizes[0]
                == self.base_layer.output_sizes[1]):
            raise ValueError(
                "LoRAColumnParallelLinear2Slice requires 2 slices with "
                "the same size.")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.qls: List[QuantLinear] = [None] * max_deltas
        """
        (without considering pack factor)
        input shape: self.base_layer.weight.shape[1]
        output size: self.base_layer.weight.shape[0]
        """
        self.qweight_stacked = tuple(
            torch.zeros(
                max_deltas,
                1,
                self.base_layer.weight.shape[0] // delta_config.pack_factor,
                self.base_layer.weight.shape[1],
                dtype=delta_config.delta_dtype,
                device=self.base_layer.weight.device
            ) for _ in range(n_slices))


class QKVParallelLinearWithDelta(ColumnParallelLinearWithDelta):

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.q_proj_total_size = (self.base_layer.total_num_heads *
                                  self.base_layer.head_size)
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.kv_proj_total_size = (self.base_layer.total_num_kv_heads *
                                   self.base_layer.head_size)

    def create_delta_weights(
            self,
            max_deltas: int,
            delta_config: DeltaConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = tp_rank
        self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        self.qls: List[QuantLinear] = [None] * max_deltas

    def set_delta(self,
                  index: int,
                  qweight: torch.Tensor,
                  qzeros: torch.Tensor,
                  scales: torch.Tensor,
                  g_idx: torch.Tensor
                  ):
        self.reset_delta(index)
        if self.tp_size == 1:
            self.qls[index] = QuantLinear.from_tensors(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                bias=None,
            )
        else:
            raise NotImplementedError(
                "QKVParallelLinearWithDelta only supports TP size 1")

    def apply_weights(self, x: torch.Tensor, bias: Any | None) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        output = _apply_delta(x, self.qls, output)
        return output


class MergedQKVParallelLinearWithDelta(ColumnParallelLinearWithDelta):
    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)

    def create_delta_weights(self, max_deltas: int, delta_config: DeltaConfig, model_config: PretrainedConfig | None = None) -> None:
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = tp_rank
        self.pack_factor = delta_config.pack_factor
        self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        # qkv deltas
        self.qweight_stacked = (
            torch.zeros(
                max_deltas,
                1,
                self.base_layer.weight.shape[0] // delta_config.pack_factor,
                self.base_layer.weight.shape[1],
                dtype=delta_config.delta_dtype,
                device=self.base_layer.weight.device
            ),
            torch.zeros(
                max_deltas,
                1,
                self.base_layer.weight.shape[0] // delta_config.pack_factor,
                self.base_layer.weight.shape[1],
                dtype=delta_config.delta_dtype,
                device=self.base_layer.weight.device
            ),
            torch.zeros(
                max_deltas,
                1,
                self.base_layer.weight.shape[0] // delta_config.pack_factor,
                self.base_layer.weight.shape[1],
                dtype=delta_config.delta_dtype,
                device=self.base_layer.weight.device
            )
        )
        self.qzeros_stacked = (
            torch.zeros(
                max_deltas,
                1,
                1,
                self.base_layer.weight.shape[1] // delta_config.pack_factor,
                dtype=torch.int32
            ) * 3
        )
        self.scales_stacked = (torch.zeros(
            max_deltas,
            1,
            1,
            self.base_layer.weight.shape[0],
            dtype=delta_config.delta_dtype,
            device=self.base_layer.weight.device
        ) *3 )
        self.g_idx = torch.tensor(
            [
                i // TEST_GROUPSIZE
                for i in range(self.base_layer.weight.shape[1])
            ],
            dtype=torch.int32,
        )

    def set_delta(self,
                  index: int,
                  qweight: List[torch.Tensor],
                  qzeros: List[torch.Tensor],
                  scales: List[torch.Tensor],
                  g_idx: List[torch.Tensor]
                  ):
        if self.tp_size > 1:
            pass
        else:
            if qweight[0] is not None:
                print(f"qweight[0].shape: {qweight[0].shape}")
                print(f"self.qweight_stacked[0].shape: {self.qweight_stacked[0].shape}")
                
                self.qweight_stacked[0][index, 0, :qweight[0].shape[0], :qweight[0].shape[1]].copy_(
                    qweight[0], non_blocking=True)
                
            if qweight[1] is not None:
                self.qweight_stacked[1][index, 0, :qweight[1].shape[1], :qweight[1].shape[0]].copy_(
                    qweight[1].T, non_blocking=True)
            if qweight[2] is not None:
                self.qweight_stacked[2][index, 0, :qweight[2].shape[1], :qweight[2].shape[0]].copy_(
                    qweight[2].T, non_blocking=True)
                
            if qzeros[0] is not None:
                self.qzeros_stacked[0][index, 0, :qzeros[0].shape[1], :qzeros[0].shape[0]].copy_(
                    qzeros[0].T, non_blocking=True)
            if qzeros[1] is not None:
                self.qzeros_stacked[1][index, 0, :qzeros[1].shape[1], :qzeros[1].shape[0]].copy_(
                    qzeros[1].T, non_blocking=True)
            if qzeros[2] is not None:
                self.qzeros_stacked[2][index, 0, :qzeros[2].shape[1], :qzeros[2].shape[0]].copy_(
                    qzeros[2].T, non_blocking=True)
            if scales[0] is not None:
                self.scales_stacked[0][index, 0, :scales[0].shape[1], :scales[0].shape[0]].copy_(
                    scales[0].T, non_blocking=True)
            if scales[1] is not None:
                self.scales_stacked[1][index, 0, :scales[1].shape[1], :scales[1].shape[0]].copy_(
                    scales[1].T, non_blocking=True)
            if scales[2] is not None:
                self.scales_stacked[2][index, 0, :scales[2].shape[1], :scales[2].shape[0]].copy_(
                    scales[2].T, non_blocking=True)
            if g_idx[0] is not None:
                self.g_idx = g_idx[0]
        
    def apply_weights(self, x:torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        # output = _apply_delta(x, , output)
        return output

class RowParallelLinearWithDelta(BaseLayerWithDelta):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

    def create_delta_weights(self, max_deltas: int, delta_config: DeltaConfig,
                             model_config: PretrainedConfig) -> None:
        self.qls = [None] * max_deltas

    def reset_delta(self, index: int):
        self.qls[index] = None

    def set_delta(self, index: int, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, g_idx: torch.Tensor):
        # self.qls[index] = QuantLinear.from_tensors(
        #     qweight,
        #     qzeros,
        #     scales,
        #     g_idx,
        #     bias=None,
        # )
        pass

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x)
        output = _apply_delta(x, self.qls, output)
        return output

    def forward(self, input_):
        output_ = self.apply_weights(input_)
        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    @property
    def weight(self):
        return self.base_layer.weight


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

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.qls = [None] * max_deltas

    def set_delta(self, index: int, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, g_idx: torch.Tensor):
        self.qls[index] = QuantLinear.from_tensors(
            qweight,
            qzeros,
            scales,
            g_idx,
            bias=None,
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get the logits for the next tokens.
        # for simplicity, we don't consider the delta here yet
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
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None) -> BaseLayerWithDelta:
    supported_layer_types = {
        VocabParallelEmbedding: VocabParallelEmbeddingWithDelta,
        ColumnParallelLinear: ColumnParallelLinearWithDelta,
        QKVParallelLinear: MergedQKVParallelLinearWithDelta,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithDelta,
        RowParallelLinear: RowParallelLinearWithDelta,
    }
    for src_layer_type, delta_layer_type in supported_layer_types.items():
        if type(layer) is src_layer_type:  # pylint: disable=unidiomatic-typecheck
            ret = delta_layer_type(layer)
            ret.create_delta_weights(max_deltas, delta_config, model_config)
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
