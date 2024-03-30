# pylint: disable=unused-argument
import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, Set, Type
from transformers.configuration_utils import PretrainedConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_gather,
)
from .quant_linear import QuantLinear
from .config import DeltaConfig
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from .deltazip import add_delta_slice
import inspect
logger = init_logger(__name__)

TEST_BITS = 2
TEST_GROUPSIZE = 128

if TYPE_CHECKING:
    pass


def _apply_delta_packed_nslice(
    x: torch.Tensor,
    qweight_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qzeros_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scales_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    g_idx_stacked: List[torch.Tensor],
    indices: torch.Tensor,
    output: torch.Tensor,
    output_slices: Tuple[int, ...]
):
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    offset_left = 0
    for slice_idx in range(len(output_slices)):
        add_delta_slice(
            output, x,
            qweight_stacked[slice_idx],
            qzeros_stacked[slice_idx],
            scales_stacked[slice_idx],
            g_idx_stacked[slice_idx],
            indices,
            1.0,
            offset_left,
            output_slices[slice_idx])
        offset_left += output_slices[slice_idx]
    return output.view_as(org_output)


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

    def create_delta_weights(
        self, max_deltas: int, delta_config: DeltaConfig, model_config: PretrainedConfig
    ) -> None:
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

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is VocabParallelEmbedding


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
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:

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
            device=self.base_layer.weight.device,
        )
        self.qzeros_stacked = torch.zeros(
            max_deltas,
            1,
            scale_and_zero_size,
            self.output_size_per_partition // delta_config.pack_factor,
            dtype=torch.int32,
        )
        self.scales_stacked = torch.zeros(
            max_deltas,
            1,
            scale_and_zero_size,
            self.output_size_per_partition,
            dtype=torch.float16,
            device=self.base_layer.weight.device,
        )
        self.g_idx = torch.tensor(
            [i //
                self.quant_config.group_size for i in range(self.input_size)],
            dtype=torch.int32,
        )

    def set_delta(
        self,
        index: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
    ):
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

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias
        )
        output = _apply_delta(x, self.qls, output)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.apply_weights(x, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    @property
    def linear_weights(self):
        return self.base_layer.linear_weights

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ColumnParallelLinear or (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 1
        )


class MergedColumnParallelLinearWithDelta(ColumnParallelLinearWithDelta):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x)

    def reset_delta(self, index: int):
        self.qweight_stacked[0][index] = 0
        self.qweight_stacked[1][index] = 0

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        n_slices = 2
        if not (
            len(self.base_layer.output_sizes) == n_slices
            and self.base_layer.output_sizes[0] == self.base_layer.output_sizes[1]
        ):
            raise ValueError(
                "LoRAColumnParallelLinear2Slice requires 2 slices with "
                "the same size."
            )
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
                device=self.base_layer.weight.device,
            )
            for _ in range(n_slices)
        )


class QKVParallelLinearWithDelta(ColumnParallelLinearWithDelta):

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.q_proj_total_size = (
            self.base_layer.total_num_heads * self.base_layer.head_size
        )
        self.q_proj_shard_size = self.base_layer.num_heads * self.base_layer.head_size
        self.kv_proj_shard_size = (
            self.base_layer.num_kv_heads * self.base_layer.head_size
        )
        self.kv_proj_total_size = (
            self.base_layer.total_num_kv_heads * self.base_layer.head_size
        )

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = self.base_layer.num_heads * self.base_layer.head_size
        self.kv_proj_shard_size = (
            self.base_layer.num_kv_heads * self.base_layer.head_size
        )
        self.q_shard_id = tp_rank
        self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        self.qls: List[QuantLinear] = [None] * max_deltas

    def set_delta(
        self,
        index: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
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
                "QKVParallelLinearWithDelta only supports TP size 1"
            )

    def apply_weights(self, x: torch.Tensor, bias: Any | None) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias
        )
        output = _apply_delta(x, self.qls, output)
        return output

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is QKVParallelLinear and len(packed_modules_list) == 1


class MergedQKVParallelLinearWithDelta(ColumnParallelLinearWithDelta):
    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = self.base_layer.num_heads * self.base_layer.head_size
        self.kv_proj_shard_size = (
            self.base_layer.num_kv_heads * self.base_layer.head_size
        )
        self.q_shard_id = tp_rank
        self.pack_factor = delta_config.pack_factor
        self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        # qkv deltas
        self.qweight_stacked = (
            torch.zeros(
                max_deltas,
                1,
                self.q_proj_shard_size // delta_config.pack_factor,
                self.base_layer.weight.shape[1],
                dtype=delta_config.delta_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_deltas,
                1,
                self.q_proj_shard_size // delta_config.pack_factor,
                self.base_layer.weight.shape[1],
                dtype=delta_config.delta_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_deltas,
                1,
                self.q_proj_shard_size // delta_config.pack_factor,
                self.base_layer.weight.shape[1],
                dtype=delta_config.delta_dtype,
                device=self.base_layer.weight.device,
            ),
        )
        self.qzeros_stacked = (
            torch.zeros(
                max_deltas,
                1,
                1,
                self.base_layer.weight.shape[1] // delta_config.pack_factor,
                dtype=torch.int32,
            ),
            torch.zeros(
                max_deltas,
                1,
                1,
                self.base_layer.weight.shape[1] // delta_config.pack_factor,
                dtype=torch.int32,
            ),
            torch.zeros(
                max_deltas,
                1,
                1,
                self.base_layer.weight.shape[1] // delta_config.pack_factor,
                dtype=torch.int32,
            ),
        )
        self.scales_stacked = (
            torch.zeros(
                max_deltas,
                1,
                1,
                self.base_layer.weight.shape[1],
                dtype=torch.float16,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_deltas,
                1,
                1,
                self.base_layer.weight.shape[1],
                dtype=torch.float16,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_deltas,
                1,
                1,
                self.base_layer.weight.shape[1],
                dtype=torch.float16,
                device=self.base_layer.weight.device,
            )
        )
        self.g_idx_stacked = [
            torch.tensor(
                [i //
                    TEST_GROUPSIZE for i in range(self.base_layer.weight.shape[1])],
                dtype=torch.int32,
            ),
            torch.tensor(
                [i //
                    TEST_GROUPSIZE for i in range(self.base_layer.weight.shape[1])],
                dtype=torch.int32,
            ),
            torch.tensor(
                [i //
                    TEST_GROUPSIZE for i in range(self.base_layer.weight.shape[1])],
                dtype=torch.int32,
            )
        ]
        self.output_slices = (
            self.q_proj_shard_size,
            self.kv_proj_shard_size,
            self.kv_proj_shard_size,
        )
        self.packed_indices: Optional[torch.Tensor] = None
        self.standard_indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None

    def set_delta(
        self,
        index: int,
        qweight: List[torch.Tensor],
        qzeros: List[torch.Tensor],
        scales: List[torch.Tensor],
        g_idx: List[torch.Tensor],
    ):
        if self.tp_size > 1:
            pass
        else:
            if qweight[0] is not None:
                self.qweight_stacked[0][
                    index, 0, : qweight[0].shape[0], : qweight[0].shape[1]
                ].copy_(qweight[0], non_blocking=True)

            if qweight[1] is not None:
                self.qweight_stacked[1][
                    index, 0, : qweight[1].shape[0], : qweight[1].shape[1]
                ].copy_(qweight[1], non_blocking=True)

            if qweight[2] is not None:
                self.qweight_stacked[2][
                    index, 0, : qweight[2].shape[0], : qweight[2].shape[1]
                ].copy_(qweight[2], non_blocking=True)

            if qzeros[0] is not None:
                self.qzeros_stacked[0][
                    index, 0, : qzeros[0].shape[0], : qzeros[0].shape[1]
                ].copy_(qzeros[0], non_blocking=True)
            if qzeros[1] is not None:
                self.qzeros_stacked[1][
                    index, 0, : qzeros[1].shape[0], : qzeros[1].shape[1]
                ].copy_(qzeros[1], non_blocking=True)
            if qzeros[2] is not None:
                self.qzeros_stacked[2][
                    index, 0, : qzeros[2].shape[0], : qzeros[2].shape[1]
                ].copy_(qzeros[2], non_blocking=True)

            if scales[0] is not None:
                self.scales_stacked[0][
                    index, 0, : scales[0].shape[0], : scales[0].shape[1]
                ].copy_(scales[0], non_blocking=True)
            if scales[1] is not None:
                self.scales_stacked[1][
                    index, 0, : scales[1].shape[0], : scales[1].shape[1]
                ].copy_(scales[1], non_blocking=True)
            if scales[2] is not None:
                self.scales_stacked[2][
                    index, 0, : scales[2].shape[0], : scales[2].shape[1]
                ].copy_(scales[2], non_blocking=True)

            if g_idx[0] is not None:
                self.g_idx_stacked[0] = g_idx[0]
            if g_idx[1] is not None:
                self.g_idx_stacked[1] = g_idx[1]
            if g_idx[2] is not None:
                self.g_idx_stacked[2] = g_idx[2]

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias
        )
        logger.info(f"output shape: {output.shape}")
        # TODO(xiaozhe): fix the bug
        # self.indices = torch.zeros_like(self.indices)
        _apply_delta_packed_nslice(
            x,
            self.qweight_stacked,
            self.qzeros_stacked,
            self.scales_stacked,
            self.g_idx_stacked,
            self.indices[: self.indices_len[0]],
            output,
            self.output_slices,
        )
        return output

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is QKVParallelLinear and len(packed_modules_list) == 3


class RowParallelLinearWithDelta(BaseLayerWithDelta):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

    def create_delta_weights(
        self, max_deltas: int, delta_config: DeltaConfig, model_config: PretrainedConfig
    ) -> None:
        self.qls = [None] * max_deltas

    def reset_delta(self, index: int):
        self.qls[index] = None

    def set_delta(
        self,
        index: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
    ):
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
            self.base_layer.linear_weights, x
        )
        # TODO(xiaozhe): apply delta here
        # output = _apply_delta(x, self.qls, output)
        return output

    def forward(self, input_):
        output_ = self.apply_weights(input_)
        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    @property
    def weight(self):
        return self.base_layer.weight

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is RowParallelLinear

class LogitsProcessorWithDelta(BaseLayerWithDelta):

    def __init__(
        self,
        base_layer: LogitsProcessor,
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
    def logits_as_input(self):
        return self.base_layer.logits_as_input

    @property
    def vocab_size(self):
        return self.base_layer.vocab_size

    @property
    def scale(self):
        return self.base_layer.scale

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
        # Keep this in sync with csrc/punica/bgmv/bgmv_config.h
        pass
        self.indices = None
        self.indices_padded = None
        self.indices_len = None

    def reset_delta(self, index: int):
        pass

    def set_delta(
        self,
        index: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
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
        self.indices = sampler_indices
        self.indices_padded = sampler_indices_padded
        self.indices_len = indices_len

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_gather(logits)
        if logits is None:
            return None

        return logits

    def forward(self, *args, **kwargs):
        return type(self.base_layer).forward(self, *args, **kwargs)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        delta_config: DeltaConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # Special handling for the LogitsProcessor.
        return False

_all_delta_classes: Set[Type[BaseLayerWithDelta]] = {
    cls
    for cls in globals().values()
    if inspect.isclass(cls)
    and issubclass(cls, BaseLayerWithDelta)
    and cls is not BaseLayerWithDelta
}

def from_layer(
    layer: nn.Module,
    max_deltas: int,
    delta_config: DeltaConfig,
    packed_modules_list: List,
    model_config: Optional[PretrainedConfig] = None,
) -> nn.Module:
    for delta_cls in _all_delta_classes:
        if delta_cls.can_replace_layer(
            layer, delta_config, packed_modules_list, model_config
        ):
            ret = delta_cls(layer)
            ret.create_delta_weights(max_deltas, delta_config, model_config)
            return ret
    return layer

def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_deltas: int,
    delta_config: DeltaConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithDelta:
    ret = LogitsProcessorWithDelta(
        layer, lm_head.embedding_dim, lm_head.weight.dtype, lm_head.weight.device
    )
    ret.create_lora_weights(max_deltas, delta_config, model_config)
    return ret
