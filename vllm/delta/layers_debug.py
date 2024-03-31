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
from vllm.model_executor.parallel_utils.utils import split_tensor_along_last_dim

from .quant_linear import QuantLinear
from .config import DeltaConfig
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from .deltazip import add_delta_slice, add_delta
import inspect

logger = init_logger(__name__)


if TYPE_CHECKING:
    pass


def _apply_delta(
    x: torch.Tensor,
    qweight_stacked: torch.Tensor,
    qzeros_stacked: torch.Tensor,
    scales_stacked: torch.Tensor,
    g_idx_stacked: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
):
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    add_delta(
        output,
        x,
        qweight_stacked,
        qzeros_stacked,
        scales_stacked,
        g_idx_stacked,
        indices,
        1.0,
    )
    return output.view_as(org_output)


def _apply_delta_packed_nslice(
    x: torch.Tensor,
    qweight_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qzeros_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scales_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    g_idx_stacked: List[torch.Tensor],
    indices: torch.Tensor,
    output: torch.Tensor,
    output_slices: Tuple[int, ...],
    debug: bool = False,
):
    """
    Applies delta to each input.
    This method applies all deltas to each input. It uses the
    indices vector to determine which delta yields the
    correct output. An index of -1 means no delta should be
    applied. This method adds the final delta results to the
    output.

    This method is used for layers that are composed of multiple sublayers
    (slices) packed together.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        qweight_stacked:    3 element tuple of (num_deltas, 1,  hidden_dim/pack_factor, hidden_dim)
        qzeros_stacked:     3 element tuple of (num_deltas, 1, 1, hidden_dim/pack_factor)
        indices:           (batch_size)
        output:            (batch_size, q_slice_size + 2*kv_slice_size)
        output_slices:     n-1 element tuple of (slice_size...),
                           where n is number of slices
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    indices = indices.view(-1)
    offset_left = 0
    for slice_idx in range(len(output_slices)):
        add_delta_slice(
            output,
            x,
            qweight_stacked[slice_idx],
            qzeros_stacked[slice_idx],
            scales_stacked[slice_idx],
            g_idx_stacked[slice_idx],
            indices,
            1.0,
            offset_left,
            output_slices[slice_idx],
            debug=debug,
        )
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
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer.forward(x)

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
        pass

    def set_delta(
        self,
        index: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
    ):
        self.reset_delta(index)
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

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # (note): this is not actually used.
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias
        )
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("ColumnParallelLinearWithDelta forward")
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

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        pass

    def reset_delta(self, index: int):
        pass

    def set_delta(
        self,
        index: int,
        qweight: List[torch.Tensor],
        qzeros: List[torch.Tensor],
        scales: List[torch.Tensor],
        g_idx: List[torch.Tensor],
    ):
        pass

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias
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
        return (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 2
        )


class QKVParallelLinearWithDelta(ColumnParallelLinearWithDelta):
    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
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

    def apply_weights(self, x: torch.Tensor, bias: Any | None) -> torch.Tensor:
        # (note): this is not actually used
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias
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
        pass

    def reset_delta(self, index: int):
        pass

    def set_delta(
        self,
        index: int,
        qweight: List[torch.Tensor],
        qzeros: List[torch.Tensor],
        scales: List[torch.Tensor],
        g_idx: List[torch.Tensor],
    ):
        pass

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias
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

    def create_delta_weights(
        self,
        max_deltas: int,
        delta_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        pass

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

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x
        )
        return output

    def forward(self, input_):
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.apply_weights(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel
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
    ret.create_delta_weights(max_deltas, delta_config, model_config)
    return ret
