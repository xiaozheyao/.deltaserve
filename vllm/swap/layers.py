# pylint: disable=unused-argument
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, Set, Type
from transformers.configuration_utils import PretrainedConfig
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
from .config import SwapConfig
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from .ops import apply_swap_embed, apply_swap_packed_nslice, apply_swap

ASYNC_COPY = True
logger = init_logger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class ModelMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)

    def __str__(self):
        return f"index_mapping: {self.index_mapping}, prompt_mapping: {self.prompt_mapping}"


class BaseLayerWithPacked(nn.Module):
    def create_packed_weights(
        self,
        max_packed_model: int,
        swap_config: SwapConfig,
        model_config: PretrainedConfig,
    ) -> None: ...

    def reset_pack(self, index: int): ...

    def set_pack(
        self,
        index: int,
        bitwidth: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ): ...

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ): ...

    def clear_base(self):
        if hasattr(self, "base_layer"):
            self.base_layer = None
        else:
            logger.warning(
                f"Trying to clear base layer from {self} but no base layer found."
            )


class VocabParallelEmbeddingWithPacked(BaseLayerWithPacked):
    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.vocab_start_index = self.base_layer.vocab_start_index
        self.vocab_end_index = self.base_layer.vocab_end_index
        self.embedding_dim = self.base_layer.embedding_dim
        self.vocab_size = self.base_layer.org_vocab_size // self.tp_size
        self.device = self.base_layer.weight.device

    def reset_pack(self, index: int):
        self.packed_weights[index] = None

    def create_packed_weights(
        self, max_packed: int, swap_config: SwapConfig, model_config: PretrainedConfig
    ) -> None:
        self.packed_weights = torch.zeros(
            max_packed,
            self.vocab_size,
            self.embedding_dim,
            dtype=model_config.torch_dtype,
            device=self.device,
        )

    def set_pack(
        self,
        index: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        self.packed_weights[index].copy_(weight, non_blocking=ASYNC_COPY)

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
        indices = self.indices[: self.indices_len[0]]
        if self.tp_size > 1:
            input_mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)
            # mask the input
            masked_input = x.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = x
        outputs = torch.zeros(
            masked_input.shape[0],
            self.embedding_dim,
            device=masked_input.device,
            dtype=torch.float16,
        )
        output_parallel = apply_swap_embed(
            masked_input,
            self.packed_weights,
            indices,
            outputs,
        )
        if self.tp_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        swap_config: SwapConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is VocabParallelEmbedding


class ColumnParallelLinearWithPacked(BaseLayerWithPacked):
    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.in_features = self.base_layer.weight.shape[1]
        self.out_features = self.base_layer.weight.shape[0]
        self.tp_size = get_tensor_model_parallel_world_size()
        self.device = self.device

    def reset_pack(self, index: int):
        self.packed_weights[index] = None

    def create_packed_weights(
        self,
        max_packed: int,
        swap_config: SwapConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.weights_packed = torch.zeros(
            max_packed,
            self.out_features,
            self.in_features,
            dtype=model_config.torch_dtype,
            device=self.device,
        )
        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None

    def set_pack(
        self,
        index: int,
        weight: torch.Tensor,
    ):
        self.reset_pack(index)
        self.weights_packed[index, :, :].copy_(weight, non_blocking=ASYNC_COPY)

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
        self, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # TODO(xiaozhe):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.apply_weights(x, bias)
        if self.base_layer.gather_output:
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
        swap_config: SwapConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ColumnParallelLinear or (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 1
        )


class MergedColumnParallelLinearWithPacked(ColumnParallelLinearWithPacked):
    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = self.base_layer.weight.device
        self.in_features = self.base_layer.weight.shape[1]
        self.out_features = self.base_layer.weight.shape[0]
        self.output_sizes = self.base_layer.output_sizes
        self.output_dim = self.out_features // 2

    def create_packed_weights(
        self,
        max_packed: int,
        swap_config: SwapConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        n_slices = 2
        if not (
            len(self.output_sizes) == n_slices
            and self.output_sizes[0] == self.output_sizes[1]
        ):
            raise ValueError(
                "MergedColumnParallelLinearWithPacked requires 2 slices with "
                "the same size."
            )
        self.weight_stacked = tuple(
            torch.zeros(
                max_packed,
                self.output_dim,
                self.in_features,
                dtype=model_config.torch_dtype,
                device=self.device,
            )
            for _ in range(n_slices)
        )
        self.indices: Optional[torch.Tensor] = None

    def reset_pack(self, index: int):
        self.weight_stacked[0][index] = 0
        self.weight_stacked[1][index] = 0

    def set_pack(
        self,
        index: int,
        weight: List[torch.Tensor],
    ):
        self.reset_pack(index)
        if weight[0] is not None:
            self.weight_stacked[0][index, :, :].copy_(
                weight[0], non_blocking=ASYNC_COPY
            )
        if weight[1] is not None:
            self.weight_stacked[1][index, :, :].copy_(
                weight[1], non_blocking=ASYNC_COPY
            )

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # TODO(xiaozhe): bias is ignored for now.
        outputs = torch.zeros(
            x.shape[0],
            self.out_features,
            device=x.device,
            dtype=x.dtype,
        )
        outputs = apply_swap_packed_nslice(
            x,
            self.weight_stacked,
            self.indices[: self.indices_len[0]],
            outputs,
            (self.output_dim, self.output_dim),
        )
        return outputs

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        swap_config: SwapConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 2
        )


class MergedQKVParallelLinearWithPacked(ColumnParallelLinearWithPacked):
    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.q_proj_shard_size = self.base_layer.num_heads * self.base_layer.head_size
        self.kv_proj_shard_size = (
            self.base_layer.num_kv_heads * self.base_layer.head_size
        )
        self.q_shard_id = self.tp_rank
        self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas
        self.output_size = self.base_layer.weight.shape[0]
        self.output_slices = (
            self.q_proj_shard_size,
            self.kv_proj_shard_size,
            self.kv_proj_shard_size,
        )
        self.in_features = self.base_layer.weight.shape[1]
        self.out_features = self.base_layer.weight.shape[0]
        self.device = self.base_layer.weight.device
        self.skip_bias_add = self.base_layer.skip_bias_add
        self.bias = self.base_layer.bias

    def create_packed_weights(
        self,
        max_packed_model: int,
        swap_config: SwapConfig,
        model_config: PretrainedConfig,
    ) -> None:
        self.weight_stacked = (
            torch.zeros(
                max_packed_model,
                self.q_proj_shard_size,
                self.in_feautres,
                dtype=model_config.torch_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_packed_model,
                self.kv_proj_shard_size,
                self.in_features,
                dtype=model_config.torch_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_packed_model,
                self.kv_proj_shard_size,
                self.in_features,
                dtype=model_config.torch_dtype,
                device=self.base_layer.weight.device,
            ),
        )
        self.packed_indices: Optional[torch.Tensor] = None
        self.standard_indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None

    def reset_pack(self, index: int):
        self.weight_stacked[0][index] = 0
        self.weight_stacked[1][index] = 0
        self.weight_stacked[2][index] = 0

    def set_pack(
        self,
        index: int,
        weight: List[torch.Tensor],
    ):
        self.reset_pack(index)
        self.weight_stacked[0][index, :, :].copy_(weight[0], non_blocking=ASYNC_COPY)
        self.weight_stacked[1][index, :, :].copy_(weight[1], non_blocking=ASYNC_COPY)
        self.weight_stacked[2][index, :, :].copy_(weight[2], non_blocking=ASYNC_COPY)

    def apply_weights(
        self, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = torch.zeros(
            x.shape[0],
            self.output_size,
            device=x.device,
            dtype=x.dtype,
        )
        output = apply_swap_packed_nslice(
            x,
            self.weight_stacked,
            self.indices[: self.indices_len[0]],
            output,
            self.output_slices,
        )
        return output

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        swap_config: SwapConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is QKVParallelLinear and len(packed_modules_list) == 3


class RowParallelLinearWithPacked(BaseLayerWithPacked):
    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = self.base_layer.weight.device
        self.in_features = self.base_layer.weight.shape[1]
        self.out_features = self.base_layer.weight.shape[0]
        self.input_is_parallel = self.base_layer.input_is_parallel
        self.reduce_results = self.base_layer.reduce_results

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

    def create_packed_weights(
        self,
        max_packed: int,
        swap_config: SwapConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.weight_stacked = torch.zeros(
            (
                max_packed,
                self.out_features,
                self.in_features,
            ),
            dtype=model_config.torch_dtype,
            device=self.device,
        )

    def reset_pack(self, index: int):
        self.weight_stacked[index] = 0

    def set_pack(
        self,
        index: int,
        weight: torch.Tensor,
    ):
        self.reset_pack(index)

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            x.shape[0],
            self.out_features,
            device=x.device,
            dtype=x.dtype,
        )
        output = apply_swap(
            x,
            self.weight_stacked,
            self.indices[: self.indices_len[0]],
        )
        return output

    def forward(self, input_):
        # TODO(xiaozhe):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()

        output_parallel = self.apply_weights(input_parallel)
        if self.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    @property
    def weight(self):
        return self.base_layer.weight

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        swap_config: SwapConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is RowParallelLinear


class LogitsProcessorWithPacked(BaseLayerWithPacked):
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
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

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

    def create_packed_weights(
        self,
        max_deltas: int,
        swap_config: SwapConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.weight_stacked = torch.zeros(
            (
                max_deltas,
                self.base_layer.vocab_size // self.tp_size,
                self.hidden_size,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self.indices = None
        self.indices_padded = None
        self.indices_len = None

    def reset_pack(self, index: int):
        self.weight_stacked[index] = 0

    def set_pack(
        self,
        index: int,
        weight: torch.Tensor,
    ):
        self.reset_delta(index)
        self.weight_stacked[index, :, :].copy_(weight, non_blocking=ASYNC_COPY)

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
        # TODO(xiaozhe): for now we assume there's no additional token added, so this simply performs additional matmuls on delta.
        if logits is None:
            return None
        # (todo)
        logits = tensor_model_parallel_gather(logits)
        return logits

    def forward(self, *args, **kwargs):
        return type(self.base_layer).forward(self, *args, **kwargs)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        swap_config: SwapConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # Special handling for the LogitsProcessor.
        return False


_all_swap_classes: Set[Type[BaseLayerWithPacked]] = {
    cls
    for cls in globals().values()
    if inspect.isclass(cls)
    and issubclass(cls, BaseLayerWithPacked)
    and cls is not BaseLayerWithPacked
}


def from_layer(
    layer: nn.Module,
    max_packed_model: int,
    swap_config: SwapConfig,
    packed_modules_list: List,
    model_config: Optional[PretrainedConfig] = None,
) -> nn.Module:
    for swap_cls in _all_swap_classes:
        if swap_cls.can_replace_layer(
            layer, swap_config, packed_modules_list, model_config
        ):
            ret = swap_cls(layer)
            ret.create_packed_weights(max_packed_model, swap_config, model_config)
            return ret
    return layer


def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_packs: int,
    swap_config: SwapConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithPacked:
    ret = LogitsProcessorWithPacked(
        layer, lm_head.embedding_dim, lm_head.weight.dtype, lm_head.weight.device
    )
    ret.create_packed_weights(max_packs, swap_config, model_config)
    return ret
