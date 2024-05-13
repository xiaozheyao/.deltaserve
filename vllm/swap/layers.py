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
        self, max_packed_model: int, swap_config: SwapConfig, model_config: PretrainedConfig
    ) -> None:
        ...
    
    def reset_pack(self, index: int):
        ...
    
    def set_pack(
        self,
        index: int,
        bitwidth: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        ...
    
    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        ...
        

class VocabParallelEmbeddingWithPacked(BaseLayerWithPacked):
    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.vocab_start_index = self.base_layer.vocab_start_index
        self.vocab_end_index = self.base_layer.vocab_end_index
    
    def reset_packed(self, index: int):
        self.packed_weights[index] = None
    
    def create_packed_weights(
        self,
        max_packed: int,
        swap_config: SwapConfig,
        model_config: PretrainedConfig
    ) -> None:
        self.packed_weights = torch.zeros(
            max_packed,
            self.base_layer.org_vocab_size // self.tp_size,
            self.base_layer.embedding_dim,
            dtype=model_config.torch_dtype,
            device=self.base_layer.weight.device
        )
    def set_packed(
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
        # TODO(xiaozhe): 

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
        self.tp_size = get_tensor_model_parallel_world_size()
    
    def reset_pack(self, index: int):
        self.packed_weights[index] = None
    
    def create_pack_weights(
        self,
        max_packed: int,
        swap_config: SwapConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.weights_packed = torch.zeros(
            max_packed,
            self.base_layer.weight.shape[0],
            self.base_layer.weight.shape[1],
            dtype=model_config.torch_dtype,
            device=self.base_layer.weight.device,
        )
        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None
        self.output_dim = self.base_layer.weight.shape[0]
    
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


# def from_layer(
#     layer: nn.Module,
#     max_deltas: int,
#     model_config: ModelConfig,
#     packed_modules_list: List,
#     model_config: Optional[PretrainedConfig] = None,
# ) -> nn.Module:
#     for delta_cls in _all_delta_classes:
#         if delta_cls.can_replace_layer(
#             layer, delta_config, packed_modules_list, model_config
#         ):
#             ret = delta_cls(layer)
#             ret.create_delta_weights(max_deltas, delta_config, model_config)
#             return ret
#     return layer