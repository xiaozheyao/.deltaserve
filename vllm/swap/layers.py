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