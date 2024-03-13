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
from .config import DeltaConfig

if TYPE_CHECKING:
    pass

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

def from_layer(
        layer: nn.Module,
        max_loras: int,
        lora_config: DeltaConfig,
        model_config: Optional[PretrainedConfig] = None) -> BaseLayerWithDelta:
    supported_layer_types = {
        # VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        # ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        # QKVParallelLinear: QKVParallelLinearWithDelta,
        # MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        # RowParallelLinear: RowParallelLinearWithLoRA,
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
) -> SamplerWithLoRA:
    ret = SamplerWithLoRA(layer, lm_head.embedding_dim, lm_head.weight.dtype,
                          lm_head.weight.device)
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret
