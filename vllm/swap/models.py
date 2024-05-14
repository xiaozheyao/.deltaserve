import os
import re
import math
import copy
import json
import torch
import cupy as cp
import torch.nn as nn
from typing import Dict, Optional, List, Callable, Hashable, Any, Type, Tuple
from timeit import default_timer as timer
import transformers
from transformers import AutoConfig
from vllm.logger import init_logger
from vllm.utils import LRUCache, in_wsl, total_bytes_count
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
)
from safetensors import safe_open
from .layers import ModelMapping, from_layer
from .packed import ModelLayerWeights
from .config import SwapConfig
from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.model_loader import (
    _get_model_architecture,
    _set_default_torch_dtype,
)
from .utils import replace_submodule


_GLOBAL_MODEL_ID = 0


def convert_mapping(
    mapping: ModelMapping,
    model_index_to_id: List[Optional[int]],
    max_models: int,
    vocab_size: int,
    extra_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Converts ModelMapping to index tensors.

    Args:
        mapping: ModelMapping mapping rows in a batch to model ids.
        delta_index_to_id: List mapping Delta ids to Delta indices.
        max_deltas: Maximum number of Deltas.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each Delta can have.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                Delta indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                Delta indices for sampler. For generation, this will be the
                same as base_indicies. For prefill, this will map requests
                to Delta indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to Delta indices for sampler with padding.
                Same as sampler_indicies, but -1 is replaced with
                max_deltas.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the Deltas, second row is for the Delta.emb
                embeddings.
            indices_len: List of lengths of the above tensors.
    """
    indices = list(mapping.index_mapping).copy()
    embedding_indices = indices.copy()
    model_indices = indices.copy()
    prompt_mapping = [
        model_index_to_id.index(x) if x > 0 else -1 for x in mapping.prompt_mapping
    ]
    delta_idx = None
    for i in range(len(indices)):
        # TODO index can be slow. optimize
        delta_idx = model_index_to_id.index(indices[i]) if indices[i] > 0 else -1
        embedding_indices[i] = delta_idx if indices[i] > 0 else 0
        indices[i] = i
        model_indices[i] = delta_idx

    indices = torch.tensor(
        [indices, model_indices, embedding_indices], dtype=torch.long, device="cuda"
    )
    prompt_mapping = torch.tensor(prompt_mapping, device="cuda", dtype=torch.long)
    embeddings_indices = torch.stack(
        [indices[2] * extra_vocab_size, indices[2] * (vocab_size + extra_vocab_size)]
    )
    embeddings_indices[embeddings_indices == -1] = max_models - 1
    base_indices = indices[1]
    sampler_indices = prompt_mapping
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded == -1] = max_models - 1
    sampler_indices_padded = torch.arange(
        0, len(sampler_indices_padded), device="cuda", dtype=torch.long
    ) + (sampler_indices_padded * len(sampler_indices_padded))
    indices_len = (
        base_indices.shape[-1],
        sampler_indices.shape[-1],
        sampler_indices_padded.shape[-1],
        embeddings_indices.shape[-1],
    )

    return (
        base_indices,
        sampler_indices,
        sampler_indices_padded,
        embeddings_indices,
        indices_len,
    )


def get_model_id():
    global _GLOBAL_MODEL_ID
    _GLOBAL_MODEL_ID += 1
    return _GLOBAL_MODEL_ID


class SwapModel:
    def __init__(self, swap_model_id: int, swaps: Dict[str, ModelLayerWeights]):
        self.id = swap_model_id
        self.swaps: Dict[str, ModelLayerWeights] = swaps

    def get_swap(self, module_name: str) -> Optional[ModelLayerWeights]:
        return self.swaps.get(module_name, None)

    @classmethod
    def from_checkpoint(
        cls,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        trust_remote_code: bool = False,
    ):
        model_class = _get_model_architecture(model_config)
        with _set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = model_class(model_config.hf_config)
                model.load_weights(
                    model_config.model,
                    model_config.download_dir,
                    model_config.load_format,
                    model_config.revision,
                )
        return model.eval()


class SwapModelManager:
    """A manager that manages multiple SwapModels."""

    def __init__(
        self,
        model,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        swap_config: SwapConfig,
        model_config: ModelConfig,
    ):
        self.model_config = model_config
        self.max_num_seqs = max_num_seqs
        self.swap_config = swap_config
        assert (
            self.capacity >= self.packed_swap_slots
        ), "Capacity must be greater than packed swap slots"
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.delta_index_to_id: List[Optional[int]] = [None] * self.delta_slots
        self.vocab_size = vocab_size
        self.base_indices = torch.empty(
            self.max_num_batched_tokens, dtype=torch.long, device="cuda"
        )
        self.sampler_indices = torch.empty(
            self.max_num_batched_tokens, dtype=torch.long, device="cuda"
        )
        self.sampler_indices_padded = torch.empty(
            self.max_num_batched_tokens, dtype=torch.long, device="cuda"
        )
        self.embeddings_indices = torch.empty(
            2, self.max_num_batched_tokens, dtype=torch.long, device="cuda"
        )
        self.offset = []
        self.indices_len = []
        self.model = model
        if hasattr(self.model, "supported_swap_modules"):
            self.supported_swap_modules = copy.deepcopy(
                self.model.supported_swap_modules
            )

            self.packed_modules_mapping = copy.deepcopy(
                self.model.packed_modules_mapping
            )
        self.packed_modules: Dict[str, List[str]] = {}
        self.modules: Dict[str, "ModelLayerWeights"] = {}
        self._registered_swaps: Dict[str, "SwapModel"] = {}
        self._activate_swaps: Dict[int, None] = {}
        self._last_mapping = None
        self._create_swap_modules()
        self.model.delta_manager = self

    @property
    def packed_swap_slots(self) -> int:
        return self.swap_config.max_packed_model

    @property
    def capacity(self) -> int:
        return self.swap_config.max_cpu_model

    def __len__(self) -> int:
        return len(self.registered_models)

    def _create_swap_modules(self):
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name):
                continue
            parts = module_name.split(".")[-1]
            packed_moduled_list = self.packed_modules_mapping.get(parts, [])
            new_module = replace_submodule(
                self.model,
                module_name,
                from_layer(
                    module,
                    self.delta_slots,
                    self.delta_config,
                    packed_moduled_list,
                    self.model.config,
                ),
            )

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module), module_name
            )
            or target_module == module_name
            for target_module in self.supported_delta_modules
        )
