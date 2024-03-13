import re
import math
import copy
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Callable, Hashable, Any, Type, Tuple

from vllm.delta.delta import DeltaLayerWeights, PackedDeltaLayerWeights
from vllm.delta.config import DeltaConfig
from vllm.delta.layers import BaseLayerWithDelta, from_layer, from_layer_sampler, DeltaMapping
from vllm.delta.utils import replace_submodule
from vllm.logger import init_logger
from vllm.utils import LRUCache, in_wsl

logger = init_logger(__name__)

_GLOBAL_DELTA_ID = 0

def convert_mapping(
    mapping: DeltaMapping, delta_index_to_id: List[Optional[int]],
    max_deltas: int, vocab_size: int, extra_vocab_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Converts DeltaMapping to index tensors.

    Args:
        mapping: DeltaMapping mapping rows in a batch to Delta ids.
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
    delta_indices = indices.copy()
    prompt_mapping = [
        delta_index_to_id.index(x) if x > 0 else -1
        for x in mapping.prompt_mapping
    ]
    delta_idx = None
    for i in range(len(indices)):
        # TODO index can be slow. optimize
        delta_idx = (delta_index_to_id.index(indices[i])
                    if indices[i] > 0 else -1)
        embedding_indices[i] = delta_idx if indices[i] > 0 else 0
        indices[i] = i
        delta_indices[i] = delta_idx

    indices = torch.tensor([indices, delta_indices, embedding_indices],
                           dtype=torch.long,
                           device="cuda")
    prompt_mapping = torch.tensor(prompt_mapping,
                                  device="cuda",
                                  dtype=torch.long)
    embeddings_indices = torch.stack([
        indices[2] * extra_vocab_size,
        indices[2] * (vocab_size + extra_vocab_size)
    ])
    embeddings_indices[embeddings_indices == -1] = max_deltas - 1
    base_indices = indices[1]
    sampler_indices = prompt_mapping
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded == -1] = max_deltas - 1
    sampler_indices_padded = (
        torch.arange(
            0, len(sampler_indices_padded), device="cuda", dtype=torch.long) +
        (sampler_indices_padded * len(sampler_indices_padded)))
    indices_len = (base_indices.shape[-1], sampler_indices.shape[-1],
                   sampler_indices_padded.shape[-1],
                   embeddings_indices.shape[-1])

    return (base_indices, sampler_indices, sampler_indices_padded,
            embeddings_indices, indices_len)

def get_delta_id():
    global _GLOBAL_DELTA_ID
    _GLOBAL_DELTA_ID += 1
    return _GLOBAL_DELTA_ID


class DeltaModel:
    """A delta model compressed from the fine-tuned variant"""

    def __init__(
        self,
        delta_model_id: int,
        deltas: Dict[str, DeltaLayerWeights],
    ):
        self.id = delta_model_id
        self.deltas: Dict[str, DeltaLayerWeights] = deltas

    def get_delta(self, module_name: str) -> Optional[DeltaLayerWeights]:
        return self.deltas.get(module_name, None)

    @classmethod
    def from_checkpoint(cls):
        raise NotImplementedError


class DeltaModelManager:
    """A manager that manages multiple full-fine-tuned models."""

    def __init__(
        self,
        model,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        delta_config: DeltaConfig,
    ) -> None:
        self.delta_config = delta_config
        self.max_num_seqs = max_num_seqs
        assert (
            self.capacity >= self.delta_slots
        ), "capacity must be greater than delta_slots"
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.delta_index_to_id: List[Optional[int]] = [None] * self.delta_slots
        self.vocab_size = vocab_size
        self.offset = []
        # todo(xiaozhe): figure out if we want to pre-define the length
        # below are dummpy for now
        self.indices_len = []
        self.model = model
        if hasattr(self.model, "supported_delta_modules"):
            self.supported_delta_modules = copy.deepcopy(
                self.model.supported_delta_modules
            )

        self.modules: Dict[str, "BaseLayerWithDelta"] = {}
        self._registered_deltas: Dict[int, DeltaModel] = {}
        self._active_deltas: Dict[int, None] = {}
        self._last_mapping = None
        self._create_delta_modules()
        self.model.delta_manager = self

    @property
    def capacity(self) -> int:
        return self.delta_config.max_cpu_deltas

    @property
    def delta_slots(self) -> int:
        return self.delta_config.max_deltas

    def __len__(self) -> int:
        return len(self._registered_deltas)

    def activate_delta(self, delta_id: int):
        """Move delta into GPU buffer to be used in the forward pass"""
        if delta_id in self._active_deltas:
            return False
        first_free_slot = next(
            (
                (i, delta_id)
                for i, delta_id in enumerate(self.delta_index_to_id)
                if delta_id is None
            ),
            None,
        )
        if first_free_slot is None:
            raise ValueError("No free delta slots")
        index, _ = first_free_slot
        self._active_deltas[delta_id] = None
        delta_model = self._registered_deltas[delta_id]
        logger.debug(f"Activating Delta. int id: {delta_model.id}, slot index: {index}")
        self.delta_index_to_id[index] = delta_model.id
        for module_name, module in self.modules.items():
            module_delta = delta_model.get_delta(module_name)
            if module_delta:
                # module_delta.optimize()
                module.set_delta(index, module_delta)
            else:
                module.reset_delta(index)
        return True

    def _deactivate_delta(self, delta_id: int):
        try:
            index = self.delta_index_to_id.index(delta_id)
            self.delta_index_to_id[index] = None
        except ValueError:
            pass

    def deactivate_delta(self, delta_id: int) -> bool:
        """Remove a delta from a GPU buffer."""
        if delta_id in self._active_deltas:
            self._deactivate_delta(delta_id)
            self._active_deltas.pop(delta_id)
            return True
        return False

    def _add_delta(self, delta: DeltaModel) -> bool:
        self._create_merged_deltas_inplace(delta)
        self._registered_deltas[delta.id] = delta

    def add_delta(self, delta: DeltaModel) -> bool:
        """Add a DeltaModel to the manager CPU cache."""
        if delta.id not in self._registered_deltas:
            if len(self._registered_deltas) >= self.capacity:
                raise RuntimeError("No free Delta slots.")
            self._add_delta(delta)
            return True
        return False

    def remove_delta(self, delta_id: int) -> bool:
        """Remove a DeltaModel from the manager CPU cache."""
        # TODO: should we check active delta?
        self.deactivate_delta(delta_id)
        return bool(self._registered_deltas.pop(delta_id, None))

    # TODO see if this can be vectorized
    def _set_delta_mapping(self, mapping: DeltaMapping) -> None:
        (
            base_indices,
            sampler_indices,
            sampler_indices_padded,
            embeddings_indices,
            indices_len,
        ) = convert_mapping(
            mapping,
            self.delta_index_to_id,
            self.delta_slots + 1,
            self.vocab_size,
            self.delta_config.delta_extra_vocab_size,
        )
        self.base_indices[: base_indices.shape[0]].copy_(base_indices)
        self.sampler_indices[: sampler_indices.shape[0]].copy_(sampler_indices)
        self.sampler_indices_padded[: sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded
        )
        self.embeddings_indices[
            : embeddings_indices.shape[0], : embeddings_indices.shape[1]
        ].copy_(embeddings_indices)
        # Maintain the reference
        self.indices_len[:] = indices_len

    def set_delta_mapping(self, delta_mapping: DeltaMapping) -> None:
        if self._last_mapping != delta_mapping:
            self._set_delta_mapping(delta_mapping)
        self._last_mapping = delta_mapping

    def list_deltas(self) -> Dict[int, DeltaModel]:
        """List all registered DeltaModels."""
        return dict(self._registered_deltas)

    def get_delta(self, delta_id: int) -> Optional[DeltaModel]:
        return self._registered_deltas.get(delta_id, None)

    def remove_all_deltas(self) -> bool:
        """Remove all DeltaModels from the manager."""
        self._registered_deltas.clear()
        self.delta_index_to_id = [None] * self.delta_slots
        self._active_deltas.clear()

    def _create_delta_modules(self):
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name):
                continue

            new_module = replace_submodule(
                self.model,
                module_name,
                from_layer(
                    module, self.delta_slots, self.delta_config, self.model.config
                ),
            )
            # (yard1): TODO make this more robust
            if "lm_head" in module_name:
                sampler_module = self.model.get_submodule("sampler")
                new_module = replace_submodule(
                    self.model,
                    "sampler",
                    from_layer_sampler(
                        sampler_module,
                        module,
                        self.delta_slots,
                        self.delta_config,
                        self.model.config,
                    ),
                )
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            new_module.set_mapping(
                self.base_indices,
                self.sampler_indices,
                self.sampler_indices_padded,
                self.embeddings_indices,
                self.indices_len,
            )

    def register_module(self, module_name: str, module: "BaseLayerWithDelta"):
        assert isinstance(module, BaseLayerWithDelta)
        self.modules[module_name] = module

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module), module_name
            )
            or target_module == module_name
            for target_module in self.supported_delta_modules
        )

    def _register_packed_modules(self, module_full_name: str) -> None:
        parts = module_full_name.split(".")
        module_name = parts[-1]
        replacements = self.packed_modules_mapping.get(module_name)
        if not replacements:
            return
        prefix = ".".join(parts[:-1])
        self.packed_modules[module_full_name] = [
            prefix + "." + r if prefix else r for r in replacements
        ]

    def _create_merged_deltas_inplace(self, delta_model: DeltaModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_deltas = []
            has_replacement = False
            for r in new_module_names:
                delta = delta_model.get_delta(r)
                replacement_deltas.append(delta)
                if delta:
                    has_replacement = True
            if not has_replacement:
                continue
            for i in range(len(replacement_deltas)):
                if replacement_deltas[i]:
                    continue
                replacement_deltas[i] = None
            delta_model.deltas[module_name] = PackedDeltaLayerWeights.pack(
                replacement_deltas
            )


class DeltaLRUCache(LRUCache):

    def __init__(self, capacity: int, deactivate_delta_fn: Callable[[Hashable], None]):
        super().__init__(capacity)
        self.deactivate_delta_fn = deactivate_delta_fn

    def _on_remove(self, key: Hashable, value: Any):
        logger.debug(f"Removing Delta. int id: {key}")
        self.deactivate_delta_fn(key)
        return super()._on_remove(key, value)


class LRUCacheDeltaModelManager(DeltaModelManager):
    """A model manager that manages multiple Deltas with LRU cache."""

    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        delta_config: DeltaConfig,
    ):
        super().__init__(
            model, max_num_seqs, max_num_batched_tokens, vocab_size, delta_config
        )
        self._registered_deltas: DeltaLRUCache = DeltaLRUCache(
            self.capacity, self.deactivate_delta
        )
        self._active_deltas: DeltaLRUCache = DeltaLRUCache(
            self.delta_slots, self._deactivate_delta
        )

    def list_deltas(self) -> Dict[int, DeltaModel]:
        """List all registered DeltaModels."""
        return dict(self._registered_deltas.cache)

    def add_delta(self, delta: DeltaModel) -> bool:
        """Add a DeltaModel to the manager."""
        if delta.id not in self._registered_deltas:
            self._add_delta(delta)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_deltas.touch(delta.id)
            was_added = False
        return was_added

    def activate_delta(
        self,
        delta_id: int,
    ) -> bool:
        if (
            delta_id not in self._active_deltas
            and len(self._active_deltas) >= self.delta_slots
        ):
            self._active_deltas.remove_oldest()
        result = super().activate_delta(delta_id)
        # We always touch to update the LRU cache order
        self._active_deltas.touch(delta_id)
        return result

    def remove_oldest_delta(self) -> bool:
        if len(self._registered_deltas) > 0:
            self._registered_deltas.remove_oldest()
            return True
        return False


def create_delta_manager(
    model: nn.Module,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    vocab_size: int,
    delta_config: DeltaConfig,
    delta_manager_cls: Type[DeltaModelManager] = DeltaModelManager,
    **kwargs,
) -> DeltaModelManager:
    """Create a Delta adapter for a given model."""
    if not hasattr(model, "supported_delta_modules"):
        raise ValueError(f"Model {type(model)} is not supported for Delta.")
    delta_manager = delta_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        delta_config=delta_config,
        **kwargs,
    )
    return delta_manager