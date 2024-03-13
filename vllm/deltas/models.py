import math
import copy
from typing import Dict, Optional, List

from vllm.deltas.delta import DeltaLayerWeights
from vllm.deltas.config import DeltaConfig
from vllm.deltas.layers import BaseLayerWithDelta

from vllm.logger import init_logger

logger = init_logger(__name__)

_GLOBAL_DELTA_ID = 0


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
        assert self.capacity >= self.delta_slots, "capacity must be greater than delta_slots"
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
    
    def activate_delta(
        self, delta_id: int
    ):
        """Move delta into GPU buffer to be used in the forward pass"""
        if delta_id in self._active_deltas:
            return False
        first_free_slot = next(
            ((i, delta_id) for i, delta_id in enumerate(self.delta_index_to_id)
             if delta_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free delta slots")
        index, _ = first_free_slot
        self._active_deltas[delta_id] = None
        delta_model = self._registered_deltas[delta_id]
        logger.debug(
            f"Activating Delta. int id: {delta_model.id}, slot index: {index}")
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
        (base_indices, sampler_indices, sampler_indices_padded,
         embeddings_indices,
         indices_len) = convert_mapping(mapping, self.delta_index_to_id,
                                        self.delta_slots + 1, self.vocab_size,
                                        self.delta_config.delta_extra_vocab_size)
        self.base_indices[:base_indices.shape[0]].copy_(base_indices)
        self.sampler_indices[:sampler_indices.shape[0]].copy_(sampler_indices)
        self.sampler_indices_padded[:sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded)
        self.embeddings_indices[:embeddings_indices.
                                shape[0], :embeddings_indices.shape[1]].copy_(
                                    embeddings_indices)
        # Maintain the reference
        self.indices_len[:] = indices_len

    def set_delta_mapping(self, delta_mapping: DeltaMapping) -> None:
        if self._last_mapping != delta_mapping:
            self._set_lora_mapping(delta_mapping)
        self._last_mapping = delta_mapping

    def list_deltas(self) -> Dict[int, DeltaModel]:
        """List all registered DeltaModels."""
        return dict(self._registered_deltas)

    def get_lora(self, delta_id: int) -> Optional[DeltaModel]:
        return self._registered_deltas.get(delta_id, None)

    def remove_all_loras(self) -> bool:
        """Remove all LoRAModels from the manager."""
        self._registered_loras.clear()
        self.lora_index_to_id = [None] * self.lora_slots
        self._active_loras.clear()

    def _create_lora_modules(self):
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name):
                continue

            new_module = replace_submodule(
                self.model, module_name,
                from_layer(module, self.lora_slots, self.lora_config,
                           self.model.config))
            # (yard1): TODO make this more robust
            if "lm_head" in module_name:
                sampler_module = self.model.get_submodule("sampler")
                new_module = replace_submodule(
                    self.model, "sampler",
                    from_layer_sampler(sampler_module, module, self.lora_slots,
                                       self.lora_config, self.model.config))
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            new_module.set_mapping(self.base_indices, self.sampler_indices,
                                   self.sampler_indices_padded,
                                   self.embeddings_indices, self.indices_len)

    def register_module(self, module_name: str, module: "BaseLayerWithLoRA"):
        assert isinstance(module, BaseLayerWithLoRA)
        self.modules[module_name] = module

    def create_dummy_lora(
            self,
            lora_id: int,
            rank: int,
            embedding_modules: Optional[Dict[str, str]] = None) -> LoRAModel:
        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAModel(lora_id, rank, {})
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name) or not isinstance(
                    module, BaseLayerWithLoRA):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                if parts[-1] in embedding_modules:
                    input_dim = (module.base_layer.org_vocab_size +
                                 self.lora_config.lora_extra_vocab_size if
                                 hasattr(module.base_layer, "org_vocab_size")
                                 else module.base_layer.weight.shape[1])
                    output_dim = module.base_layer.embedding_dim if hasattr(
                        module.base_layer,
                        "embedding_dim") else module.base_layer.weight.shape[0]
                    embeddings_tensor_dim = (module.base_layer.embedding_dim if
                                             hasattr(module.base_layer,
                                                     "embedding_dim") else
                                             module.base_layer.weight.shape[1])
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        input_dim,
                        output_dim,
                        rank,
                        module.lora_a_stacked.dtype,
                        "cpu",
                        embeddings_tensor_dim=embeddings_tensor_dim)
                else:
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.lora_a_stacked.shape[-1],
                        module.lora_b_stacked.shape[-2],
                        rank,
                        module.lora_a_stacked.dtype,
                        "cpu",
                    )
                lora.optimize()
            else:
                parts = module_name.split(".")
                replacements = self.packed_modules_mapping[parts[-1]]
                subloras = []
                for i, r in enumerate(replacements):
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name + "." + r,
                        module.lora_a_stacked[i].shape[-1],
                        module.lora_b_stacked[i].shape[-2],
                        rank,
                        module.lora_a_stacked[i].dtype,
                        "cpu",
                    )
                    lora.optimize()
                    subloras.append(lora)
                lora = PackedLoRALayerWeights.pack(subloras)
            model.loras[module_name] = lora
        return model

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module),
                module_name) or target_module == module_name
            for target_module in self.supported_lora_modules)

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

    def _create_merged_loras_inplace(self, lora_model: LoRAModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_loras = []
            has_replacement = False
            for r in new_module_names:
                lora = lora_model.get_lora(r)
                replacement_loras.append(lora)
                if lora:
                    has_replacement = True
            if not has_replacement:
                continue
            for i in range(len(replacement_loras)):
                if replacement_loras[i]:
                    continue
                replacement_loras[i] = None
            lora_model.loras[module_name] = PackedLoRALayerWeights.pack(
                replacement_loras)


class LoRALRUCache(LRUCache):

    def __init__(self, capacity: int, deactivate_lora_fn: Callable[[Hashable],
                                                                   None]):
        super().__init__(capacity)
        self.deactivate_lora_fn = deactivate_lora_fn

    def _on_remove(self, key: Hashable, value: Any):
        logger.debug(f"Removing LoRA. int id: {key}")
        self.deactivate_lora_fn(key)
        return super()._on_remove(key, value)


class LRUCacheLoRAModelManager(LoRAModelManager):
    """A model manager that manages multiple LoRAs with LRU cache."""

    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
    ):
        super().__init__(model, max_num_seqs, max_num_batched_tokens,
                         vocab_size, lora_config)
        self._registered_loras: LoRALRUCache = LoRALRUCache(
            self.capacity, self.deactivate_lora)
        self._active_loras: LoRALRUCache = LoRALRUCache(
            self.lora_slots, self._deactivate_lora)

    def list_loras(self) -> Dict[int, LoRAModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_loras.cache)

    def add_lora(self, lora: LoRAModel) -> bool:
        """Add a LoRAModel to the manager."""
        if lora.id not in self._registered_loras:
            self._add_lora(lora)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_loras.touch(lora.id)
            was_added = False
        return was_added

    def activate_lora(
        self,
        lora_id: int,
    ) -> bool:
        if lora_id not in self._active_loras and len(
                self._active_loras) >= self.lora_slots:
            self._active_loras.remove_oldest()
        result = super().activate_lora(lora_id)
        # We always touch to update the LRU cache order
        self._active_loras.touch(lora_id)
        return result

    def remove_oldest_lora(self) -> bool:
        if len(self._registered_loras) > 0:
            self._registered_loras.remove_oldest()
            return True
        return False


def create_lora_manager(
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        lora_manager_cls: Type[LoRAModelManager] = LoRAModelManager,
        **kwargs) -> LoRAModelManager:
    """Create a LoRA adapter for a given model."""
    if not hasattr(model, "supported_lora_modules"):
        raise ValueError(f"Model {type(model)} is not supported for LoRA.")
    lora_manager = lora_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        lora_config=lora_config,
        **kwargs)
    return lora_manager
