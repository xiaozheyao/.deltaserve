from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Optional, Set, Type
import torch

from vllm.lora.layers import LoRAMapping

from vllm.deltas.request import DeltaRequest
from vllm.deltas.config import DeltaCompressionConfig
from loguru import logger


class AbstractWorkerManager(ABC):
    """Abstract class for managing LoRA/Delta models on the worker side."""

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int,
                 vocab_size: int, delta_config: DeltaCompressionConfig,
                 device: torch.device):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size
        self.device = device
        self.delta_config = delta_config

    @abstractproperty
    def is_enabled(self) -> bool:
        ...

    @abstractmethod
    def create_delta_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        ...

    @abstractmethod
    def set_active_deltas(self, lora_requests: List[DeltaRequest],
                          lora_mapping: LoRAMapping) -> None:
        ...

    @abstractmethod
    def add_delta(self, delta_request: DeltaRequest) -> bool:
        ...

    @abstractmethod
    def add_dummy_delta(self, delta_request: DeltaRequest) -> bool:
        ...

    @abstractmethod
    def remove_delta(self, delta_id: int) -> bool:
        ...

    @abstractmethod
    def remove_all_deltas(self) -> bool:
        ...

    @abstractmethod
    def list_deltas(self) -> Set[int]:
        ...


class WorkerDeltaManager(AbstractWorkerManager):
    """WorkerDeltaManager manages the deltas on the worker side. """
    _delta_manager_cls: Type

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        delta_config: DeltaCompressionConfig,
        device: torch.device,
        delta_model_cls: Type[DeltaModel] = DeltaModel,
    ):
        self._delta_manager: Optional[DeltaModelManager] = None
        self._delta_model_cls = delta_model_cls
        super().__init__(max_num_seqs, max_num_batched_tokens, vocab_size,
                         delta_config, device)

    @property
    def is_enabled(self) -> bool:
        return True

    def create_delta_manager(self, model) -> Any:
        raise NotImplementedError

    def _apply_deltas(self, delta_requests: List[DeltaRequest]) -> None:
        deltas_that_exist = self.list_deltas()
        deltas_map = {
            delta_request.delta_id: delta_request
            for delta_request in delta_requests
        }
        if len(deltas_map) > self._delta_manager.delta_slots:
            raise RuntimeError(
                f"Number of requested deltas ({len(deltas_map)}) is greater than the number of GPU delta slots "
                f"({self._delta_manager.delta_slots}).")
        new_deltas = set(deltas_map)
        deltas_to_add = new_deltas - deltas_that_exist
        deltas_to_remove = deltas_that_exist - new_deltas

        for delta_id in deltas_to_remove:
            self.remove_delta(delta_id)

        for delta_id in deltas_to_add:
            self.add_delta(deltas_map[delta_id])

    def _load_delta(self, delta_request: DeltaRequest) -> DeltaModel:
        try:
            pass
        except Exception as e:
            logger.error(
                f"Failed to load delta model from {delta_request.delta_path}: {e}"
            )
            return None
        return delta_model

    def add_delta(self, delta_request: DeltaRequest) -> bool:
        if delta_request.delta_int_id in self.list_deltas():
            return False
        delta = self._load_delta(delta_request)
        loaded = self._delta_manager.add_lora(delta)
        self._delta_manager.activate_delta(delta.id)
        return loaded

    def remove_delta(self, delta_id: int) -> bool:
        return self._delta_manager.remove_delta(delta_id)

    def remove_all_deltas(self) -> bool:
        return self._delta_manager.remove_all_deltas()

    def list_deltas(self) -> Set[int]:
        return set(self._delta_manager.list_deltas())
