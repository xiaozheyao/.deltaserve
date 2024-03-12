import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Set, Type
import torch

from vllm.lora.models import (LoRAModel, LoRAModelManager,
                              LRUCacheLoRAModelManager, create_lora_manager)
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.config import LoRAConfig

from vllm.deltas.request import DeltaRequest
from vllm.deltas.config import DeltaCompressionConfig
from loguru import logger

class AbstractWorkerManager(ABC):
    """Abstract class for managing LoRA/Delta models on the worker side."""

    def __init__(self, 
                 max_num_seqs: int,
                 max_num_batched_tokens: int,
                 vocab_size: int, 
                 delta_config: DeltaCompressionConfig,
                 device: torch.device
                ):
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