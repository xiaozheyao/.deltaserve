# pylint: disable=unused-argument
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.lora.punica import add_lora, add_lora_slice, bgmv
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_gather,
)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear,
                                               QKVParallelLinear,
                                               MergedColumnParallelLinear)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.utils import (
    split_tensor_along_last_dim)

if TYPE_CHECKING:
    pass

