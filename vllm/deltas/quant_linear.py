import math
import torch
import numpy as np
import torch.nn as nn
import transformers
from loguru import logger
from vllm.deltas.utils import make_q_matrix, ext_gemm_half_q_half

class QuantLinear(nn.Module):
    def __init__(
        self,
        bits: int,
        infeatures: int,
        outfeatures: int,
        bias,
    ):
        super().__init__()
        global _autogptq_cuda_available
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = infeatures
        self.maxq = 2**self.bits - 1
        
        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros", torch.zeros((1, outfeatures // 32 * self.bits), dtype=torch.int32)
        )
        self.register_buffer(
            "scales", torch.zeros((1, outfeatures), dtype=torch.float16)
        )
        self.register_buffer(
            "g_idx",
            torch.tensor(
                [i // infeatures for i in range(infeatures)], dtype=torch.int32
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((outfeatures), dtype=torch.float16)
            )
        else:
            self.bias = None
            
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(
                list(range(0, 32, self.bits)), dtype=torch.int32
            ).unsqueeze(0)