import torch
import torch.nn as nn
from vllm.logger import init_logger
from triteia.ao.ops.linalg.matmul.bitblas_matmul_lowprec import bitblas_quant_bmm_248

logger = init_logger(__name__)

class QuantLinear(nn.Module):
    def __init__(
        self,
        bitwidth: int,
    ):
        super().__init__()
        self.bitwidth = bitwidth

    def forward(self, x):        
        output = bitblas_quant_bmm_248(self.bitwidth, x, qweight=self.qweight, qzero=self.zeros, scale=self.scales)
        return output
    
    @classmethod
    def from_tensors(
        cls, bitwidth, qweight, qzeros, scales, g_idx, bias, device_tensor
    ):
        obj = cls(bitwidth)
        obj.qweight  = qweight
        obj.zeros    = qzeros
        obj.scales   = scales
        return obj
