import torch
import torch.nn as nn
from vllm.logger import init_logger
from triteia.ao.ops.linalg.matmul.bitblas_matmul_lowprec import bitblas_quant_bmm_248

logger = init_logger(__name__)

class QuantLinear(nn.Module):
    def __init__(
        self,
        bitwidth: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
    ):
        super().__init__()
        self.bitwidth = bitwidth
        self.qweight  = qweight
        self.zeros    = qzeros
        self.scales   = scales

    @torch.inference_mode()
    def forward(self, x):
        print("x.shape: ", x.shape, "x.dtype: ", x.dtype)
        print("self.qweight.shape: ", self.qweight.shape, "self.qweight.dtype: ", self.qweight.dtype)
        print("self.zeros.shape: ", self.zeros.shape, "self.zeros.dtype: ", self.zeros.dtype)
        print("self.scales.shape: ", self.scales.shape, "self.scales.dtype: ", self.scales.dtype)
        
        return bitblas_quant_bmm_248(self.bitwidth, x, qweight=self.qweight, qzero=self.zeros, scale=self.scales)

    @classmethod
    def from_tensors(
        cls, bitwidth, qweight, qzeros, scales, g_idx, bias, device_tensor
    ):
        obj = cls(bitwidth, qweight, qzeros, scales)
        return obj
