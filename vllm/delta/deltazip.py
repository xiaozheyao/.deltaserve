import torch
from typing import Optional

# from .quant_linears.quant_linear_naive import QuantLinear
from .quant_linears.quant_linear_exllama import QuantLinear


def add_delta(
    y: torch.Tensor,
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    indices: torch.LongTensor,
    layer_idx: int,
    device_tensor: any,
):
    """
    semantics:
        y[i] += (
            x[i].unsqueeze(0)
            @ qweight[indices[i], :, :].transpose(-1, -2)
        ).squeeze(0)
    """
    ql = QuantLinear.from_tensors(
        qweight[0][0],
        qzeros[0][0],
        scales[0][0],
        g_idx,
        bias=None,
        device_tensor=device_tensor,
    )
    output = ql(x)
    y += output


def add_delta_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    indices: torch.LongTensor,
    scale: float,
    y_offset: int,
    y_slice_size: int,
    *,
    buffer: Optional[torch.Tensor] = None,
    device_tensor: any,
):
    """
    semantics:
        y[i] += (
            x[i].unsqueeze(0)
            @ qweight[indices[i], :, :].transpose(-1, -2)
        ).squeeze(0)
    """

    ql = QuantLinear.from_tensors(
        qweight[0][0],
        qzeros[0][0],
        scales[0][0],
        g_idx,
        bias=None,
        device_tensor=device_tensor,
    )
    output = ql(x, y[:, y_offset : y_offset + y_slice_size])
    y[:, y_offset : y_offset + y_slice_size] += output
