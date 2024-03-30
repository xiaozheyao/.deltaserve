import torch
from typing import Optional
from .quant_linear_debug import QuantLinear

# def add_lora_slice(
#     y: torch.Tensor,
#     x: torch.Tensor,
#     wa_t_all: torch.Tensor,
#     wb_t_all: torch.Tensor,
#     indicies: torch.LongTensor,
#     layer_idx: int,
#     scale: float,
#     y_offset: int,
#     y_slice_size: int,
#     *,
#     buffer: Optional[torch.Tensor] = None
# ):
#     """
#     Same as `add_lora` but you can operate on slices of y.
#     Pass whole y, define y_offset and y_slice_size.

#     Semantics:
#       y[i] += (
#           x[i].unsqueeze(0)
#           @ wa_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
#           @ wb_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
#           * scale
#         ).squeeze(0)

#     Args:
#       y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
#       x: Shape: `[B, H1]`. Input vectors.
#       wa_t_all: Shape: `[None, L, R, H1]`. All of the transposed
#         LoRA A matrices.
#       wb_t_all: Shape: `[None, L, H2, R]`. All of the transposed
#         LoRA B matrices.
#       indicies: Shape: `[B]`. Indices of the LoRA weights.
#       layer_idx: Layer index of LoRA weights.
#       scale: Scaling factor.
#       y_offset: Offset to apply to the starting column of y.
#       y_slice_size: Size of the y column slice.
#     """
#     try:
#         import vllm._punica_C as punica_kernels
#     except ImportError as e:
#         _raise_import_error(e)

#     r = wb_t_all.size(-1)
#     if buffer is None:
#         # We set the buffer to be float32 by default to avoid
#         # numerical inaccuracies that would otherwise happen
#         # due to downcasting.
#         buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)
#     punica_kernels.dispatch_bgmv_low_level(
#         buffer,
#         x,
#         wa_t_all,
#         indicies,
#         layer_idx,
#         1.0,
#         x.size(1),
#         buffer.size(1),
#         0,
#     )
#     punica_kernels.dispatch_bgmv_low_level(
#         y,
#         buffer,
#         wb_t_all,
#         indicies,
#         layer_idx,
#         scale,
#         buffer.size(1),
#         y_slice_size,
#         y_offset,
#     )


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
    buffer: Optional[torch.Tensor] = None
):
    """
    semantics:
        y[i] += (
            x[i].unsqueeze(0)
            @ qweight[indices[i], :, :].transpose(-1, -2)
        ).squeeze(0)
    """
    ql = QuantLinear.from_tensors(
        qweight[0][0], qzeros[0][0], scales[0][0], g_idx, bias=None
    )
    output = ql(x)
    y[:, y_offset : y_offset + y_slice_size] += output
