import os
import torch
from typing import Optional, Tuple, List, Any
import torch.nn.functional as F

USE_BITBLAS = os.environ.get("USE_BITBLAS", "0") == "1"
BITWIDTH = int(os.environ.get("BITWIDTH", "4"))
USE_TRITEIA = os.environ.get("USE_TRITEIA", "0") == "1"
if USE_BITBLAS:
    from .quant_linears.quant_linear_bitblas import QuantLinear
elif USE_TRITEIA:
    from triteia.ao.ops.linalg.matmul.bmm_lowprec import quant_bmm_248
else:
    # from .quant_linears.quant_linear_naive import QuantLinear
    # from .quant_linears.quant_linear_exllama import QuantLinear
    pass


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
    # y.shape [2048, 4096]
    # output.shape [max_deltas, 2048, 4096]
    # x.shape [max_deltas, 2048, 4096]
    # indices [2048]
    x = x[indices != -1]
    if x.shape[0] == 0:
        return y
    x = x.repeat(qweight.shape[0], 1, 1)
    g_idx = g_idx.repeat(qweight.shape[0], 1)
    g_idx = g_idx.to(qweight.device)
    output = quant_bmm_248(BITWIDTH, x, qweight, qzeros, scales, g_idx, bias=None)
    valid_mask = indices != -1
    filtered_indices = indices[valid_mask]
    try:
        yrange = torch.arange(y.shape[0], device=y.device)
        partial = output[filtered_indices, torch.arange(y.shape[0], device=y.device)[valid_mask], :]
    except:
        with open("error.txt", "w") as f:
            f.write(f"{output.shape}\n")
            f.write(f"{y.shape}\n")
            f.write(f"{valid_mask.shape}\n")
    # y[valid_mask] += partial
    return y


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
    bsz = qweight.shape[0]
    x = x[indices != -1]
    if x.shape[0] == 0:
        return y
    x = x.repeat(qweight.shape[0], 1, 1)
    g_idx = g_idx.repeat(qweight.shape[0], 1)
    g_idx = g_idx.to(qweight.device)
    output = quant_bmm_248(BITWIDTH, x, qweight, qzeros, scales, g_idx, bias=None)
    valid_mask = indices != -1
    filtered_indices = indices[valid_mask]
    y[valid_mask, y_offset:y_offset + y_slice_size] += output[filtered_indices, torch.arange(y.shape[0], device=y.device)[valid_mask], :y_slice_size]
    return y
    

def apply_delta(
    x: torch.Tensor,
    qweight_stacked: torch.Tensor,
    qzeros_stacked: torch.Tensor,
    scales_stacked: torch.Tensor,
    g_idx_stacked: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    device_tensor: Any,
):
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    add_delta(
        output,
        x,
        qweight_stacked,
        qzeros_stacked,
        scales_stacked,
        g_idx_stacked,
        indices,
        1.0,
        device_tensor=device_tensor,
    )
    return output.view_as(org_output)


def apply_delta_packed_nslice(
    x: torch.Tensor,
    qweight_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qzeros_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scales_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    g_idx_stacked: List[torch.Tensor],
    indices: torch.Tensor,
    output: torch.Tensor,
    output_slices: Tuple[int, ...],
    device_tensor: Any,
):
    """
    Applies delta to each input.
    This method applies all deltas to each input. It uses the
    indices vector to determine which delta yields the
    correct output. An index of -1 means no delta should be
    applied. This method adds the final delta results to the
    output.

    This method is used for layers that are composed of multiple sublayers
    (slices) packed together.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        qweight_stacked:    3 element tuple of (num_deltas, 1,  hidden_dim/pack_factor, hidden_dim)
        qzeros_stacked:     3 element tuple of (num_deltas, 1, 1, hidden_dim/pack_factor)
        indices:           (batch_size)
        output:            (batch_size, q_slice_size + 2*kv_slice_size)
        output_slices:     n-1 element tuple of (slice_size...),
                           where n is number of slices
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    indices = indices.view(-1)
    offset_left = 0
    for slice_idx in range(len(output_slices)):
        add_delta_slice(
            output,
            x,
            qweight_stacked[slice_idx],
            qzeros_stacked[slice_idx],
            scales_stacked[slice_idx],
            g_idx_stacked[slice_idx],
            indices,
            1.0,
            offset_left,
            output_slices[slice_idx],
            device_tensor=device_tensor,
        )
        offset_left += output_slices[slice_idx]
    return output.view_as(org_output)


def apply_delta_uncompressed(
    x: torch.Tensor,
    delta_weights: torch.Tensor,
    indices: torch.Tensor,
    base_output: torch.Tensor,
):
    """
    Applies delta to each input.

    This method applies all deltas to each input. An index of -1 means no delta should be applied.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        weights:           ()
        indices:           (batch_size)
        output:            (batch_size, hidden_dim)
    """
    outputs = torch.zeros(
        (len(delta_weights), base_output.shape[0], base_output.shape[1]),
        device=base_output.device,
    )
    valid_mask = indices != -1
    filtered_indices = indices[valid_mask]
    outputs = torch.matmul(x, delta_weights.mT)
    base_output[valid_mask] += outputs[filtered_indices, torch.arange(base_output.shape[0], device=base_output.device)[valid_mask], :]
    return base_output

def apply_delta_embed(
    x: torch.Tensor,
    delta_weights: List,
    indices: torch.Tensor,
    base_output: torch.Tensor,
):
    """
    Applies delta to each input.
    This method applies all deltas to each input. It uses the
    indices vector to determine which delta yields the
    correct output. An index of -1 means no delta should be
    applied. This method adds the final delta results to the
    output.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        delta_weights:     list of delta weights
        indices:           (batch_size)
    """
    outputs = torch.zeros(
        (len(delta_weights), base_output.shape[0], base_output.shape[1]),
        device=base_output.device,
    )
    embedding_2d = delta_weights.view(
        delta_weights.shape[0] * delta_weights.shape[1],
        delta_weights.shape[2],
    )
    outputs = F.embedding(x, embedding_2d)
    base_output += outputs
    return base_output
