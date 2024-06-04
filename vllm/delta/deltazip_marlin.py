import os
import torch
from typing import List
import torch.nn.functional as F

BITWIDTH = int(os.environ.get("BITWIDTH", "4"))

from triteia.ao.ops.ibmm.ibmm_marlin import (
    ibmm_sparse_marlin_stream as ibmm,
)

def apply_delta(
    x: torch.Tensor,
    qweight_stacked: torch.Tensor,
    scales_stacked: torch.Tensor,
    meta_stacked: torch.Tensor,
    indices: torch.Tensor,
    base_weight: torch.Tensor,
):
    # if there are -1 in indices, it should always at the beginning
    if torch.any(indices == -1):
        # check how many -1 are there
        total_minus_one = torch.sum(indices == -1)
        assert torch.sum(indices[:total_minus_one]) == -total_minus_one, f"index -1 should always be at the beginning: got {indices[:total_minus_one]}, total_minus_one: {total_minus_one}, len(indices): {len(indices)}"
    
    # if torch.any(indices == -1) and not indices[0] == -1:
    #     print(f"wrong indices: {indices}")
    #     assert indices[0] == -1, "index -1 should always be at the beginning"
    # if len(indices) > 2:
    #     print(f"indices: {indices}")
    y = ibmm(
        BITWIDTH, indices, meta_stacked, None, x, qweight_stacked, scales_stacked, None, bias=None, base_weight=base_weight,
    )
    return y

def apply_delta_uncompressed(
    x: torch.Tensor,
    delta_weights: torch.Tensor,
    indices: torch.Tensor,
    base_embedding: torch.Tensor,
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
    base_output = torch.zeros(
        x.shape[0], 
        delta_weights[0].shape[0], 
        dtype=x.dtype, 
        device=x.device,
    )
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        inp = x[indices == id]
        if id == -1:
            w = base_embedding
        else:
            w = delta_weights[id]
        output = F.linear(inp, w)
        base_output[indices == id] = output
    return base_output


def apply_delta_embed(
    x: torch.Tensor,
    delta_weights: List,
    indices: torch.Tensor,
    base_weight: torch.Tensor,
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
    base_output = torch.zeros(
        (x.shape[0], delta_weights[0].shape[1]), device=x.device, dtype=delta_weights.dtype)
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        if id == -1:
            w = base_weight
        else:
            w = delta_weights[id]
        base_output[idx_mask] = F.embedding(inp, w)
    return base_output
