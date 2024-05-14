import torch
from typing import List
import torch.nn.functional as F


def apply_swap_embed(
    x: torch.Tensor,
    packed_weights: List,
    indices: torch.Tensor,
):
    outputs = torch.zeros(
        x.shape[0], x.shape[1], packed_weights[0].shape[1], device=x.device
    )
    print(f"outputs.shape: {outputs.shape}")
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        output = F.embedding(inp, packed_weights[id])
        print(f"output.shape: {output.shape}")
        outputs[idx_mask] = output
    return outputs

def apply_swap_packed_nslice(
    x: torch.Tensor,
    stacked_weights: List,
    indices: torch.Tensor,
):
    outputs = torch.zeros(
        x.shape[0], x.shape[1], stacked_weights[0].shape[1], device=x.device
    )
    print(f"apply_swap_packed_nslice outputs.shape: {outputs.shape}")
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        output = torch.matmul(inp, stacked_weights[id].T)
        print(f"apply_swap_packed_nslice output.shape: {output.shape}")
        outputs[idx_mask] = output
    return outputs