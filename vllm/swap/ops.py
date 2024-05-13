import torch
from typing import List

def apply_swap_embed(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    packed_weights: List,
    indices: torch.Tensor,
):
    pass