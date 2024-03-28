import torch
from typing import Optional, List
from .config import CompressionConfig


class DeltaLayerWeights:
    """Delta weights for a layer composed of base model and compressed delta.
    """

    def __init__(
        self,
        module_name: str,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        compress_config: CompressionConfig,
    ) -> None:
        self.module_name = module_name
        self.config = compress_config
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx


class PackedDeltaLayerWeights(DeltaLayerWeights):
    """Delta used for packed layers (eg. qkv_proj)."""

    def __init__(
        self,
        module_name: str,
        qweight: List[torch.Tensor],
        qzeros: List[torch.Tensor],
        scales: List[torch.Tensor],
        g_idx: List[torch.Tensor],
        compress_config: CompressionConfig,
    ) -> None:
        super().__init__(
            module_name=module_name,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            compress_config=compress_config,
        )

    @classmethod
    def pack(cls,
             deltas: List["DeltaLayerWeights"]) -> "PackedDeltaLayerWeights":
        """Pack a list of Deltas into a single LoRA.

        If LoRA is None, it signifies that the submodule does not have a LoRA.
        """
        first_delta = next(delta for delta in deltas if delta is not None)
        module_name = first_delta.module_name
        obj = cls(
                module_name,
                [delta.qweight if delta is not None else None for delta in deltas],
                [delta.qzeros if delta is not None else None for delta in deltas],
                [delta.scales if delta is not None else None for delta in deltas],
                [delta.g_idx if delta is not None else None
                for delta in deltas], first_delta.config
            )
        return obj

    @property
    def input_dim(self) -> int:
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        raise NotImplementedError()

    @property
    def is_packed(self) -> bool:
        return True


class DeltaZipWeight:

    def __init__(self, qweight) -> None:
        self.qweight = qweight
