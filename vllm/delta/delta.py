import torch
from typing import Optional, List


class DeltaLayerWeights:
    """Delta weights for a layer composed of base model and compressed delta.
    """

    def __init__(
        self,
        module_name: str,
        compress_config: Optional[dict] = None,
    ) -> None:
        self.module_name = module_name

class PackedDeltaLayerWeights(DeltaLayerWeights):
    """Delta used for packed layers (eg. qkv_proj)."""

    def __init__(
        self,
        module_name: str,
    ) -> None:
        super().__init__(
            module_name=module_name,
        )
        

    @classmethod
    def pack(cls, loras: List["DeltaLayerWeights"]) -> "PackedDeltaLayerWeights":
        """Pack a list of LoRAs into a single LoRA.

        If LoRA is None, it signifies that the submodule does not have a LoRA.
        """
        raise NotImplementedError()

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