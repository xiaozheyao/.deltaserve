from typing import List, Optional
import torch
from vllm.utils import in_wsl


class DeltaLayerWeights:
    """Delta weights for a layer composed of base model and compressed delta.
    """

    def __init__(
        self,
        module_name: str,
        compress_config: Optional[dict] = None,
    ) -> None:
        self.module_name = module_name
