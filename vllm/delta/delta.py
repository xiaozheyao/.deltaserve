from typing import Optional


class DeltaLayerWeights:
    """Delta weights for a layer composed of base model and compressed delta.
    """

    def __init__(
        self,
        module_name: str,
        compress_config: Optional[dict] = None,
    ) -> None:
        self.module_name = module_name
