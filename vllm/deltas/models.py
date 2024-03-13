from typing import (Dict, Optional)

from vllm.deltas.delta import DeltaLayerWeights

_GLOBAL_DELTA_ID = 0


def get_delta_id():
    global _GLOBAL_DELTA_ID
    _GLOBAL_DELTA_ID += 1
    return _GLOBAL_DELTA_ID


class DeltaModel:
    """A delta model compressed from the fine-tuned variant"""

    def __init__(
        self,
        delta_model_id: int,
        deltas: Dict[str, DeltaLayerWeights],
    ):
        self.id = delta_model_id
        self.deltas: Dict[str, DeltaLayerWeights] = deltas

    def get_delta(self, module_name: str) -> Optional[DeltaLayerWeights]:
        return self.deltas.get(module_name, None)

    @classmethod
    def from_checkpoint(cls):
        raise NotImplementedError
