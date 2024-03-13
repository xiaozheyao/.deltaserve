# pylint: disable=unused-argument
from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Tuple
if TYPE_CHECKING:
    pass

@dataclass
class DeltaMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)

