from dataclasses import dataclass


@dataclass
class SwapRequest:
    """
    Request for a Swappable model.
    """

    swap_name: str
    swap_int_id: int
    swap_local_path: str

    def __post_init__(self):
        if self.swap_int_id < 1:
            raise ValueError(f"delta_int_id must be > 0, got {self.swap_int_id}")

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, SwapRequest) and self.swap_int_id == value.swap_int_id
        )

    def __hash__(self) -> int:
        return self.swap_int_id
