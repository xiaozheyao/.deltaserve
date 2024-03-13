import json
from dataclasses import dataclass, field, fields
from transformers.utils.hub import PushToHubMixin
from os.path import join


@dataclass
class DeltaCompressionConfig(PushToHubMixin):
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8, 16]})
    # sparsity = how many parameters we set to zero after quantization
    sparsity: float = field(default=0)
    prunen: int = field(default=0)
    prunem: int = field(default=0)
    group_size: int = field(default=-1)
    group_rows: int = field(
        default=-1)  # deprecated, for backward compatibility
    block_size: int = field(default=128)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=True)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    lossless: str = field(default="none")
    dtype: str = field(default="fp16")

    def __post_init__(self):
        fields_info = fields(self)
        if self.sparsity < 0 or self.sparsity > 1:
            raise ValueError("sparsity must be [0, 1]")
        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(
                f"only support quantize to {fields_info[0].metadata['choices']} bits."
            )
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError(
                "unless equal to -1, group_size must greater then 0.")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, "compress_config.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str):
        with open(join(save_dir, "compress_config.json"),
                  "r",
                  encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "sparsity": self.sparsity,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "lossless": self.lossless,
            "prunen": self.prunen,
            "prunem": self.prunem,
            "block_size": self.block_size,
        }
