from vllm.delta.compressor import LosslessCompressor
import safetensors as st
import json
import cupy as cp
from safetensors.torch import save_file

FILEPATH = ".idea/models/lmsys.vicuna-7b-v1.5.2b50s128g/deltazip-compressed.safetensors"

lc = LosslessCompressor()
tensors = {}
with st.safe_open(FILEPATH, "torch") as f:
    metadata = f.metadata()
    keys = f.keys()
    for key in keys:
        tensors[key] = f.get_tensor(key)
    tensor_dtypes = json.loads(metadata["dtype"])
    tensor_shapes = json.loads(metadata["shape"])

with cp.cuda.Device(0):
    for key in tensors.keys():
        tensors[key] = cp.array(tensors[key], copy=False)
tensors = lc.decompress_state_dict(
    tensors,
    tensor_shapes,
    tensor_dtypes,
    use_bfloat16=False,
    target_device="cuda:0",
)
# save the decompressed tensors
save_file(
    tensors,
    ".idea/models/lmsys.vicuna-7b-v1.5.2b50s128g/deltazip-compressed.safetensors",
)
