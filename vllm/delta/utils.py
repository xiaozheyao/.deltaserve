import torch
from vllm.logger import init_logger
import gc
import cupy as cp
import torch.nn as nn
from typing import Optional, Union
import transformers

logger = init_logger(__name__)

none_tensor = torch.empty((1, 1), device="meta")

try:
    from vllm._C import deltazip

except ImportError as e:
    logger.error(f"Error importing deltazip: {e}")

    def error_raise_exllama(*args, **kwargs):
        raise ImportError("deltazip not found")

    make_q_matrix = error_raise_exllama
    gemm_half_q_half = error_raise_exllama


def _torch_device(idx):
    if idx == -1:
        return "cpu"
    return f"cuda:{idx}"


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width, )
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width),
                         dtype=torch.half,
                         device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


def ext_make_q_matrix(w: dict, temp_dq, key: str = None):
    """
    Create Q matrix
    """
    # EXL2
    # won't work as the moment because the tensors are not the same.
    if "q_weight" in w:
        w["q_scale_max"] /= 256
        w["q_perm"] = w["q_perm"].short()
        w["q_invperm"] = w["q_invperm"].short()
        return make_q_matrix(
            w["q_weight"],
            w["q_perm"],
            w["q_invperm"],
            w["q_scale"],
            w["q_scale_max"],
            w["q_groups"],
            none_tensor,
            none_tensor,
            none_tensor,
            temp_dq,
        )
    # GPTQ
    elif "qweight" in w:
        if w["scales"].dtype == torch.float:
            w["scales"] = w["scales"].half()

        # GPTQ with g_idx (act_order)
        if "g_idx" in w and not (w["g_idx"] == 0).all().item():
            w["q_perm"] = torch.empty(
                (w["qweight"].shape[0] * 8, ),
                dtype=torch.short,
                device=w["qweight"].device,
            )
            w["q_invperm"] = torch.empty_like(w["q_perm"])
            # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
            return make_q_matrix(
                w["qweight"],
                w["q_perm"],
                w["q_invperm"],
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                w["g_idx"].cpu(),
                temp_dq,
            )
        # GPTQ without g_idx
        else:
            return make_q_matrix(
                w["qweight"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                none_tensor,
                temp_dq,
            )


class ExLlamaV2DeviceTensors:
    device_idx: int
    scratch_bytes: int
    scratch_idx: int
    scratch: torch.tensor = None

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty(
            (self.scratch_bytes // 2, ),
            dtype=torch.half,
            device=_torch_device(self.device_idx),
        )

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice


def get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def garbage_collection():
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def deltazip_post_init(model,
                       use_act_order: bool,
                       max_input_length: Optional[int] = None):
    """
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    ## exllamav2
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for _, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE"):
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed,
                                      fixed_bytes.get(device, 0))

    if model_uses_exllamav2:
        from deltazip.nn_modules.exllama_utils import ExLlamaV2DeviceTensors

        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ExLlamaV2DeviceTensors(
                device.index, scratch_bytes)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

        for _, submodule in model.named_modules():
            if hasattr(submodule, "QUANT_TYPE"):
                device = submodule.qweight.device
                submodule.post_init(temp_dq=model.device_tensors[device])
    torch.cuda.empty_cache()

    return model


def find_layers(module, layers=None, name=""):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child,
                        layers=layers,
                        name=name + "." + name1 if name != "" else name1))
    return res


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def move_to_device(obj: Union[torch.Tensor, nn.Module], device: torch.device):
    if get_device(obj) != device:
        obj = obj.to(device)
    return obj


def make_quant(
    module,
    names,
    bits,
    name="",
    use_triton=False,
    use_cuda_fp16=True,
    desc_act=False,
    use_exllama: bool = False,
):
    from .quant_linear import QuantLinear

    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            ori_layer_device = get_device(getattr(module, attr))
            delattr(module, attr)
            if type(tmp) == nn.Linear:
                in_features = tmp.in_features
                out_features = tmp.out_features
            if isinstance(bits, dict):
                real_bits = bits[name1]
            else:
                real_bits = bits
            new_layer = QuantLinear(
                real_bits,
                in_features,
                out_features,
                tmp.bias is not None,
                # use_triton=use_triton,
                # use_exllama=use_exllama,
            )
            new_layer.device = ori_layer_device
            setattr(module, attr, new_layer.to(ori_layer_device))

    for name1, child in module.named_children():
        make_quant(
            child,
            names,
            bits,
            name + "." + name1 if name != "" else name1,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=desc_act,
            use_exllama=use_exllama,
        )
