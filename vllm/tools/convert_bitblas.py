import os
from tqdm import tqdm
import safetensors as st
from triteia.ao.utils.bitblas import convert_to_bitblas
from safetensors.torch import save_file

os.environ["NUMEXPR_MAX_THREADS"] = "32"


def main(args):
    print(args)
    tensors = {}
    new_tensors = {}
    with st.safe_open(args.ckpt, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    quantized_modules = [
        x.removesuffix(".qweight") for x in tensors.keys() if "qweight" in x
    ]
    remaining_modules = list(tensors.keys())
    for module in tqdm(quantized_modules, desc="Converting"):
        qweight, scales, zeros, bias = convert_to_bitblas(
            args.bitwidth, module, tensors
        )
        new_tensors[module + ".qweight"] = qweight
        new_tensors[module + ".scales"] = scales
        new_tensors[module + ".zeros"] = zeros
        if bias is not None:
            new_tensors[module + ".bias"] = bias
        remaining_modules.remove(module + ".qweight")
        remaining_modules.remove(module + ".qzeros")
        remaining_modules.remove(module + ".scales")
        if module + ".bias" in remaining_modules:
            remaining_modules.remove(module + ".bias")
        remaining_modules.remove(module + ".g_idx")

    for module in remaining_modules:
        new_tensors[module] = tensors[module]
    save_file(new_tensors, args.output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bitwidth", type=int, required=True)

    args = parser.parse_args()
    main(args)
