import os
import json
import safetensors as st
from fractions import Fraction
from safetensors.torch import save_file

transpose_modules = False


def main(args):
    column_chunking_modules = [
        "self_attn.q_proj.qweight",
        "self_attn.k_proj.qweight",
        "self_attn.v_proj.qweight",
        "mlp.gate_proj.qweight",
        "mlp.up_proj.qweight",
    ]

    row_chunking_modules = [
        "self_attn.o_proj.qweight",
        "mlp.down_proj.qweight",
        "embed_tokens.weight",
        "lm_head.weight",
    ]

    with open(os.path.join(args.input, "compress_config.json"), "r") as fp:
        compress_config = json.load(fp)
    pack_factor = Fraction(32, compress_config["bits"])
    tensors = {}
    rank_tensors = [{}] * args.tp_size
    with st.safe_open(
        os.path.join(args.input, "deltazip-compressed.safetensors"), "torch"
    ) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    chunked_keys = []
    for key in tensors.keys():
        if any([module in key for module in column_chunking_modules]):
            for i in range(args.tp_size):
                shard_size = tensors[key].shape[1] // args.tp_size
                rank_tensors[i][key] = tensors[key][
                    :, i * shard_size : (i + 1) * shard_size
                ]
                if transpose_modules:
                    rank_tensors[i][key] = rank_tensors[i][key].transpose(0, 1)
                rank_tensors[i][key] = rank_tensors[i][key].contiguous()
            chunked_keys.append(key)

        if any([module in key for module in row_chunking_modules]):
            for i in range(args.tp_size):
                shard_size = tensors[key].shape[0] // args.tp_size
                rank_tensors[i][key] = tensors[key][
                    i * shard_size : (i + 1) * shard_size, :
                ]
                rank_tensors[i][key] = rank_tensors[i][key].contiguous()
            chunked_keys.append(key)

    for key in chunked_keys:
        del tensors[key]
    print(f"Chunking Finished, saving to {args.input}/rank.[rank_id].safetensors")
    for rank_id, rank_tensor in enumerate(rank_tensors):
        print(f"Saving rank {rank_id}")
        save_file(rank_tensor, f"{args.input}/rank.{rank_id}.safetensors")
    save_file(tensors, f"{args.input}/deltazip-compressed-remain.safetensors")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize the I/O of a safetensors file"
    )
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument(
        "--tp-size", type=int, help="Tensor Parallelism Size", default=1
    )

    args = parser.parse_args()
    main(args)