import os
import pandas as pd
from vllm.tools.utils import parse_data, get_short_system_name

DEFAULT_PATH = ".artifact/benchmarks/results/pjlab/ready2"

import os


def walk_through_files(path, file_extension=".jsonl"):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dirpath, filename)


def prepare_df(input_file, order=False):
    if os.path.isdir(input_file):
        # walk through the directory and get all the jsonl files, including the subdirectories
        inputs = list(walk_through_files(input_file))
    else:
        inputs = [input_file]
    inputs = list(set(inputs))
    results_df = pd.DataFrame([])
    for input_file in inputs:
        metadata, results = parse_data(input_file, order=order)
        short_sys_name, sys_order = get_short_system_name(metadata)
        results = pd.DataFrame(results)
        results["filename"] = input_file
        results["max_deltas"] = metadata["max_deltas"]
        results["max_cpu_deltas"] = metadata["max_cpu_deltas"]
        results["max_swaps"] = metadata["max_swaps"]
        results["max_cpu_swaps"] = metadata["max_cpu_swaps"]
        results["sys_name"] = short_sys_name
        results["order"] = sys_order
        results["distribution"] = metadata["distribution"]
        results["ar"] = (
            metadata["ar"] if metadata["distribution"] != "distinct" else "0"
        )
        results["tp_size"] = metadata["tp_size"]
        results["policy"] = metadata["policy"]
        results["total_models"] = metadata["total_models"]
        results_df = pd.concat([results_df, results])
    return results_df


if __name__ == "__main__":
    input_file = ".artifact/benchmarks/results/pjlab/ready"
    df = prepare_df(input_file)
    ars = [1, 2, 3, 6, 9]
    distribution = ["uniform", "zipf:1.5", "azure"]
    for distribution in distribution:
        for ar in ars:
            for tp_size in [1, 2, 4, 8]:
                for policy in ["lru", "fifo", "lfu"]:
                    print(f"{distribution}, {ar}, {tp_size}, {policy}")
                    print(
                        df[
                            (df["distribution"] == distribution)
                            & (df["ar"] == ar)
                            & (df["tp_size"] == tp_size)
                            & (df["policy"] == policy)
                        ]
                        .groupby(["sys_name", "order", "type"])["time"]
                        .mean()
                    )
                    print()
