import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
from utils import set_matplotlib_style
from vllm.tools.utils import parse_data, get_title, color_palette
import matplotlib.pyplot as plt

set_matplotlib_style()
cmp = sns.color_palette("tab10")


def plot(args):
    metadata, data = parse_data(args.input, order=True)
    title = get_title(metadata)
    filename = args.input.split("/")[-1].removesuffix(".jsonl")
    df = pd.DataFrame(data)
    min_arrival = df["arrival"].min()
    # if columns are not id and model, then subtract min_arrival
    df.loc[:, (df.columns != "id") & (df.columns!="model")] -= min_arrival
    # select df where type is not in Arrival and Finish
    df = df.sort_values(by="arrival")
    # rewrite id in ascending order
    id_map = {v: i for i, v in enumerate(df["id"].unique())}
    df["id"] = df["id"].map(id_map)
    fig, ax = plt.subplots()
    patches = []
    types = ["Queuing", "Inference", "Loading", "TTFT", "Preempty"]
    for idx, job_type in enumerate(types):
        patches.append(matplotlib.patches.Patch(color=cmp[idx], label=job_type))
    type_colors = {k: v for k, v in zip(types, cmp)}
    
    for index, row in df.iterrows():
        plt.barh(
            y=row["id"],
            width=row["queueing_end"] - row["queueing_start"],
            left=row["queueing_start"],
            color=type_colors["Queuing"],
        )
        plt.barh(
            y=row["id"],
            width=row["loading_end"] - row["loading_start"],
            left=row["loading_start"],
            color=type_colors["Loading"],
        )
        plt.barh(
            y=row["id"],
            width=row["first_token_end"] - row["first_token_start"],
            left=row["first_token_start"],
            color=type_colors["TTFT"],
        )
        plt.barh(
            y=row["id"],
            width=row["inference_end"] - row["inference_start"],
            left=row["inference_start"],
            color=type_colors["Inference"],
        )
        plt.text(
            x=row["inference_end"] + 0.5,
            y=row['id'] + 0.25,
            s=row['model'],
            fontdict=dict(color='black', fontsize=3),
        )
        for col in df.columns:
            if col.startswith("preempt_out"):
                if not pd.isna(row[col]):
                    empt_id = col.split("_")[-1]
                    empt_in_time = row[f"preempt_in_{empt_id}"]
                    plt.barh(
                        y=row["id"],
                        width=empt_in_time - row[col],
                        left=row[col],
                        color=type_colors["Preempty"],
                    )
    plt.title("Time Breakdown", fontsize=15)
    plt.gca().invert_yaxis()
    ax.xaxis.grid(True, alpha=0.5)
    # Adding a legend
    ax.legend(handles=patches, fontsize=11)
    fig.savefig(f"{args.output}/{filename}.png", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot latency per request")
    parser.add_argument(
        "--input", type=str, help="Input file", default=".artifact/benchmarks/results"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output folder",
        default=".artifact/benchmarks/results",
    )
    parser.add_argument(
        "--type", type=str, choices=["nfs", "nvme"], help="harddrive", default="nfs"
    )
    parser.add_argument("--ar", type=str, default="0.5")
    args = parser.parse_args()
    plot(args)
