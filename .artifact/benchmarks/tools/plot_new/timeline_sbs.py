import os
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
from utils import set_matplotlib_style
from vllm.tools.utils import parse_data, get_title, color_palette,get_system_name
import matplotlib.pyplot as plt

set_matplotlib_style()
cmp = sns.color_palette("tab10")

def get_model_name(model_name):
    if "delta" in model_name:
        return model_name.replace("delta-", "Model #")
    else:
        return "Base Model"

def plot(args):
    dfs = []
    for file in [x for x in os.listdir(args.input) if x.endswith(".jsonl")]:
        filename = os.path.join(args.input, file)
        print(filename)
        metadata, data = parse_data(filename, order=True)
        title = get_title(metadata)
        df = pd.DataFrame(data)
        # remove E2E Latency and TTFT columns since we don't need them
        min_arrival = df["arrival"].min()
        df.loc[:, (df.columns != "id") & (df.columns!="model")] -= min_arrival
        # select df where type is not in Arrival and Finish
        df = df.sort_values(by="arrival")
        id_map = {v: i for i, v in enumerate(df["id"].unique())}
        df["id"] = df["id"].map(id_map)
        df['system'] = get_system_name(title)
        dfs.append(df)
    df = pd.concat(dfs)
    systems = df['system'].unique()
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75)
    )
    patches = []
    figs = [ax1, ax2]
    types = ["Queuing", "Inference", "Loading", "TTFT"]
    for idx, job_type in enumerate(types):
        patches.append(matplotlib.patches.Patch(color=cmp[idx], label=job_type))
    type_colors = {k: v for k, v in zip(types, cmp)}
    for id, system in enumerate(systems):
        ax = figs[id]
        sub_df = df[df['system'] == system]
        for index, row in sub_df.iterrows():
            ax.barh(
                y=row["id"],
                width=row["queueing_end"] - row["queueing_start"],
                left=row["queueing_start"],
                color=type_colors["Queuing"],
            )
            ax.barh(
                y=row["id"],
                width=row["loading_end"] - row["loading_start"],
                left=row["loading_start"],
                color=type_colors["Loading"],
            )
    fig.savefig(f"{args.output}/timeline.png", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    plot(args)