import os
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
from utils import set_matplotlib_style
from vllm.tools.utils import parse_data, get_title, color_palette,get_system_name
import matplotlib.pyplot as plt
all_models = []
set_matplotlib_style()
cmp = sns.color_palette("tab10")

def get_model_name(model_name):
    if "delta" in model_name:
        return model_name.replace("delta-", "#")
    else:
        return "Base Model"

def get_model_alpha(model_name):
    if "delta" in model_name:
        delta_id = int(model_name.split("-")[-1])
        # delta_id is in [1,16], map to [0.3, 0.9]
        delta_alpha = 0.3 + 0.045 * delta_id
        return delta_alpha
    else:
        return 1.0

def get_model_id(model_name):
    global all_models
    all_models = sorted(all_models)
    sorted_all_models = []
    for model in all_models:
        if "delta" in model:
            sorted_all_models.append(model.replace("delta-", "#"))
        else:
            sorted_all_models.append("Base")
    sorted_all_models = sorted(all_models)
    # put the last
    index = sorted_all_models.index(model_name)
    # if not last, return '#'+str(index+1), else return 'Base'
    if index == len(sorted_all_models) - 1:
        return "Base"
    else:
        return "#" + str(index + 1)
    

def plot(args):
    dfs = []
    for file in [x for x in os.listdir(args.input) if x.endswith(".jsonl")]:
        global all_models
        filename = os.path.join(args.input, file)
        metadata, data = parse_data(filename, order=True)
        title = get_title(metadata)
        df = pd.DataFrame(data)
        all_models = df["model"].unique()
        # sort all_models
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
        ncols=2, nrows=1,
        constrained_layout=True, 
        figsize=(9, 3.75),
        sharey=True
    )
    patches = []
    figs = [ax1, ax2]
    types = ["Queuing", "Inference", "Loading"]
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
                alpha=get_model_alpha(row['model']),
            )
            ax.barh(
                y=row["id"],
                width=row["loading_end"] - row["loading_start"],
                left=row["loading_start"],
                color=type_colors["Loading"],
                alpha=get_model_alpha(row['model']),
            )
            ax.barh(
                y=row["id"],
                width=row["inference_end"] - row["first_token_start"],
                left=row["first_token_start"],
                color=type_colors["Inference"],
                alpha=get_model_alpha(row['model']),
            )
            ax.text(
                x=row["inference_end"] + 0.5,
                y=row['id'] + 0.25,
                s=get_model_id(row['model']),
                fontdict=dict(color='black', fontsize=11),
            )
            ax.xaxis.grid(True, alpha=0.5)
    # add legend to the bottom of the figure, in 3 cols and 1 row
    ax1.set_xlabel("(a) Ours")
    ax2.set_xlabel("(b) Baseline")
    ax1.set_ylabel("Task ID")
    
    ax1.invert_yaxis()
    plt.legend(
        handles=patches,
        fontsize=14,
    )
    sns.despine()
    fig.savefig(f"{args.output}/timeline.pdf", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    plot(args)