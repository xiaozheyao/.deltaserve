import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH
from utils import autolabel, set_matplotlib_style
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

def plot(args):
    SAVEPATH = args.savepath
    set_matplotlib_style()
    full_df = prepare_df(args.path, order=False)
    # if 3090 in full_df['filename'], add a new column called model_type and fill in 7b
    result_df = []
    metrics = ["E2E Latency","TTFT"]
    for metric in metrics:
        metric_id = metric.lower().replace(" ", "_")
        df = full_df[full_df["type"] == metric]
        tp_sizes = df['tp_size'].unique()
        for tp_size in tp_sizes:
            tp_df = df[df['tp_size'] == tp_size]
            distribution = tp_df['distribution'].unique()
            for dist in distribution:
                dist_df = tp_df[tp_df['distribution'] == dist]
                sys_name = f"TP-{tp_size}"
                model_sizes = dist_df['model_size'].unique()
                for model_size in model_sizes:
                    model_size_df = dist_df[dist_df['model_size'] == model_size]
                    result_df.append({
                        "system": sys_name,
                        "distribution": dist,
                        "mean": model_size_df["time"].mean(),
                        "metric": metric,
                        "model_size": model_size,
                        "tp_size": tp_size,
                    })
    result_df = pd.DataFrame(result_df)
    result_df = result_df[result_df["distribution"] == "zipf:1.5"]
    sgs_df = result_df[result_df["model_size"] == "7B"]
    pjlab_df = result_df[result_df["model_size"] == "13B"]
    print(pjlab_df)
    print(sgs_df)
    sgs_df = sgs_df.set_index(["metric", "system"])["mean"].unstack()
    pjlab_df = pjlab_df.set_index(["metric", "system"])["mean"].unstack()
    grid_params = dict(width_ratios=[1, 1])
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75)
    )
    x = np.arange(1, 3)
    width = 0.22
    p1 = ax1.bar(
        x - 0.5 * width,
        sgs_df.loc[("E2E Latency")],
        width,
        label="TP-1",
        alpha=0.8,
        linewidth=1,
        edgecolor="k",
    )
    p2 = ax1.bar(
        x + 0.5 * width,
        sgs_df.loc[("TTFT")],
        width,
        label="TP-2",
        alpha=0.8,
        linewidth=1,
        edgecolor="k",
    )
    p3 = ax2.bar(
        x - 0.5 * width,
        pjlab_df.loc[("E2E Latency")],
        width,
        label="TP-2",
        alpha=0.8,
        linewidth=1,
        edgecolor="k",
    )
    p4 = ax2.bar(
        x + 0.5 * width,
        pjlab_df.loc[("TTFT")],
        width,
        label="TP-4",
        alpha=0.8,
        linewidth=1,
        edgecolor="k",
    )

    autolabel(p1, ax1, prec=1)
    autolabel(p2, ax1, prec=1)
    autolabel(p3, ax2, prec=1)
    autolabel(p4, ax2, prec=1)
    
    ax1.set_xlabel(f"(a) 7B")
    ax1.set_ylabel(f"Time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["1x 3090", "2x 3090"])
    ax1.set_xlim(0.5, 2.5)
    ax1.grid(axis="y", linestyle=":")

    ax2.set_xlabel(f"(b) 13B")
    ax2.set_ylabel(f"")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["2x A800", "4x A800"])
    ax2.set_xlim(0.5, 2.5)
    ax2.grid(axis="y", linestyle=":")

    handles, labels = ax1.get_legend_handles_labels()
    labels = ['E2E Latency', 'TTFT']
    fig.legend(
        handles=handles,
        labels=labels,
        ncols=3,
        bbox_to_anchor=(0.25, 1.145),
        loc=2,
    )
    sns.despine()
    fig.savefig(f"{SAVEPATH}/tensor_parallel.pdf", bbox_inches="tight")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--savepath", type=str, default=".")
    args = parser.parse_args()
    plot(args)