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
                result_df.append({
                    "system": sys_name,
                    "distribution": dist,
                    "mean": dist_df["time"].mean(),
                    "metric": metric,
                })
    result_df = pd.DataFrame(result_df)
    tp_2_df = result_df[result_df["system"] == "TP-2"]
    tp_4_df = result_df[result_df["system"] == "TP-4"]
    tp_2_df = tp_2_df.set_index(["distribution", "metric"])["mean"].unstack()
    tp_4_df = tp_4_df.set_index(["distribution", "metric"])["mean"].unstack()
    grid_params = dict(width_ratios=[1, 1])
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75)
    )
    x = np.arange(1, 2)
    width = 0.22
    p1 = ax1.bar(
        x - width,
        tp_2_df.loc[("zipf:1.5", "E2E Latency")],
        width,
        label="E2E",
        alpha=0.8,
        linewidth=1,
        edgecolor="k",
    )
    p2 = ax1.bar(
        x,
        tp_4_df.loc[("zipf:1.5", "E2E Latency")],
        width,
        label="+Delta (N=8)",
        alpha=0.8,
        linewidth=1,
        edgecolor="k",
    )
    p3 = ax1.bar(
        x - width,
        tp_2_df.loc[("zipf:1.5", "E2E Latency")],
        width,
        label="E2E",
        alpha=0.8,
        linewidth=1,
        edgecolor="k",
    )
    p4 = ax1.bar(
        x,
        tp_4_df.loc[("zipf:1.5", "E2E Latency")],
        width,
        label="+Delta (N=8)",
        alpha=0.8,
        linewidth=1,
        edgecolor="k",
    )
    # p4 = ax2.bar(
    #     x - width,
    #     baseline_df.loc[("uniform")],
    #     width,
    #     label="Baseline",
    #     alpha=0.8,
    #     linewidth=1,
    #     edgecolor="k",
    # )
    # p5 = ax2.bar(
    #     x,
    #     delta_8_df.loc[("uniform")],
    #     width,
    #     label="+Delta (N=8)",
    #     alpha=0.8,
    #     linewidth=1,
    #     edgecolor="k",
    # )
    # p6 = ax2.bar(
    #     x + width,
    #     delta_12_df.loc[("uniform")],
    #     width,
    #     label="+Delta (N=12)",
    #     alpha=0.8,
    #     linewidth=1,
    #     edgecolor="k",
    # )
    # p7 = ax3.bar(
    #     x - width,
    #     baseline_df.loc[("zipf:1.5")],
    #     width,
    #     label="Baseline",
    #     alpha=0.8,
    #     linewidth=1,
    #     edgecolor="k",
    # )
    # p8 = ax3.bar(
    #     x,
    #     delta_8_df.loc[("zipf:1.5")],
    #     width,
    #     label="+Delta (N=8)",
    #     alpha=0.8,
    #     linewidth=1,
    #     edgecolor="k",
    # )
    # p9 = ax3.bar(
    #     x + width,
    #     delta_12_df.loc[("zipf:1.5")],
    #     width,
    #     label="+Delta (N=12)",
    #     alpha=0.8,
    #     linewidth=1,
    #     edgecolor="k",
    # )
    autolabel(p1, ax1, prec=0)
    # autolabel(p2, ax1, prec=0)
    # autolabel(p3, ax1, prec=0)
    # autolabel(p4, ax2, prec=0)
    # autolabel(p5, ax2, prec=0)
    # autolabel(p6, ax2, prec=0)
    # autolabel(p7, ax3, prec=0)
    # autolabel(p8, ax3, prec=0)
    # autolabel(p9, ax3, prec=0)
    
    # ax1.set_xlabel(f"(a) Azure")
    # ax1.set_ylabel(f"E2E Latency (s)")
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(["0.5", "2.0"])
    # ax1.set_xlim(0.5, 2.5)
    # ax1.grid(axis="y", linestyle=":")

    # ax2.set_xlabel(f"(b) Uniform")
    # ax2.set_ylabel(f"")
    # ax2.set_xticks(x)
    # ax2.set_xticklabels(["0.5", "2.0"])
    # ax2.set_xlim(0.5, 2.5)
    # # ax2.set_ylim(0, 10)
    # ax2.grid(axis="y", linestyle=":")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        ncols=3,
        bbox_to_anchor=(0.05, 1.145),
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