import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH
from utils import autolabel, set_matplotlib_style
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

cmp = sns.color_palette("tab10")

def plot(args):
    SAVEPATH = args.savepath
    set_matplotlib_style()
    full_df = prepare_df(args.path, order=False)
    
    metrics = ["E2E Latency","TTFT"]
    for metric in metrics:
        metric_id = metric.lower().replace(" ", "_")
        result_df = []
        df = full_df[full_df["type"] == metric]
        systems = df["sys_name"].unique()
        for system in systems:
            # get num_of deltas
            sub_df_sys = df[df["sys_name"] == system]
            if system == "+Delta":
                max_deltas = sub_df_sys["max_deltas"].unique()
                for max_delta in max_deltas:
                    sys_name = f"{system} (N={max_delta})"
                    delta_df = sub_df_sys[sub_df_sys["max_deltas"] == max_delta]
                    distributions = delta_df["distribution"].unique()
                    for dist in distributions:
                        dist_df = delta_df[delta_df["distribution"] == dist]
                        arrival_rate = dist_df["ar"].unique()
                        for ar in arrival_rate:
                            ar_df = dist_df[dist_df["ar"] == ar]
                            result_df.append(
                                {
                                    "system": sys_name,
                                    "distribution": dist,
                                    "ar": ar_df["ar"].iloc[0],
                                    "mean": ar_df["time"].mean(),
                                    "metric": metric,
                                }
                            )
            elif system == "Baseline-1":
                distributions = sub_df_sys["distribution"].unique()
                for dist in distributions:
                    dist_df = sub_df_sys[sub_df_sys["distribution"] == dist]
                    arrival_rate = dist_df["ar"].unique()
                    for ar in arrival_rate:
                        ar_df = dist_df[dist_df["ar"] == ar]
                        result_df.append(
                            {
                                "system": system,
                                "distribution": dist,
                                "ar": ar_df["ar"].iloc[0],
                                "mean": ar_df["time"].mean(),
                                "metric": metric,
                            }
                        )
        result_df = pd.DataFrame(result_df)
        # pick either ar=0.5 or 2
        result_df = result_df[result_df["ar"].isin(["0.5", "2.0"])]
        baseline_df = result_df[result_df["system"] == "Baseline-1"]
        delta_8_df = result_df[result_df["system"] == "+Delta (N=8)"]
        delta_12_df = result_df[result_df["system"] == "+Delta (N=12)"]
        baseline_df = baseline_df.set_index(["distribution","ar"])["mean"].unstack()
        delta_8_df = delta_8_df.set_index(["distribution","ar"])["mean"].unstack()
        delta_12_df = delta_12_df.set_index(["distribution","ar"])["mean"].unstack()
        
        grid_params = dict(width_ratios=[1, 1])
        fig, (ax1, ax2, ax3) = plt.subplots(
            ncols=3, nrows=1, constrained_layout=True, figsize=(9, 3.75)
        )
        x = np.arange(1, 3)
        width = 0.22
        p1 = ax1.bar(
            x - width,
            baseline_df.loc[("azure")],
            width,
            label="vLLM+SCB",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
        )
        p2 = ax1.bar(
            x,
            delta_8_df.loc[("azure")],
            width,
            label="+Delta (N=8)",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=cmp[2],
        )
        p3 = ax1.bar(
            x + width,
            delta_12_df.loc[("azure")],
            width,
            label="+Delta (N=12)",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=cmp[3],
        )
        p4 = ax2.bar(
            x - width,
            baseline_df.loc[("uniform")],
            width,
            label="vLLM+SCB",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
        )
        p5 = ax2.bar(
            x,
            delta_8_df.loc[("uniform")],
            width,
            label="+Delta (N=8)",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=cmp[2],
        )
        p6 = ax2.bar(
            x + width,
            delta_12_df.loc[("uniform")],
            width,
            label="+Delta (N=12)",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=cmp[3],
        )
        p7 = ax3.bar(
            x - width,
            baseline_df.loc[("zipf:1.5")],
            width,
            label="vLLM+SCB",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
        )
        p8 = ax3.bar(
            x,
            delta_8_df.loc[("zipf:1.5")],
            width,
            label="+Delta (N=8)",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=cmp[2],
        )
        p9 = ax3.bar(
            x + width,
            delta_12_df.loc[("zipf:1.5")],
            width,
            label="+Delta (N=12)",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=cmp[3],
        )
        autolabel(p1, ax1, prec=0)
        autolabel(p2, ax1, prec=0)
        autolabel(p3, ax1, prec=0)
        autolabel(p4, ax2, prec=0)
        autolabel(p5, ax2, prec=0)
        autolabel(p6, ax2, prec=0)
        autolabel(p7, ax3, prec=0)
        autolabel(p8, ax3, prec=0)
        autolabel(p9, ax3, prec=0)
        
        ax1.set_xlabel(f"(a) Azure")
        ax1.set_ylabel(f"E2E Latency (s)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["0.5", "2.0"])
        ax1.set_xlim(0.5, 2.5)
        ax1.grid(axis="y", linestyle=":")

        ax2.set_xlabel(f"(b) Uniform")
        ax2.set_ylabel(f"")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["0.5", "2.0"])
        ax2.set_xlim(0.5, 2.5)
        # ax2.set_ylim(0, 10)
        ax2.grid(axis="y", linestyle=":")

        ax3.set_xlabel(f"(c) Zipf:1.5")
        ax3.set_ylabel(f"")
        ax3.set_xticks(x)
        ax3.set_xticklabels(["0.5", "2.0"])
        ax3.set_xlim(0.5, 2.5)
        # ax2.set_ylim(0, 10)
        ax3.grid(axis="y", linestyle=":")

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles=handles,
            labels=labels,
            ncols=3,
            bbox_to_anchor=(0.05, 1.145),
            loc=2,
        )
        sns.despine()
        fig.savefig(f"{SAVEPATH}/latency_improv_{metric_id}.pdf", bbox_inches="tight")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--savepath", type=str, default=".")
    args = parser.parse_args()
    plot(args)