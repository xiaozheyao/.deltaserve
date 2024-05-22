import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH
from utils import autolabel, set_matplotlib_style
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

SAVEPATH = ".artifact/benchmarks/figures"
set_matplotlib_style()

full_df = prepare_df(DEFAULT_PATH)

total_models = 33
total_max_deltas = 24

metrics = ["E2E Latency", "TTFT"]
result_df = []
for metric in metrics:
    df = full_df[full_df['type'] == metric]
    df = df[df['total_models'] == total_models]
    # for each row, if max_deltas>0, then choose the row with max_deltas

    distributions = df['distribution'].unique()
    ars = df['ar'].unique()
    for dist in distributions:
        if dist != "zipf:1.5" and dist != "distinct":
            for ar in ars:
                sub_df = df[df['distribution'] == dist]
                sub_df = sub_df[sub_df['ar'] == ar]
                systems = sub_df['sys_name'].unique()
                max_deltas = sub_df['max_deltas'].unique()
                for system in systems:
                    sub_df_sys = sub_df[sub_df['sys_name'] == system]
                    if system == "Baseline-1":
                        sub_df_sys = sub_df_sys[sub_df_sys['max_deltas'] == 0]
                    else:
                        sub_df_sys = sub_df_sys[sub_df_sys['max_deltas'] == total_max_deltas]
                    result_df.append({
                        "system": system,
                        "distribution": dist,
                        "ar": ar,
                        "mean": sub_df_sys['time'].mean(),
                        "metric": metric,
                    })
                    
result_df = pd.DataFrame(result_df)
baseline_df = result_df[result_df['system'] == "Baseline-1"]
# normalize to baseline
result_df = result_df.merge(baseline_df, on=["distribution", "ar", "metric"], suffixes=("", "_baseline"))
print(result_df)
# calculate relative mean
result_df['mean'] = 10 * result_df['mean'] / result_df['mean_baseline']

wanted_distribution = "azure"
result_df = result_df[result_df['distribution'] == wanted_distribution]

baseline_df = result_df[result_df['system'] == "Baseline-1"]
delta_df = result_df[result_df['system'] == "+Delta"]
prefetch_df = result_df[result_df['system'] == "+Prefetch"]
# set index as ar and mean
baseline_df = baseline_df.set_index(["ar", "metric"])["mean"].unstack()
delta_df = delta_df.set_index(["ar", "metric"])["mean"].unstack()
prefetch_df = prefetch_df.set_index(["ar", "metric"])["mean"].unstack()

grid_params = dict(width_ratios=[1, 1])
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75))

x = np.arange(1, 3)
width = 0.22
p1 = ax1.bar(
    x - width,
    baseline_df.loc[["2.0","6.0"], "E2E Latency"],
    width,
    label="Baseline",
    alpha=0.8,
    linewidth=1,
    edgecolor="k",
)
p2 = ax1.bar(
    x, delta_df.loc[["2.0","6.0"], "E2E Latency"], width, label="+Delta", alpha=0.8, linewidth=1, edgecolor="k"
)
p3 = ax1.bar(
    x + width, prefetch_df.loc[["2.0", "6.0"], "E2E Latency"], width, label="+Prefetch", alpha=0.8, linewidth=1, edgecolor="k"
)

p4 = ax2.bar(
    x - width,
    baseline_df.loc[["2.0","6.0"], "TTFT"],
    width,
    label="+Delta",
    alpha=0.8,
    linewidth=1,
    edgecolor="k",
)
p5 = ax2.bar(
    x, delta_df.loc[["2.0","6.0"], "TTFT"], width, label="+Delta", alpha=0.8, linewidth=1, edgecolor="k"
)
p6 = ax2.bar(
    x + width,
    prefetch_df.loc[["2.0", "6.0"], "TTFT"],
    width,
    label="+Prefetch",
    alpha=0.8,
    linewidth=1,
    edgecolor="k",
)

autolabel(p1, ax1)
autolabel(p2, ax1)
autolabel(p3, ax1)
autolabel(p4, ax2)
autolabel(p5, ax2)
autolabel(p6, ax2)

ax1.set_xlabel(f"(a) Latency")
ax1.set_ylabel(f"Normalized Time")
ax1.set_xticks(x)
ax1.set_xticklabels(["2.0","6.0"])
ax1.set_xlim(0.5, 2.5)
# ax1.set_ylim(0, 10)
ax1.grid(axis="y", linestyle=":")

ax2.set_xlabel(f"(b) TTFT")
ax2.set_ylabel(f"Normalized Time")
ax2.set_xticks(x)
ax2.set_xticklabels(["2.0","6.0"])
ax2.set_xlim(0.5, 2.5)
# ax2.set_ylim(0, 10)
ax2.grid(axis="y", linestyle=":")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles=handles,
    labels=labels,
    ncols=5,
    bbox_to_anchor=(0.18, 1.145),
    loc=2,
)

sns.despine()
fig.savefig(f"{SAVEPATH}/latency_improv.pdf", bbox_inches="tight")