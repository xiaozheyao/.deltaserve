import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH
from utils import autolabel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

SAVEPATH = ".artifact/benchmarks/figures"

pd.set_option("display.max_columns", 500)

sns.set_style("ticks")
font = {
    "font.family": "Roboto",
    "font.size": 12,
}
sns.set_style(font)
paper_rc = {
    "lines.linewidth": 3,
    "lines.markersize": 10,
}
sns.set_context("paper", font_scale=2, rc=paper_rc)
cmp = sns.color_palette("tab10")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

full_df = prepare_df(DEFAULT_PATH)

total_models = 33
total_max_deltas = 24
wanted_distribution = "azure"
wanted_columns = ['Arrival', 'Finish']
df = full_df[full_df['total_models'] == total_models]

distributions = df['distribution'].unique()
df = df[df['distribution'] == wanted_distribution]
systems = df['sys_name'].unique()
ars = df['ar'].unique()

print(ars)
for system in systems:
    for ar in ars:
        sub_df = df[df['ar'] == ar]
        sub_df = sub_df[sub_df['sys_name'] == system]
        max_deltas = sub_df['max_deltas'].unique()
        if system == "Baseline-1":
            sub_df = sub_df[sub_df['max_deltas'] == 0]
        else:
            sub_df = sub_df[sub_df['max_deltas'] == total_max_deltas]
        print(f"System: {system}")
        sub_df = sub_df[sub_df['type'].isin(wanted_columns)]
        # minimal arrival time
        min_arrival = sub_df[sub_df['type'] == 'Arrival']['time'].min()
        max_finish = sub_df[sub_df['type'] == 'Finish']['time'].max()
        total_span = max_finish - min_arrival
        total_requests = len(sub_df) // 2
        print(f"AR: {ar}, Total Requests: {total_requests}, Total Span: {total_span:.2f}, Throughput: {100 * total_requests / total_span:.2f}")

# result_df = pd.DataFrame(result_df)
# baseline_df = result_df[result_df['system'] == "Baseline-1"]
# # normalize to baseline
# result_df = result_df.merge(baseline_df, on=["distribution", "ar", "metric"], suffixes=("", "_baseline"))
# print(result_df)
# # calculate relative mean
# result_df['mean'] =result_df['mean'] / result_df['mean_baseline']

# wanted_distribution = "uniform"
# result_df = result_df[result_df['distribution'] == wanted_distribution]
# baseline_df = result_df[result_df['system'] == "Baseline-1"]
# delta_df = result_df[result_df['system'] == "+Delta"]

# # set index as ar and mean
# baseline_df = baseline_df.set_index(["ar", "metric"])["mean"].unstack()
# delta_df = delta_df.set_index(["ar", "metric"])["mean"].unstack()


# grid_params = dict(width_ratios=[1, 1])
# fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75))

# x = np.arange(1, 3)
# print(x)
# width = 0.22
# p1 = ax1.bar(
#     x - width,
#     baseline_df.loc[["3.0", "9.0"], "E2E Latency"],
#     width,
#     label="Baseline",
#     alpha=0.8,
#     linewidth=1,
#     edgecolor="k",
# )
# p2 = ax1.bar(
#     x, delta_df.loc[["3.0", "9.0"], "E2E Latency"], width, label="+Delta", alpha=0.8, linewidth=1, edgecolor="k"
# )
# # p3 = ax1.bar(
# #     x + width, df.loc[["Seren", "Kalos"], "fail_rate_gpu"] * 100, width, label="Failed", alpha=0.8, linewidth=1, edgecolor="k"
# # )

# p4 = ax2.bar(
#     x - width,
#     baseline_df.loc[["3.0", "9.0"], "TTFT"],
#     width,
#     label="+Delta",
#     alpha=0.8,
#     linewidth=1,
#     edgecolor="k",
# )
# p5 = ax2.bar(
#     x, delta_df.loc[["3.0", "9.0"], "TTFT"], width, label="+Delta", alpha=0.8, linewidth=1, edgecolor="k"
# )
# # p6 = ax2.bar(
# #     x + width,
# #     df.loc[["Seren", "Kalos"], "fail_rate_gpu_time"] * 100,
# #     width,
# #     label="Failed",
# #     alpha=0.8,
# #     linewidth=1,
# #     edgecolor="k",
# # )

# autolabel(p1, ax1)
# autolabel(p2, ax1)
# # autolabel(p3, ax1)
# autolabel(p4, ax2)
# autolabel(p5, ax2)
# # autolabel(p6, ax2)

# ax1.set_xlabel(f"(a) Latency")
# ax1.set_ylabel(f"Normalized Time")
# ax1.set_xticks(x)
# ax1.set_xticklabels(["3", "9"])
# ax1.set_xlim(0.5, 2.5)
# # ax1.set_ylim(0, 100)
# ax1.grid(axis="y", linestyle=":")

# ax2.set_xlabel(f"(b) TTFT")
# ax2.set_ylabel(f"Normalized Time")
# ax2.set_xticks(x)
# ax2.set_xticklabels(["3", "9"])
# ax2.set_xlim(0.5, 2.5)
# # ax2.set_ylim(0, 100)
# ax2.grid(axis="y", linestyle=":")

# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(
#     handles=handles,
#     labels=labels,
#     ncols=5,
#     bbox_to_anchor=(0.18, 1.145),
#     loc=2,
#     #   columnspacing=1, handletextpad=0.2
# )

# sns.despine()
# fig.savefig(f"{SAVEPATH}/latency_improv.pdf", bbox_inches="tight")