import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH
from utils import autolabel, set_matplotlib_style
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

set_matplotlib_style()
cmp = sns.color_palette("tab10")
sns.set_style("ticks")

def create_slo_data(df, max_value):
    assert len(df["sys_name"].unique()) == 1, f"Multiple systems in the dataframe, got {df['sys_name'].unique()}"
    assert len(df["max_deltas"].unique()) == 1, f"Multiple deltas in the dataframe, got {df['max_deltas'].unique()}"
    assert len(df["distribution"].unique()) == 1, "Multiple distributions in the dataframe"
    assert len(df["ar"].unique()) == 1, "Multiple arrival rates in the dataframe"
    
    system = df["sys_name"].unique()[0]
    max_deltas = df['max_deltas'].unique()[0]
    distribution = df['distribution'].unique()[0]
    ar = df['ar'].unique()[0]
    slo_requirements = np.arange(1, max_value, 0.5)
    slo_data = []
    if system == 'Baseline-1':
        sys_name = "Baseline"
    else:
        sys_name = f"{system} (N={max_deltas})"
    for slo in slo_requirements:
        slo_data.append({
            "system": sys_name,
            "slo": slo,
            "percentage": len(df[df["time"] <= slo]) / len(df),
            "distribution": distribution,
            "ar": ar
        })
    slo_df = pd.DataFrame(slo_data)
    return slo_df
    
def plot(args):
    SAVEPATH = args.savepath
    set_matplotlib_style()
    full_df = prepare_df(args.path, order=False)
    metrics = ["E2E Latency", "TTFT"]
    for metric in metrics:
        print(f"Handling {metric}, preparing data...")
        slo_dfs = []
        metric_id = metric.lower().replace(" ", "_")
        met_df = full_df[full_df['type'] == metric]
        systems = met_df["sys_name"].unique()
        max_met_value = met_df['time'].max()
        distributions = met_df["distribution"].unique()
        for dist in distributions:
            dist_df = met_df[met_df["distribution"] == dist]
            arrival_rates = dist_df["ar"].unique()
            for ar in arrival_rates:
                ar_df = dist_df[dist_df["ar"] == ar]
                max_lat = ar_df["time"].max()
                for system in systems:
                    sys_df = ar_df[ar_df["sys_name"] == system]
                    if system == "+Delta":
                        max_deltas = sys_df["max_deltas"].unique()
                        for max_delta in max_deltas:
                            delta_df = sys_df[sys_df["max_deltas"] == max_delta]
                            slo_dfs.append(create_slo_data(delta_df, max_lat))
                    else:
                        slo_dfs.append(create_slo_data(sys_df, max_lat))

        slo_df = pd.concat(slo_dfs)
        target_distribution = 'azure'
        target_ar = ['0.5', '2.0']
        slo_df = slo_df[slo_df['distribution'] == target_distribution]
        slo_df = slo_df[slo_df['ar'].isin(target_ar)]
        ar_1_slo = slo_df[slo_df['ar'] == '0.5']
        ar_2_slo = slo_df[slo_df['ar'] == '2.0']
        # set index
        print(ar_1_slo[ar_1_slo['system'] == 'Baseline']['percentage'])
        
        grid_params = dict(width_ratios=[1, 1])
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75))
        x1, x2 = ar_1_slo['slo'].unique(), ar_2_slo['slo'].unique()
        
        linestyles = [":", "-", "--"]
        ax1.plot(x1, ar_1_slo[ar_1_slo['system'] == 'Baseline']['percentage'], linestyles[0], linewidth=3, alpha=0.9, color=cmp[0], label="Baseline")
        ax1.plot(x1, ar_1_slo[ar_1_slo['system'] == '+Delta (N=8)']['percentage'], linestyles[1], linewidth=3, alpha=0.9, color=cmp[1], label='+Delta (N=8)')
        ax1.plot(x1, ar_1_slo[ar_1_slo['system'] == '+Delta (N=12)']['percentage'], linestyles[2], linewidth=3, alpha=0.9, color=cmp[2], label='+Delta (N=12)')
        
        ax2.plot(x2, ar_2_slo[ar_2_slo['system'] == 'Baseline']['percentage'], linestyles[0], linewidth=3, alpha=0.9, color=cmp[0], label="Baseline")
        ax2.plot(x2, ar_2_slo[ar_2_slo['system'] == '+Delta (N=8)']['percentage'], linestyles[1], linewidth=3, alpha=0.9, color=cmp[1], label='+Delta (N=8)')
        ax2.plot(x2, ar_2_slo[ar_2_slo['system'] == '+Delta (N=12)']['percentage'], linestyles[2], linewidth=3, alpha=0.9, color=cmp[2], label='+Delta (N=12)')
        
        ax1.set_xlabel(f"(a) Arrival Rate=0.5")
        ax2.set_xlabel(f"(b) Arrival Rate=2.0")
        ax1.set_ylabel(f"Success Rate (%)")
        ax1.grid(linestyle=":")
        ax2.grid(linestyle=":")
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, ncols=5, bbox_to_anchor=(0.1, 1.145), loc=2, columnspacing=1.5, handletextpad=0.5)
        
        sns.despine()
        fig.savefig(f"{SAVEPATH}/slo_{metric_id}.pdf", bbox_inches="tight")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--savepath", type=str, default=".")
    args = parser.parse_args()
    plot(args)