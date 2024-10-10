import os
import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH
from utils import autolabel, set_matplotlib_style
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from vllm.tools.utils import parse_data, walk_through_files, get_mixed_system_name
cmp = sns.color_palette("tab10")

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
        short_sys_name, sys_order = get_mixed_system_name(metadata)
        results = pd.DataFrame(results)
        results["filename"] = input_file
        if '3090' in input_file:
            results['model_size'] = '7B'
        else:
            results['model_size'] = '13B'
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

def process_entry(df):
    model = df['model'].unique()
    # select df where "lora" is in df['model']
    lora_df = df[df['model'].str.contains("lora")]
    delta_df = df[df['model'].str.contains("delta")]
    lora_mean = lora_df['time'].mean()
    delta_mean = delta_df['time'].mean()
    return {
        'lora_mean': lora_mean,
        'delta_mean': delta_mean,
    }
def plot(args):
    SAVEPATH = args.savepath
    set_matplotlib_style()
    full_df = prepare_df(args.path, order=False)
    metrics = ["E2E Latency","TTFT"]
    result_df = []
    for metric in metrics:
        metric_id = metric.lower().replace(" ", "_")
        df = full_df[full_df["type"] == metric]
        systems = df["sys_name"].unique()
        for system in systems:
            sub_df_sys = df[df["sys_name"] == system]
            dists = sub_df_sys["distribution"].unique()
            for dist in dists:
                dist_df = sub_df_sys[sub_df_sys["distribution"] == dist]
                ars = dist_df["ar"].unique()
                for ar in ars:
                    res = process_entry(dist_df[dist_df["ar"] == ar])
                    result_df.append({
                        "system": system,
                        "distribution": dist,
                        "ar": ar,
                        "mean": res['lora_mean'],
                        "models": "lora",
                        "metric": metric,
                    })
                    result_df.append({
                        "system": system,
                        "distribution": dist,
                        "ar": ar,
                        "mean": res['delta_mean'],
                        "models": "delta",
                        "metric": metric,
                    })
    full_df = pd.DataFrame(result_df)
    
    for metric in metrics:
        fig, (ax1, ax2) = plt.subplots(
            ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75)
        )
        result_df = full_df[full_df["metric"] == metric]
        metric_id = metric.lower().replace(" ", "_")
        result_df = result_df.set_index(["system", "models"])['mean'].unstack()
        x = np.arange(1, 2)
        width = 0.22
        p1 = ax1.bar(
            x - width,
            result_df.loc[("Swap + LoRA", "lora")],
            width,
            label="vLLM+SCB",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
        )
        p2 = ax1.bar(
            x,
            result_df.loc[("Delta + LoRA", "lora")],
            width,
            label="Ours",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=cmp[2],
        )
        p3 = ax2.bar(
            x - width,
            result_df.loc[("Swap + LoRA", "delta")],
            width,
            label="vLLM+SCB",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
        )
        p4 = ax2.bar(
            x,
            result_df.loc[("Delta + LoRA", "delta")],
            width,
            label="Our",
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=cmp[2],
        )
        autolabel(p1, ax1, prec=0)
        autolabel(p2, ax1, prec=0)
        autolabel(p3, ax2, prec=0)
        autolabel(p4, ax2, prec=0)
        ax1.set_xlabel(f"(a) LoRA")
        ax2.set_xlabel(f"(b) Delta")
        ax1.set_ylabel(f"{metric} (s)")
        ax1.set_xticks(x)
        # ax1.set_xticklabels(["0.5", "1.0"])
        # ax1.set_xlim(0.5, 2.5)
        ax1.grid(axis="y", linestyle=":")
        print(result_df)
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles=handles,
            labels=labels,
            ncols=3,
            bbox_to_anchor=(0.05, 1.145),
            loc=2,
        )
        sns.despine()
        fig.savefig(f"{SAVEPATH}/mixed_latency_improv_{metric_id}.pdf", bbox_inches="tight")
        
        print('done')
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--savepath", type=str, default=".")
    args = parser.parse_args()
    plot(args)