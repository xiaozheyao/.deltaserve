import os
import argparse
import pandas as pd
from dstool.plot.style import set_font, set_plotly_theme
from utils import parse_data, get_sys_name, color_palette,get_short_system_name
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.kaleido.scope.mathjax = None

def plot(args):
    print(args)
    files = [x for x in os.listdir(args.input) if x.endswith(".jsonl")]
    df = pd.DataFrame()
    for file in files:
        metadata, data = parse_data(os.path.join(args.input, file))
        focus_group = [x for x in data if x["type"] in ["E2E Latency", "TTFT"]]
        filename = args.input.split("/")[-1].removesuffix(".jsonl")
        partial_df = pd.DataFrame(focus_group)
        partial_df["system"], partial_df["order"] = get_short_system_name(metadata)
        
        df = pd.concat([df, partial_df])
    avg_latency = df.groupby(["system", "type"]).mean().reset_index()
    print(avg_latency)
    # normalized latency to the baseline-1 system
    baseline_latency = avg_latency[avg_latency.system == "Baseline-1"]
    # divide the latency of each type by the baseline-1 latency
    avg_latency = pd.merge(avg_latency, baseline_latency, on="type", suffixes=("", "_baseline"))
    avg_latency["speedup"] =  avg_latency["time_baseline"] / avg_latency["time"]
    # order by order
    avg_latency = avg_latency.sort_values(by="order")
    print(avg_latency)


if __name__ == "__main__":
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
