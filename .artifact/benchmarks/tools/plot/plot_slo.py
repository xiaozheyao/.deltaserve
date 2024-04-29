import os
import argparse
import pandas as pd
from dstool.plot.style import set_font, set_plotly_theme
from utils import parse_data, get_sys_name, color_palette
import plotly.express as px
import plotly.graph_objects as go

def plot(args):
    print(args)
    files = [x for x in os.listdir(args.input) if x.endswith(".jsonl")]
    df = pd.DataFrame()
    for file in files:
        metadata, data = parse_data(os.path.join(args.input, file))
        print(file, metadata)
        if metadata['ar'] != args.ar:
            continue
        title = get_sys_name(metadata)
        filename = args.input.split("/")[-1].removesuffix(".jsonl")
        if args.type == "ttft":
            data = [x for x in data if x["type"] == "TTFT"]
        elif args.type == "e2e":
            data = [x for x in data if x["type"] == "E2E Latency"]
        else:
            raise ValueError(f"Unsupported type: {args.type}")
        partial_df = pd.DataFrame(data)
        partial_df['system'] = title
        print(partial_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latency per request")
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument("--type", type=str, choices=['ttft', 'e2e'], help="Type of latency")
    parser.add_argument("--ar", type=str, default="0.5")
    args = parser.parse_args()
    plot(args)
