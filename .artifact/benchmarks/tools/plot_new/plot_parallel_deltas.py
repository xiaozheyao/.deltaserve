import matplotlib
import pandas as pd
import seaborn as sns
from utils import set_matplotlib_style
from vllm.tools.utils import parse_data, get_title, color_palette
import matplotlib.pyplot as plt
from data_utils import prepare_df, DEFAULT_PATH

set_matplotlib_style()
cmp = sns.color_palette("tab10")


def plot(args):
    full_df = prepare_df(args.input)
    print(full_df)
    metrics = ["E2E Latency", "TTFT"]
    result_df = []
    for metric in metrics:
        sub_df = full_df[full_df["type"] == metric]
        print(f"Metric: {metric}")
        print(sub_df['max_deltas'].unique())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    args = parser.parse_args()
    plot(args)
