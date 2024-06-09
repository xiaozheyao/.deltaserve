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
    full_df = prepare_df(args.input, order=True)
    metrics = ["E2E Latency", "TTFT"]
    result_df = []
    for metric in metrics:
        print(f"metric: {metric}")
        sub_df = full_df[full_df["type"] == metric]
        # group by max_deltas and calculate average
        sub_df = sub_df.groupby("max_deltas").mean().reset_index()
        print(sub_df)
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    args = parser.parse_args()
    plot(args)
