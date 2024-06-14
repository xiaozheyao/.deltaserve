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
    metrics = ["E2E Latency","TTFT"]
    print(full_df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--savepath", type=str, default=".")
    args = parser.parse_args()
    plot(args)