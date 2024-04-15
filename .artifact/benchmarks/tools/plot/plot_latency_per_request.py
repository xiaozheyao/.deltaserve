import os
import argparse
import pandas as pd
from dstool.plot.style import set_font, set_plotly_theme
from utils import parse_data
import plotly.express as px

def plot(args):
    print(args)
    metadata, data = parse_data(args.input)
    filename = args.input.split('/')[-1].removesuffix('.jsonl')
    df = pd.DataFrame(data)
    print("df ready")
    print(df)
    fig = px.bar(df, x='id', y='time', color='type')
    set_font(fig)
    set_plotly_theme(fig)
    fig.write_image(os.path.join(args.output, f"{filename}.png"), width=1200)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot latency per request')
    parser.add_argument('--input', type=str, help='Input file')
    parser.add_argument('--output', type=str, help='Output folder')
    args = parser.parse_args()
    plot(args)