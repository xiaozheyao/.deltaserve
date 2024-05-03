import os
import argparse
import pandas as pd
from dstool.plot.style import set_plotly_theme
from utils import parse_data, get_title, color_palette
from style import set_font
import plotly.express as px
import plotly.graph_objects as go


def plot(args):
    print(args)
    metadata, data = parse_data(args.input)
    title = get_title(metadata)
    filename = args.input.split("/")[-1].removesuffix(".jsonl")
    except_e2e = [x for x in data if x["type"] not in ["E2E Latency", "TTFT"]]
    e2e_latency = [x for x in data if x["type"] == "E2E Latency"]
    df = pd.DataFrame(except_e2e)
    e2e_df = pd.DataFrame(e2e_latency)
    df["Time Spent On"] = df["type"]
    fig = px.bar(
        df,
        x="id",
        y="time",
        color="Time Spent On",
        title=title,
        color_discrete_sequence=color_palette["general"],
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=e2e_df.id,
    #         y=e2e_df.time,
    #         mode="lines",
    #         name="E2E Latency",
    #     )
    # )
    # set title font
    fig.update_yaxes(title_text="Time (s)")
    fig.update_xaxes(title_text="Request ID")
    fig.update_layout(title_x=0.5)
    set_plotly_theme(fig)
    set_font(fig)
    fig.write_image(
        os.path.join(args.output, f"{filename}.pdf"), width=1200, height=800
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latency per request")
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--output", type=str, help="Output folder")
    args = parser.parse_args()
    plot(args)
