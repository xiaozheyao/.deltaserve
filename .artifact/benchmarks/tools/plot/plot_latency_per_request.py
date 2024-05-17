import os
import argparse
import pandas as pd
from dstool.plot.style import set_plotly_theme
from vllm.tools.utils import parse_data, get_title, color_palette
from style import set_font
import plotly.express as px
import plotly.graph_objects as go


def plot(args):
    print(args)
    metadata, data = parse_data(args.input)
    title = get_title(metadata)
    filename = args.input.split("/")[-1].removesuffix(".jsonl")
    except_e2e = [x for x in data if x["type"] not in ["E2E Latency", "TTFT", "Arrival", "Finish"]]
    e2e_latency = [x for x in data if x["type"] == "E2E Latency"]
    df = pd.DataFrame(except_e2e)
    e2e_df = pd.DataFrame(e2e_latency)
    # sort df by arrival time
    e2e_df = e2e_df.sort_values(by="arrival_time")
    df = df.sort_values(by="arrival_time")
    df["Time Spent On"] = df["type"]
    # rewrite id
    # map arrival_time to id, in ascending order
    id_map = {v: i for i, v in enumerate(e2e_df["id"])}
    df["id"] = df["id"].map(id_map)
    fig = px.bar(
        df,
        x="id",
        y="time",
        color="Time Spent On",
        title=title,
        color_discrete_sequence=color_palette["general"],
    )
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
