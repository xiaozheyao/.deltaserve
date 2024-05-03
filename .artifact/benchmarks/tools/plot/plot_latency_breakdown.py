import os
import argparse
import pandas as pd
from dstool.plot.style import set_font, set_plotly_theme
from utils import parse_data, get_sys_name, color_palette
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
        if args.type == "nfs":
            if metadata["is_nvme"]:
                continue
        else:
            if not metadata["is_nvme"]:
                continue
        if metadata["bitwidth"] == 2:
            continue
        title = get_sys_name(metadata)
        except_e2e = [x for x in data if x["type"] not in ["E2E Latency", "TTFT"]]
        filename = args.input.split("/")[-1].removesuffix(".jsonl")
        partial_df = pd.DataFrame(except_e2e)
        partial_df["system"] = title
        df = pd.concat([df, partial_df])
    systems = list(df.system.unique())
    systems = sorted(systems, reverse=True)
    system_names = ["Baseline-1", "Ours", "Ours+"]
    fig = make_subplots(
        rows=1,
        cols=len(systems),
        shared_yaxes=True,
        subplot_titles=system_names,
        horizontal_spacing=0.015,
        vertical_spacing=0.05,
        x_title="Request ID",
        y_title="Time (s)",
    )
    for i, system in enumerate(systems):
        system_df = df[df.system == system]
        fig2 = px.bar(
            system_df,
            x="id",
            y="time",
            color="type",
            color_discrete_sequence=color_palette["general"],
        )
        for fig2_data in fig2["data"]:
            fig.add_trace(
                go.Bar(
                    x=fig2_data["x"],
                    y=fig2_data["y"],
                    marker=fig2_data["marker"],
                    name=fig2_data["name"],
                    showlegend=True if i == 0 else False,
                ),
                row=1,
                col=i + 1,
            )
    fig.update_layout(title_x=0.5)
    fig.update_layout(barmode="stack")
    # fig.update_yaxes(type="log")
    fig.update_layout(
        width=1800,
        height=800,
    )
    fig.update_xaxes(title_font=dict(size=24), tickfont_size=24)
    fig.update_annotations(
        font_size=36,
        font_color="black",
        font_family="CMU Sans Serif",
    )
    for annotation in fig["layout"]["annotations"]:
        if annotation["text"] == "Request ID":
            annotation["yshift"] -= 5
        if annotation["text"] == "Time (s)":
            if args.type == "nvme":
                annotation["yshift"] = -60
            else:
                annotation["yshift"] = 60
    fig.update_xaxes(nticks=4)
    fig.update_yaxes(nticks=4)
    set_plotly_theme(fig)
    set_font(fig)
    fig.update_layout(margin=dict(t=50, l=20, r=10))
    fig.write_image(
        os.path.join(args.output, f"latency_{args.type}.pdf"),
        width=1800,
        height=800,
    )


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
