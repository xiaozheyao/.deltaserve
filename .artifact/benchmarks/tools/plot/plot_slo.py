import os
import argparse
import numpy as np
import pandas as pd
from dstool.plot.style import set_font, set_plotly_theme
from utils import parse_data, get_sys_name, system_color_mapping, get_system_name
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
        focus_group = [x for x in data if x["type"] in ["E2E Latency", "TTFT"]]
        filename = args.input.split("/")[-1].removesuffix(".jsonl")
        partial_df = pd.DataFrame(focus_group)
        partial_df["system"] = title
        df = pd.concat([df, partial_df])

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=["Time to First Token (TTFT)", "End-to-End Latency"],
        horizontal_spacing=0.015,
        vertical_spacing=0.05,
        x_title="SLO Requirement (s)",
        y_title="Success Rate (%)",
    )
    systems = list(df.system.unique())
    systems = sorted(systems, reverse=True)
    for i, latency_type in enumerate(["TTFT", "E2E Latency"]):
        latency_df = df[df.type == latency_type]
        max_latency = latency_df["time"].max()
        slo_requirements = np.arange(1, max_latency, 0.5)
        slo_data = []
        for system in systems:
            sys_latency_df = latency_df[latency_df.system == system]
            for slo in slo_requirements:
                slo_data.append(
                    {
                        "system": get_system_name(system),
                        "slo": slo,
                        "percentage": len(sys_latency_df[sys_latency_df.time <= slo])
                        / len(sys_latency_df),
                    }
                )
        slo_df = pd.DataFrame(slo_data)
        fig2 = px.line(
            slo_df,
            x="slo",
            y="percentage",
            color="system",
            color_discrete_map=system_color_mapping,
        )

        for datum in fig2["data"]:
            fig.add_trace(
                go.Scatter(
                    x=datum["x"],
                    y=datum["y"],
                    name=datum["name"],
                    mode="lines",
                    line=dict(width=8),
                    marker=dict(color=datum["line"]["color"]),
                    showlegend=True if i == 0 else False,
                ),
                row=1,
                col=i + 1,
            )
    for annotation in fig["layout"]["annotations"]:
        if annotation["text"] == "SLO Requirement (s)":
            annotation["xshift"] = 0
            annotation["yshift"] = -40
        if annotation["text"] == "Success Rate (%)":
            if args.type == "nvme":
                # annotation['yshift'] = 180
                annotation["xshift"] = 0
            else:
                annotation["yshift"] = 0
                annotation["xshift"] = 0
    fig.update_layout(title_x=0.5)
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
    fig.update_xaxes(nticks=4, title_text="")
    fig.update_yaxes(nticks=2)
    set_plotly_theme(fig)
    set_font(fig)
    fig.update_layout(margin=dict(t=50, l=50, r=10, b=5))
    fig.write_image(
        os.path.join(args.output, f"slo_{args.type}.pdf"),
        width=1800,
        height=800,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latency per request")
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument(
        "--type", type=str, choices=["nfs", "nvme"], help="Type of latency"
    )
    parser.add_argument("--ar", type=str, default="0.5")
    args = parser.parse_args()
    plot(args)
