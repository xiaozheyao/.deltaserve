import os
import argparse
import pandas as pd
from dstool.plot.style import set_font, set_plotly_theme
from vllm.tools.utils import parse_data, get_sys_name, color_palette
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def get_system_name(sys):
    sys = sys.lower()
    if "vllm" in sys:
        return "Baseline-1"
    if "deltaserve" in sys:
        if "prefetch" not in sys:
            return "Ours"
        else:
            return "Ours+"
    return "Unknown"


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
        subplot_titles=["TTFT", "End-to-End"],
        horizontal_spacing=0.015,
        vertical_spacing=0.05,
        x_title="",
        y_title="Time (s)",
    )
    for i, latency_type in enumerate(["TTFT", "E2E Latency"]):
        latency_df = df[df.type == latency_type]
        # for each system, calculate average latency
        avg_latency = latency_df.groupby(["system"]).mean().reset_index()
        avg_latency["system"] = avg_latency["system"].apply(get_system_name)
        # calculate baseline-1/ours ratio
        ours_plus = avg_latency[avg_latency.system == "Ours+"].time.values[0]
        ours = avg_latency[avg_latency.system == "Ours"].time.values[0]
        baseline = avg_latency[avg_latency.system == "Baseline-1"].time.values[0]
        ours_plus_ratio = baseline / ours_plus
        ours_ratio = baseline / ours
        fig2 = px.bar(avg_latency, x="system", y="time")
        for fig2_data in fig2["data"]:
            fig.add_trace(
                go.Bar(
                    x=fig2_data["x"],
                    y=fig2_data["y"],
                    marker=fig2_data["marker"],
                    name=fig2_data["name"],
                    showlegend=False,
                ),
                row=1,
                col=i + 1,
            )
            if args.type == "nfs":
                if i == 0:
                    y1 = 45
                    y2 = 150
                else:
                    y1 = 85
                    y2 = 120
            if args.type == "nvme":
                if i == 0:
                    y1 = 12
                    y2 = 15
                else:
                    y1 = 12
                    y2 = 52
            fig.add_annotation(
                go.layout.Annotation(
                    x=0,
                    y=y1,
                    text=f"{ours_plus_ratio:.2f}x",
                ),
                row=1,
                col=i + 1,
            )
            fig.add_annotation(
                go.layout.Annotation(
                    x=1,
                    y=y2,
                    text=f"{ours_ratio:.2f}x",
                ),
                row=1,
                col=i + 1,
            )

    fig.update_layout(title_x=0.5)
    fig.update_layout(barmode="group")
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
                annotation["yshift"] = 200
                annotation["xshift"] = 0
            else:
                annotation["xshift"] = 0
                annotation["yshift"] = 250

    fig.update_xaxes(nticks=4, title_text="")
    fig.update_yaxes(nticks=4)
    set_plotly_theme(fig)
    set_font(fig)
    fig.update_layout(margin=dict(t=50, l=20, r=10, b=5))
    fig.write_image(
        os.path.join(args.output, f"latencies_{args.type}.pdf"),
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
