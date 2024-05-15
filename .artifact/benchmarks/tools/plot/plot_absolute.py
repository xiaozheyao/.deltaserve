import os
import argparse
import pandas as pd
from dstool.plot.style import set_font, set_plotly_theme
from vllm.tools.utils import parse_data, get_sys_name, color_palette, get_short_system_name
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def plot(args):
    distributions = [
        x for x in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, x))
    ]
    for dist in distributions:
        arrival_rates = [
            x
            for x in os.listdir(os.path.join(args.input, dist))
            if os.path.isdir(os.path.join(args.input, dist, x))
        ]
        for ar in arrival_rates:
            ar_str = ar.split("=")[1]
            files = [
                x
                for x in os.listdir(os.path.join(args.input, dist, ar))
                if x.endswith(".jsonl")
            ]
            df = pd.DataFrame()
            distribution = []
            for file in files:
                metadata, data = parse_data(os.path.join(args.input, dist, ar, file))
                distribution.append(metadata["distribution"])
                focus_group = [x for x in data if x["type"] in ["E2E Latency", "TTFT"]]
                filename = args.input.split("/")[-1].removesuffix(".jsonl")
                partial_df = pd.DataFrame(focus_group)
                partial_df["system"], partial_df["order"] = get_short_system_name(
                    metadata
                )
                df = pd.concat([df, partial_df])
            distribution = list(set(distribution))
            assert len(distribution) == 1, "Multiple distributions found"
            distribution = distribution[0]
            avg_latency = df.groupby(["system", "type"]).mean().reset_index()
            # normalized latency to the baseline-1 system
            # divide the latency of each type by the baseline-1 latency
            # order by order
            avg_latency = avg_latency.sort_values(by="order")
            for metric in ["E2E Latency", "TTFT"]:
                subdf = avg_latency[avg_latency.type == metric]
                # set line width
                print(subdf)
                fig = px.line(subdf, x="system", y="time", title="")
                fig.update_traces(line={"width": 6})
                set_plotly_theme(fig)
                fig.update_layout(margin=dict(t=50, l=20, r=10, b=15))
                set_font(fig)
                fig.update_yaxes(
                    title_text="Speedup",
                    tickfont=dict(size=36),
                    title_font=dict(size=36),
                )
                # set font size
                fig.update_xaxes(
                    title_text="", tickfont=dict(size=36), title_font=dict(size=36)
                )

                fig.write_image(
                    os.path.join(
                        args.output, f"ablation_{distribution}_{ar_str}_{metric}.pdf"
                    ),
                    width=800,
                    height=500,
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
