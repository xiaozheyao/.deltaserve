def set_font(fig):
    fig.update_layout(
        font={
            "color": "black",
            "family": "CMU Sans Serif",
        },
        title={
            "font": {
                "family": "CMU Sans Serif",
            },
        },
        xaxis={
            "title": {
                "font": {
                    "family": "CMU Sans Serif",
                },
            },
            "tickfont": {
                "size": 30,
            },
        },
        legend=dict(font=dict(size=28)),
        legend_title=dict(font=dict(size=32)),
    )
    fig.update_xaxes(title=dict(font=dict(size=28)), tickfont_size=30)
    fig.update_yaxes(title=dict(font=dict(size=28)), tickfont_size=30)
    return fig
