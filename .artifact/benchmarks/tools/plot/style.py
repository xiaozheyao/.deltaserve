def set_font(fig):
    fig.update_layout(
        font={
            "family": "Latin Modern",
        },
        font_color="black",
        title_font_color="black",
        legend_title_font_color="black",
        title={
            'font': {
                "family": "Latin Modern",
            },
        },
        xaxis={
            'title': {
                'font': {
                    "family": "Latin Modern",
                },
            },
            'tickfont': {
                "size": 30,
            },
        },
        legend=dict(font=dict(size=28)),
        legend_title=dict(font=dict(size=32)),
    )
    fig.update_xaxes(title=dict(font=dict(size=28)), tickfont_size=30)
    fig.update_yaxes(title=dict(font=dict(size=28)), tickfont_size=30)
    return fig