from time import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def make_figure(timestamps, data, future_timestamps, predicted, danger_levels, labels, height):
    fig = make_subplots(rows=3, cols=1, subplot_titles=labels)

    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data[:, i],
                mode='lines',
                line={'color': '#E81F64'},
                name=''
                ),
            row=i+1, col=1
        )
        fig.add_vline(
            x=timestamps[-1], 
            line_color='#633a48'
        )
        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=predicted[:, i],
                mode='lines',
                line={'color': '#e6b8c7'},
                name=''
                ),
            row=i+1, col=1
        )
        if data[:, i].min() <= danger_levels[i][0]:
            fig.add_hline(
                y=danger_levels[i][0],
                line_color='#ffd503',
                row=i+1, col=1
            )
            fig.add_hline(
                y=danger_levels[i][1],
                line_color='#ff4203',
                row=i+1, col=1
            )

    fig.update_layout(
        showlegend=False,
        height=height
    )

    return fig