"""Esse módulo possui a definição da classe
de histograma.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px

from .entities import Plot


class Histogram(Plot):
    """Gera um histograma.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 x_column: str,
                 x_axis_bins: dict | None = None,
                 color_column: str | None = None, 
                 color_discrete_map: dict[str, str] | None = None,
                 histunc: str = 'count',
                 y_axis_title: str | None = None,
                 bargap: float | None = None,
                 title: str = 'Histogram',
                 **kwargs) -> None:
        self._fig = px.histogram(df,
                                 x=x_column,
                                 color=color_column,
                                 title=title,
                                 histfunc=histunc,
                                 color_discrete_map=color_discrete_map,
                                 **kwargs)

        if y_axis_title is not None:
            self._fig.update_layout(yaxis_title=y_axis_title)

        if bargap is not None:
            self._fig.update_layout(bargap=bargap)

        if x_axis_bins is not None:
            self._fig.update_traces(xbins=x_axis_bins)

    def show(self, **kwargs) -> None:
        self._fig.show()

    def save(self,
             fname: str,
             resolution: float = 1.0,
             **kwargs) -> None:
        self._fig.write_image(fname,
                              scale=resolution)
