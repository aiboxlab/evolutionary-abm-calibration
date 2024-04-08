"""Esse módulo contém a classe para
realizar scatter plots.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from .entities import Plot


class Scatter(Plot):
    """Gera um scatter plot.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 x_column: str,
                 y_column: str,
                 opacity: float = 1.0,
                 range_x: tuple[float, float] | None = None,
                 range_y: tuple[float, float] | None = None,
                 color_discrete_map: dict | None = None,
                 symbol_map: dict | None = None,
                 color_column: str | None = None,
                 symbol_column: str | None = None,
                 size_column: str | None = None,
                 legend_font_size: int | None = None,
                 title: str = 'Scatter') -> None:
        self._df = df
        self._x_column = x_column
        self._rx = range_x
        self._y_column = y_column
        self._ry = range_y
        self._color_column = color_column
        self._symbol_column = symbol_column
        self._size_column = size_column
        self._color_discrete_map = color_discrete_map
        self._symbol_map = symbol_map
        self._title = title
        self._opacity = opacity
        self._leg_size = legend_font_size

    def show(self, **kwargs) -> None:
        fig = self._create_fig()
        fig.update_layout(**kwargs)
        fig.show()

    def save(self,
             fname: str,
             resolution: float = 1.0,
             **kwargs) -> None:
        fig = self._create_fig()
        fig.update_layout(**kwargs)
        fig.write_image(fname,
                        scale=resolution)

    def _create_fig(self) -> Figure:
        fig = px.scatter(self._df,
                         x=self._x_column,
                         y=self._y_column,
                         color=self._color_column,
                         symbol=self._symbol_column,
                         size=self._size_column,
                         range_x=self._rx,
                         range_y=self._ry,
                         title=self._title,
                         opacity=self._opacity,
                         color_discrete_map=self._color_discrete_map,
                         symbol_map=self._symbol_map)
        fig.update_traces(marker=dict(line=dict(width=0.8,
                                                color='DarkSlateGrey')))
        if self._leg_size is not None:
            fig.update_layout(legend=dict(font=dict(size=self._leg_size),
                                          itemsizing='constant'))

        return fig
