"""Module for generating a boxplot graph."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

from .entities import Plot


class BoxPlot(Plot):
    """Classe para geração de box plots.

    Exemplo
    --------
    >>> plot = BoxPlot([1, 2, 3, 4, 5])
    >>> plot.show()
    """

    def __init__(self,
                 data: list[float],
                 title: str = 'Box Plot',
                 y_axis: str = 'Values'):
        self._fig = px.box(pd.DataFrame({y_axis: data}),
                           y=y_axis,
                           title=title)

    def show(self,
             **kwargs):
        self._fig.show()

    def save(self,
             fname: str,
             resolution: float = 1.0,
             **kwargs):
        self._fig.write_image(fname,
                              scale=resolution)
