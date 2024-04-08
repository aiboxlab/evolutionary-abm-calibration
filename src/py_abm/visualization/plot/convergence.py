"""Module for generating a convergence graph."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .entities import Plot


class Convergence(Plot):
    """Classe para geração de gráficos de convergência.

    Exemplo
    --------
    >>> plot = Convergence([0.5, 0.3, 0.24, 0.22, 0.2])
    >>> plot.show()
    >>> plot.save('image.png')
    """

    def __init__(self,
                 values: list[float],
                 labels: list[str] | None = None,
                 title: str = 'Convergence Graph',
                 x_axis: str = 'Milestones',
                 y_axis: str = 'Best Fitness Value') -> None:
        df = pd.DataFrame({x_axis: labels,
                           y_axis: values})
        self._fig = px.line(df,
                            x=x_axis,
                            y=y_axis,
                            title=title)

    def show(self, **kwargs) -> None:
        self._fig.show()

    def save(self,
             fname: str,
             resolution: float = 1.0,
             **kwargs) -> None:
        self._fig.write_image(fname,
                              scale=resolution)
