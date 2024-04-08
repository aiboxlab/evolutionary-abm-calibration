"""Module for generating a convergence graph for R1 and R2 variables."""
from __future__ import annotations

import plotly.graph_objects as go

from .entities import Plot


class BoxPlotR1R2(Plot):
    """Classe responsável por gerar um gráfico com 2 Box plots: um para
    os valores de r1; outro para os valores de r2.

    Exemplo
    -------
    >>> boxplot = BoxPlotR1R2([0.5,0.5,0.5], [0.5,0.5,0.5])
    >>> boxplot.show()  # Mostra a imagem
    >>> boxplot.save('imagem.png')
    """

    def __init__(self,
                 r1_values: list[float],
                 r2_values: list[float],
                 title: str = 'Boxplot of r1 and r2'):
        self._r1 = r1_values
        self._r2 = r2_values
        variables = ['r1'] * len(self._r1) + ['r2'] * len(self._r2)
        values = self._r1 + self._r2
        trace = go.Box(y=values,
                       x=variables,
                       boxpoints=False)
        layout = go.Layout(title=title,
                           yaxis=dict(title='Values'))
        self._fig = go.Figure(data=[trace],
                              layout=layout)
        self._fig.update_layout(yaxis_range=[0, 1])

    def show(self, **kwargs):
        self._fig.show()

    def save(self,
             fname: str,
             resolution: float = 1.0,
             **kwargs):
        """Salvar a figura,

        Args:
            fname (str): nome do arquivo.
        """
        self._fig.write_image(fname,
                              scale=resolution)

