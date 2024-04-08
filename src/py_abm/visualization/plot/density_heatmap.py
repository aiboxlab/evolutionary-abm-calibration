"""Esse módulo contém os códigos para geração
de um Heatmap.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from .entities import Plot


class DensityHeatmap(Plot):
    """
    Example
    --------

    Use the DensityHeatmap to plot the heat map graph
    `Heat map <https://en.wikipedia.org/wiki/Heat_map>`_:

    .. code-block:: python
        :linenos:

    >>> df = pd.read_csv('Dataset.csv')
    >>> plot = DensityHeatmap(df, 'x_col', 'y_col', 'z_col',
                              histfunc='avg', nbinsx=21, nbinsy=21)
    >>> plot.show(width=800, height=600)
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 x_col_name: str,
                 y_col_name: str,
                 z_col_name: str,
                 nbinsx: int = 20,
                 nbinsy: int = 20,
                 color_continuous_scale: str = "Viridis",
                 histfunc: str = "sum") -> None:
        """
        Constructor for the DensityHeatmap class.

        Parameters:
        - dataframe: pd.DataFrame, input DataFrame.
        - x_col_name: str, column name for the x-axis values.
        - y_col_name: str, column name for the y-axis values.
        - z_col_name: str | None, column name for the z-axis values.
        - nbinsx: int, optional, number of bins for x-axis. Default is 20.
        - nbinsy: int, optional, number of bins for y-axis. Default is 20.
        - color_continuous_scale: str, optional, color scale for the heatmap. Default is "Viridis".
        - histfunc: str, optional, aggregation function for the z-axis values. Default is "sum".
        """
        self.df = dataframe
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name
        self.z_col_name = z_col_name
        self.nbinsx = nbinsx
        self.nbinsy = nbinsy
        self.color_continuous_scale = color_continuous_scale
        self.histfunc = histfunc

    def show(self, **kwargs) -> None:
        """
        Generates and shows a density heatmap plot using Plotly Express.

        Parameters:
        - xaxis_title: str, title for the x-axis.
        - yaxis_title: str, title for the y-axis.
        - height: int, optional, height of the plot in pixels. Default is 500.
        - width: int, optional, width of the plot in pixels. Default is 300.
        """
        xaxis_title: str = kwargs.get('xaxis_title', None)
        yaxis_title: str = kwargs.get('yaxis_title', None)
        height: int = kwargs.get('height', 500)
        width: int = kwargs.get('width', 300)

        fig = self._create_fig(width,
                               height,
                               x_axis_title=xaxis_title,
                               y_axis_title=yaxis_title)
        fig.show()

    def save(self,
             fname: str,
             resolution: float = 1.0,
             **kwargs):
        """
        Saves the density heatmap plot to an image file.

        Parameters:
        - filename: str, name of the output file.
        - extension: str, optional, format of the output file. Default is "png".
        - width: int, optional, width of the plot in pixels. Default is 800.
        - height: int, optional, height of the plot in pixels. Default is 600.
        """
        xaxis_title: str = kwargs.get('xaxis_title', None)
        yaxis_title: str = kwargs.get('yaxis_title', None)
        height: int = kwargs.get('height', 800)
        width: int = kwargs.get('width', 600)

        fig = self._create_fig(width,
                               height,
                               x_axis_title=xaxis_title,
                               y_axis_title=yaxis_title)
        fig.write_image(fname,
                        scale=resolution)

    def _create_fig(self,
                    width: int,
                    height: int,
                    x_axis_title: str | None = None,
                    y_axis_title: str | None = None) -> Figure:
        """Cria a figura de heatmap.

        Args:
            width (int): largura da figura.
            height (int): altura da figura.

        Returns:
            Figure: heatmap.
        """
        if x_axis_title is None:
            x_axis_title = self.x_col_name

        if y_axis_title is None:
            y_axis_title = self.y_col_name

        fig = px.density_heatmap(self.df,
                                 x=self.x_col_name,
                                 y=self.y_col_name,
                                 z=self.z_col_name,
                                 nbinsx=self.nbinsx,
                                 nbinsy=self.nbinsy,
                                 color_continuous_scale=self.color_continuous_scale,
                                 histfunc=self.histfunc,
                                 labels=None)

        fig.update_traces(colorbar=dict(title=None))
        fig.update_layout(xaxis_title=x_axis_title,
                          yaxis_title=y_axis_title,
                          height=height,
                          width=width,
                          showlegend=False)

        return fig
