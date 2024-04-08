"""Esse módulo contém a definição das interfaces 
relativas à função de fitness.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Plot(ABC):
    """Determina os métodos que todas
    funções de fitness devem possuir.
    """

    @abstractmethod
    def show(self, **kwargs) -> None:
        """Mostra o gráfico.
        """

    @abstractmethod
    def save(self,
             fname: str,
             resolution: float = 1.0,
             **kwargs) -> None:
        """Salva o gráfico para um arquivo
        de imagem.

        Args:
            fname (str): nome do arquivo.
            resolution (float): taxa de resolução.
        """
