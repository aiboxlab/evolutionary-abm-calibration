"""Esse módulo contém a definição das interfaces 
relativas ao Runner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Runner(ABC):
    """Essa classe representa um runner.

    Um runner, representa todo o conjunto de configurações 
    necessárias para calibrar os parâmetros de um ABM
    usando um dado algoritmo evolucionário.

    Em outras palavras, o runner é composto por:
        - Algoritmo (e configurações);
        - Função de fitness;
        - Definição do problema/experimento;
        - Métodos para execução e recuperação dos resultados;
    """

    @abstractmethod
    def run(self,
            *args,
            **kwargs) -> dict:
        """Esse método executa o runner seguindo as
        configurações definidas.
        """

    @abstractmethod
    def result(self,
               *args,
               **kwargs) -> dict | None:
        """Esse método retorna os resultados obtidos 
        após execução do Runner.
        """

    @abstractmethod
    def config(self) -> dict:
        """Esse método retorna a configuração do
        runner.
        """
