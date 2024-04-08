"""Esse módulo contém a definição das interfaces 
relativas à função de fitness.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Fitness(ABC):
    """Determina os métodos que todas
    funções de fitness devem possuir.
    """

    @abstractmethod
    def evaluate(self,
                 individuals: np.ndarray,
                 n_parallel: int | None = None,
                 **kwargs) -> np.ndarray:
        """Principal método de avaliação. Permite avaliar
        um conjunto de indivíduos (representados como Numpy Arrays)

        Args:
            individuals (np.ndarray): array com os indivíduos.
            n_parallel (int | None, optional): quantidade de workers. Defaults to None.

        Returns:
            np.ndarray: resultado da avaliação de fitness.
        """

    @abstractmethod
    def evaluate_single(self,
                        individual: np.ndarray,
                        **kwargs) -> np.ndarray:
        """Método que permite avaliar um único
        indivíduo. É preferível utilizar o método `evaluate`
        diretamente.

        Args:
            individual (np.ndarray): indivíduo.

        Returns:
            np.ndarray: valor da função de fitness.
        """

    @abstractmethod
    def info(self) -> dict:
        """Retorna informações sobre essa função de
        fitness.

        Returns:
            dict: informações.
        """
