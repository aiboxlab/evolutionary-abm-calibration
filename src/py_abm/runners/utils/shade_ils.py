"""Esse módulo contém funções e funções utilitárias
para utilizar em conjunto com a biblioteca shade_ils.
"""
from __future__ import annotations

import numpy as np
from shade_ils.entities import FitnessFunction as _FF

from py_abm.entities import Fitness


class FitnessFunction(_FF):
    """Representa uma função de fitness do py_abm
    como um Problem do shade_ils.
    """

    def __init__(self,
                 fn: Fitness,
                 n_workers: int):
        """Construtor.

        Args:
            fn (Fitness): função de fitness.
            n_workers (int): quantidade de workers para execução
                paralela.
        """
        self._fn = fn
        self._n_workers = n_workers

    def call(self, population: np.ndarray) -> np.ndarray:
        return self._fn.evaluate(population,
                                 n_parallel=self._n_workers)

    def info(self) -> dict:
        lower, upper = self._fn.info()['bounds']
        dims = self._fn.info()['dims']
        return {
            'lower': np.array(lower, dtype=np.float32),
            'upper': np.array(upper, dtype=np.float32),
            'dimension': dims
        }

    def name(self) -> str:
        return self._fn.__class__.__name__
