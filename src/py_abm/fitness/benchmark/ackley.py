"""Ackley benchmark function.
"""
from __future__ import annotations

import numpy as np

from py_abm.entities import Fitness


class Ackley(Fitness):
    def __init__(self,
                 dims: int,
                 lower_bound: float = -32.768,
                 upper_bound: float = 32.768,
                 a: float = 20,
                 b: float = 0.2,
                 c: float = 2 * np.pi):
        self.dims = dims
        self.lower = lower_bound
        self.upper = upper_bound
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self,
                 individuals: np.ndarray,
                 n_parallel: int | None = None,
                 **kwargs) -> np.ndarray:
        d = self.dims
        sum1 = np.sum(individuals * individuals, axis=-1)
        sum2 = np.sum(np.cos(self.c * individuals), axis=-1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = np.exp(sum2 / d)
        result = term1 - term2 + self.a + np.e

        return result

    def evaluate_single(self,
                        individual: np.ndarray,
                        **kwargs) -> np.ndarray:
        return self.evaluate(np.expand_dims(individual,
                                            axis=0)).flatten()

    def info(self) -> dict:
        return {
            'dims': self.dims,
            'bounds': (self.lower, self.upper),
            'n_objectives': 1,
            'objectives_names': ['Ackley']
        }
