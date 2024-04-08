"""Esse módulo contém o runner para o SHADE.
"""
from __future__ import annotations

import logging

import numpy as np
from shade_ils.shade import SHADEOptimizer

from py_abm.entities import Fitness
from py_abm.runners.base import BaseRunner
from py_abm.runners.utils import shade_ils as utils

logger = logging.getLogger(__name__)


class SHADERunner(BaseRunner):
    """Runner do SHADE para calibração de ABMs através
    do shade_ils.
    """

    def __init__(self,
                 fitness_fn: Fitness,
                 n_workers: int,
                 population_size: int,
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 history_size: int = 100,
                 **kwargs) -> None:
        # Pré-condições
        assert fitness_fn.info()['n_objectives'] == 1

        super().__init__(fitness_fn=fitness_fn,
                         seed=seed,
                         n_workers=n_workers,
                         max_fn_evaluation=max_fn_evaluation,
                         **kwargs)

        # Armazenando configurações
        self._pop_size = population_size
        self._history_size = history_size

        # Instanciando o otimizador
        self._optimizer = SHADEOptimizer(
            fn=utils.FitnessFunction(fn=self._fn,
                                     n_workers=self._n_workers),
            population_size=self._pop_size,
            max_evaluations=self._max_evaluation,
            seed=self._seed,
            history_size=self._history_size)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        result = self._optimizer.optimize()
        solution = np.broadcast_to(result.solution,
                                   (1, self._fn.info()['dims']))
        fitness = np.broadcast_to(result.fitness,
                                  (1, 1))

        return solution, fitness

    def _algorithm_params(self) -> dict:
        config = {
            'algorithm': {
                'name': self._optimizer.__class__.__name__,
                'provider': 'shade_ils',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': {
                    'population_size': self._pop_size,
                    'history_size': self._history_size,
                    'seed': self._seed
                }
            }
        }

        return config
