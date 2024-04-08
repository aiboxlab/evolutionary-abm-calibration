"""Esse módulo contém o runner para o SHADE.
"""
from __future__ import annotations

import logging

import numpy as np
from shade_ils.shade_ils import SHADEILSOptimizer

from py_abm.entities import Fitness
from py_abm.runners.base import BaseRunner
from py_abm.runners.utils import shade_ils as utils

logger = logging.getLogger(__name__)


class SHADEILSRunner(BaseRunner):
    """Runner do SHADE para calibração de ABMs através
    do shade_ils.
    """

    def __init__(self,
                 fitness_fn: Fitness,
                 n_workers: int,
                 population_size: int,
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 evaluations_gs: int = None,
                 evaluations_de: int = None,
                 evaluations_ls: int = None,
                 threshold_reset_ls: float = 0.05,
                 history_size: int = 100,
                 **kwargs) -> None:
        # Pré-condições
        assert fitness_fn.info()['n_objectives'] == 1

        super().__init__(fitness_fn=fitness_fn,
                         seed=seed,
                         n_workers=n_workers,
                         max_fn_evaluation=max_fn_evaluation,
                         **kwargs)

        if evaluations_gs is None:
            evaluations_gs = max(self._max_evaluation // 6, 10)

        if evaluations_de is None:
            evaluations_de = max(self._max_evaluation // 6, 10)

        if evaluations_ls is None:
            evaluations_ls = max(self._max_evaluation // 6, 10)

        # Armazenando configurações
        self._pop_size = population_size
        self._history_size = history_size
        self._threshold = threshold_reset_ls
        self._evaluations_gs = evaluations_gs
        self._evaluations_de = evaluations_de
        self._evaluations_ls = evaluations_ls

        # Instanciando o otimizador
        self._optimizer = SHADEILSOptimizer(
            fn=utils.FitnessFunction(fn=self._fn,
                                     n_workers=self._n_workers),
            population_size=self._pop_size,
            max_evaluations=self._max_evaluation,
            seed=self._seed,
            history_size=self._history_size,
            evaluations_de=self._evaluations_de,
            evaluations_ls=self._evaluations_ls,
            evaluations_gs=self._evaluations_gs,
            threshold=self._threshold)

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
                    'evaluations_de': self._evaluations_de,
                    'evaluations_gs': self._evaluations_gs,
                    'evaluations_ls': self._evaluations_ls,
                    'threshold_ls_reset': self._threshold,
                    'seed': self._seed
                }
            }
        }

        return config
