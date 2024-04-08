"""Esse módulo contém o runner para
o DECC.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from decc import Problem
from decc.optimizers.decc import DECCOptimizer

from py_abm.entities import Fitness
from py_abm.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class DECCRunner(BaseRunner):
    """Runner do DECC(-O,-H) para calibração
    de ABMs através do decc.
    """

    def __init__(
            self,
            fitness_fn: Fitness,
            n_workers: int,
            population_size: int,
            variant: Literal['O', 'H'],
            seed: int = 42781,
            F: float = 0.5,
            CR: float = 0.7,
            max_fn_evaluation: int = 30000,
            **kwargs) -> None:
        # Pré-condições
        assert fitness_fn.info()['n_objectives'] == 1

        super().__init__(fitness_fn=fitness_fn,
                         seed=seed,
                         n_workers=n_workers,
                         max_fn_evaluation=max_fn_evaluation,
                         **kwargs)

        # Armazenando configurações
        self._sub_pop_size = population_size
        self._F = F
        self._CR = CR

        # Instanciando o problema
        problem = Problem(objective=lambda x: self._fn.evaluate(x, None),
                          dims=self._fn.info()['dims'],
                          lower_bound=self._fn.info()['bounds'][0],
                          upper_bound=self._fn.info()['bounds'][1])

        # Instanciando o otimizador
        self._optimizer = DECCOptimizer(
            problem=problem,
            subpopulation_size=self._sub_pop_size,
            max_fn=self._max_evaluation,
            grouping='halve' if variant.lower() == 'H' else 'dims',
            F=self._F,
            CR=self._CR,
            seed=self._seed,
            ensure_evaluations=True)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        del args, kwargs
        result = self._optimizer.optimize()
        solution = np.broadcast_to(result['best_solution'],
                                   (1, self._fn.info()['dims']))
        fitness = np.broadcast_to(result['best_fitness'],
                                  (1, 1))

        return solution, fitness

    def _algorithm_params(self) -> dict:
        config = {
            'algorithm': {
                'name': str(self._optimizer),
                'provider': 'decc',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': {
                    'sub_population_size': self._sub_pop_size,
                    'F': self._F,
                    'CR': self._CR,
                    'n_subproblems': self._optimizer.parameters()['n_subproblems'],
                    'seed': self._seed
                }
            }
        }

        return config
