"""Esse módulo contém o runner para
o DECC-G.
"""
from __future__ import annotations

import logging

import numpy as np
from decc import Problem
from decc.optimizers.decc_g import DECCGOptimizer

from py_abm.entities import Fitness
from py_abm.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class DECCGRunner(BaseRunner):
    """Runner do DECC-G.
    """

    def __init__(
            self,
            fitness_fn: Fitness,
            n_workers: int,
            population_size: int,
            de_evaluations: int,
            sansde_evaluations: int,
            n_subproblems: int,
            seed: int = 42781,
            F: float = 0.5,
            CR: float = 0.7,
            weights_bound: tuple[float, float] = (-10.0, 10.0),
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
        self._pop_size = population_size
        self._F = F
        self._CR = CR
        self._n_subproblems = n_subproblems
        self._de_eval = de_evaluations
        self._sansde_eval = sansde_evaluations
        self._w_bounds = weights_bound

        # Instanciando o problema
        problem = Problem(objective=lambda x: self._fn.evaluate(x, None),
                          dims=self._fn.info()['dims'],
                          lower_bound=self._fn.info()['bounds'][0],
                          upper_bound=self._fn.info()['bounds'][1])

        # Instanciando o otimizador
        self._optimizer = DECCGOptimizer(
            problem=problem,
            population_size=self._pop_size,
            n_subproblems=self._n_subproblems,
            sansde_evaluations=self._sansde_eval,
            de_evaluations=self._de_eval,
            max_fn=self._max_evaluation,
            F=self._F,
            CR=self._CR,
            weights_bound=self._w_bounds,
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
                    'population_size': self._pop_size,
                    'weight_bounds': self._w_bounds,
                    'F': self._F,
                    'CR': self._CR,
                    'de_evaluations': self._de_eval,
                    'sansde_evaluations': self._sansde_eval,
                    'n_subproblems': self._n_subproblems,
                    'seed': self._seed
                }
            }
        }

        return config
