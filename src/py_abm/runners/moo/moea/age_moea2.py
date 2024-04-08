"""Esse módulo contém o runner para o
AGE-MOEA2.
"""
from __future__ import annotations

import logging

import numpy as np
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from py_abm.entities import Fitness
from py_abm.runners.base import BaseRunner
from py_abm.runners.utils.pymoo import FitnessProblem

logger = logging.getLogger(__name__)


class AGEMOEA2Runner(BaseRunner):
    """Runner do AGE-MOEA2.
    """

    def __init__(
            self,
            fitness_fn: Fitness,
            n_workers: int,
            population_size: int,
            seed: int = 42781,
            max_fn_evaluation: int = 30000,
            **kwargs) -> None:
        # Pré-condições
        assert fitness_fn.info()['n_objectives'] > 1

        super().__init__(fitness_fn=fitness_fn,
                         seed=seed,
                         n_workers=n_workers,
                         max_fn_evaluation=max_fn_evaluation,
                         **kwargs)

        # Variáveis
        self._pop_size = population_size
        self._alg = AGEMOEA2(pop_size=self._pop_size)

        # Instanciando o problema
        self._problem = FitnessProblem(
            fn=self._fn,
            n_workers=self._n_workers)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        del args, kwargs
        result: Result = minimize(
            problem=self._problem,
            algorithm=self._alg,
            termination=get_termination("n_eval",
                                        self._max_evaluation),
            copy_algorithm=False,
            copy_termination=False,
            save_history=False,
            seed=self._seed)

        # Convert X and F to np arrays
        X = np.array(result.X, copy=False)
        F = np.array(result.F, copy=False)
        return X, F

    def _algorithm_params(self) -> dict:
        config = {
            'algorithm': {
                'name': self._alg.__class__.__name__,
                'provider': 'pymoo',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': {
                    'population_size': self._alg.pop_size,
                }
            }
        }

        return config
