"""Esse módulo contém o runner para o PSO.
"""
from __future__ import annotations

import logging

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.result import Result
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from py_abm.entities import Fitness
from py_abm.runners.base import BaseRunner
from py_abm.runners.utils.pymoo import FitnessProblem

logger = logging.getLogger(__name__)


class PSORunner(BaseRunner):
    """Runner do SPSO para calibração de ABMs através
    do pymoo.

    Referência: https://pymoo.org/algorithms/soo/pso.html
    Baseado em: Z. Zhan, J. Zhang, Y. Li, and H. S. Chung.
                Adaptive particle swarm optimization (2009)
    """

    def __init__(self,
                 fitness_fn: Fitness,
                 n_workers: int,
                 population_size: int,
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 w: float = 0.7213475204444817,
                 c1: float = 1.1931471805599454,
                 c2: float = 1.1931471805599454,
                 initial_velocity: str = 'random',
                 start_strategy: str | np.ndarray = 'random',
                 adaptative: bool = True,
                 max_velocity_rate: float = 0.20,
                 mutate_best: bool = False,
                 **kwargs) -> None:
        # Pré-condições
        assert fitness_fn.info()['n_objectives'] == 1

        super().__init__(fitness_fn=fitness_fn,
                         seed=seed,
                         n_workers=n_workers,
                         max_fn_evaluation=max_fn_evaluation,
                         **kwargs)

        if isinstance(start_strategy, str):
            # Adicionar outros métodos para geração
            #   da população inicial.
            start_strategy = LHS()

        # Instanciando PSO
        self._alg = PSO(pop_size=population_size,
                        w=w,
                        c1=c1,
                        c2=c2,
                        initial_velocity=initial_velocity,
                        sampling_strategy=0,
                        adaptive=adaptative,
                        max_velocity_rate=max_velocity_rate,
                        pertube_best=mutate_best,
                        sampling=start_strategy)

        # Instanciando o problema
        self._problem = FitnessProblem(
            fn=self._fn,
            n_workers=self._n_workers)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        result: Result = minimize(
            problem=self._problem,
            algorithm=self._alg,
            termination=get_termination("n_eval",
                                        self._max_evaluation),
            copy_algorithm=False,
            copy_termination=False,
            save_history=False,
            seed=self._seed)
        solution = np.broadcast_to(result.X,
                                   (1, self._fn.info()['dims']))
        fitness = np.broadcast_to(result.F,
                                  (1, 1))
        return solution, fitness

    def _algorithm_params(self) -> dict:
        config = {
            'algorithm': {
                'name': self._alg.__class__.__name__,
                'provider': 'pymoo',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': {
                    'population_size': self._alg.pop_size,
                    'w': self._alg.w,
                    'c1': self._alg.c1,
                    'c2': self._alg.c2,
                    'adaptative': self._alg.adaptive,
                    'max_velocity_rate': self._alg.max_velocity_rate,
                    'mutate_best': self._alg.pertube_best,
                    'seed': self._seed,
                    'initial_velocity': self._alg.initial_velocity,
                }
            }
        }

        return config
