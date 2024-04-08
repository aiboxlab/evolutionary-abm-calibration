"""Esse módulo contém o runner para o NSGA-II.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Type

import numpy as np

from py_abm.fitness.abm import Market, Objective, OptimizationType
from py_abm.runners.abm.base import BaseABMRunnerMO
from py_abm.runners.abm.utils import ABMFitnessWithCallback
from py_abm.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class GenericABMRunnerMO(BaseABMRunnerMO):
    """Runner genérico para runners multi-objetivo do ABM.
    """

    def __init__(
            self,
            runner: Type[BaseRunner],
            market: Market,
            optimization_type: OptimizationType,
            time_steps: int,
            mc_runs: int,
            n_groups: int | None,
            objectives: list[Objective],
            discretize_search_space: bool,
            add_adj_rand_score: bool,
            n_workers: int,
            save_directory: Path,
            seed: int = 42781,
            max_fn_evaluation: int = 30000,
            milestone_interval: int = 100,
            log_frequency: int = 100,
            ensure_save_directory_is_empty: bool = True,
            min_max_bounds: np.ndarray | None = None,
            **runner_kwargs) -> None:
        # Inicializando classe base
        super().__init__(
            market=market,
            optimization_type=optimization_type,
            time_steps=time_steps,
            mc_runs=mc_runs,
            n_groups=n_groups,
            objectives=objectives,
            discretize_search_space=discretize_search_space,
            n_workers=n_workers,
            save_directory=save_directory,
            seed=seed,
            max_fn_evaluation=max_fn_evaluation,
            milestone_interval=milestone_interval,
            add_adj_rand_score=add_adj_rand_score,
            log_frequency=log_frequency,
            ensure_save_directory_is_empty=ensure_save_directory_is_empty)

        # Variáveis auxiliares
        self._min_max_bounds = min_max_bounds

        if min_max_bounds is not None:
            m = self._fn.info()['n_objectives']
            assert min_max_bounds.shape == (2, m)

            def post_eval(x: np.ndarray) -> np.ndarray:
                min = self._min_max_bounds[0]
                max = self._min_max_bounds[1]
                return (x - min) / (max - min)
        else:
            def post_eval(x: np.ndarray) -> np.ndarray:
                return x

        # Aplicando wrappers na funcção de fitness
        fn = self._fn
        fn = ABMFitnessWithCallback(fn=fn,
                                    callback=self.fitness_callback,
                                    post_evaluation=post_eval)

        # Instanciando runner com função do ABM
        self._runner = runner(
            fitness_fn=fn,
            seed=seed,
            n_workers=n_workers,
            max_fn_evaluation=self._max_evaluation,
            **runner_kwargs)

    def config(self, *arg, **kwargs):
        base = super().config(*arg, **kwargs)
        if self._min_max_bounds is not None:
            base['normalization'] = {
                'kind': 'MinMax',
                'minimum': self._min_max_bounds[0].tolist(),
                'maximum': self._min_max_bounds[1].tolist()
            }
        return base

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self._runner._run()

    def _algorithm_params(self) -> dict:
        return self._runner._algorithm_params()
