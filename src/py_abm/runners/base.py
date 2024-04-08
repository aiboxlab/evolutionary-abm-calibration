"""Esse módulo contém a classe
base para runners.
"""
from __future__ import annotations

import logging
import time
from abc import abstractmethod

import numpy as np

from py_abm.entities import Fitness, Runner

logger = logging.getLogger(__name__)


class BaseRunner(Runner):
    def __init__(self,
                 fitness_fn: Fitness,
                 seed: int,
                 n_workers: int,
                 max_fn_evaluation: int,
                 **kwargs):
        self._fn = fitness_fn
        self._seed = seed
        self._n_workers = n_workers
        self._max_evaluation = max_fn_evaluation
        self._result = None

    def run(self,
            *args,
            **kwargs) -> dict:

        if self._result is not None:
            raise ValueError("Runners só podem ser executado uma vez.")

        start = time.perf_counter()
        logger.info('Optimization process started.')

        individual, fitness = self._run(*args, **kwargs)

        end = time.perf_counter()
        duration = end - start
        logger.info('Optimization finished in %f seconds.',
                    duration)

        # Criando dicionário com resultados
        n_ind = individual.shape[0]
        self._result = {
            'duration': duration,
            'best_solutions': [
                {
                    'fitness': {
                        name: v.item()
                        for name, v in zip(self._fn.info()['objectives_names'],
                                           fitness[i])
                    },
                    'array': individual[i].tolist()
                }
                for i in range(n_ind)
            ],
        }

        return self._result

    def result(self,
               *args,
               **kwargs) -> dict | None:
        return self._result

    def config(self) -> dict:
        config = self._algorithm_params()
        config.update({
            'fitness': {
                'name': self._fn.__class__.__name__,
                **self._fn.info()
            }
        })

        return config

    @abstractmethod
    def _algorithm_params(self) -> dict:
        """Retorna os parâmetros do algoritmo
        utilizado por esse runner.

        Returns:
            dict: parâmetros dos algoritmos.
        """

    @abstractmethod
    def _run(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Executa o processo de otimização com
        o algoritmo do runner, retornando como
        resultado uma tupla da melhor solução (indivíduo
        e fitness).

        O indivíduo possui shape (n_ind, n_dims) e o fitness (n_ind,
        n_objectives). Para runners mono-objetivos n_ind = 1, para
        multi-objetivos n_ind >= 1 e a solução representa o Pareto
        front/set.

        Returns:
            tuple[np.ndarray, np.ndarray]: solution (indivíduo),
                fitness.
        """
