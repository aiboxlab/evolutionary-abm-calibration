"""Esse módulo contém o runner para o PSO.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pyswarms as ps
from pyswarms.backend.topology import (Pyramid, Random, Ring, Star, Topology,
                                       VonNeumann)

from py_abm.entities import Fitness
from py_abm.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class PSOTopologyRunner(BaseRunner):
    """Runner do PSO para calibração de ABMs através
    do PySwarms.
    """

    def __init__(self,
                 fitness_fn: Fitness,
                 n_workers: int,
                 population_size: int,
                 topology: str | Topology = 'ring',
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 w: float = 0.7213475204444817,
                 c1: float = 1.1931471805599454,
                 c2: float = 1.1931471805599454,
                 k: int | None = None,
                 p: int | None = None,
                 r: int | None = None,
                 initial_position: None | np.ndarray = None,
                 **kwargs) -> None:
        # Pré-condições
        assert fitness_fn.info()['n_objectives'] == 1

        super().__init__(fitness_fn=fitness_fn,
                         seed=seed,
                         n_workers=n_workers,
                         max_fn_evaluation=max_fn_evaluation,
                         **kwargs)

        if isinstance(topology, str):
            available = ['random', 'ring', 'vonneuman', 'pyramid', 'star']
            topologies = [Random(), Ring(), VonNeumann(), Pyramid(), Star()]
            topology = next(c
                            for t, c in zip(available, topologies)
                            if topology == t)

        requires_k = any(map(lambda t: isinstance(topology, t),
                             [Ring, VonNeumann, Random]))

        if k is None and requires_k:
            # Caso não seja passado o valor de K,
            #   selecionamos 1/5 da população para ele.
            k = max(1, population_size // 5)

        requires_p = any(map(lambda t: isinstance(topology, t),
                             [Ring, VonNeumann]))

        if p is None and requires_p:
            # Caso não seja passado o valo de P,
            #   selecionamos a normal L2 (euclidiana).
            p = 2

        requires_r = isinstance(topology, VonNeumann)
        if r is None and requires_r:
            # Caso não seja passado o range para
            #   a arquitetura, utilizamos o valor 1.
            r = 1

        dims = self._fn.info()['dims']
        l, u = self._fn.info()['bounds']
        self._optimizer = ps.single.GeneralOptimizerPSO(
            n_particles=population_size,
            dimensions=dims,
            options=dict(c1=c1, c2=c2, w=w, k=k, p=p, r=r),
            topology=topology,
            bounds=(l * np.ones((dims,), dtype=np.float32),
                    u * np.ones((dims,), dtype=np.float32)),
            init_pos=initial_position)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        n_iterations = self._max_evaluation / self._optimizer.n_particles
        n_iterations = math.ceil(n_iterations)
        logger.info(f'Setting number of iterations to {n_iterations}...')
        logger.info('Setting global random seed with np.random.seed...')
        np.random.seed(self._seed)

        logger.info('Starting optimization...')
        fitness, individual = self._optimizer.optimize(
            lambda x: self._fn.evaluate(x, n_parallel=self._n_workers),
            iters=n_iterations,
            verbose=False)
        individual = np.broadcast_to(individual,
                                     (1, self._fn.info()['dims']))
        fitness = np.broadcast_to(fitness,
                                  (1, 1))

        return individual, fitness

    def _algorithm_params(self) -> dict:
        params = dict()
        params.update(self._optimizer.options)
        params.update({
            'population_size': self._optimizer.n_particles,
            'topology': self._optimizer.top.__class__.__name__,
            'seed': self._seed
        })

        config = {
            'algorithm': {
                'name': self._optimizer.__class__.__name__,
                'provider': 'pyswarms',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': params
            }
        }

        return config
