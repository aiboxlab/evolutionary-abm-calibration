from __future__ import annotations

import logging
import math

import numpy as np
from cmaes import CMA

from py_abm.entities import Fitness
from py_abm.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class BIPOPRunner(BaseRunner):
    """BIPOP CMA-ES runner.
    """

    def __init__(self,
                 fitness_fn: Fitness,
                 n_workers: int,
                 population_size: int,
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 sigma: float = 1 / 5,
                 **kwargs) -> None:
        # Pré-condições
        assert fitness_fn.info()['n_objectives'] == 1

        super().__init__(fitness_fn=fitness_fn,
                         seed=seed,
                         n_workers=n_workers,
                         max_fn_evaluation=max_fn_evaluation,
                         **kwargs)

        self._population_size = population_size
        self._sigma = sigma

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        # Carregando informações da função de fitness
        dims = self._fn.info()['dims']
        l, u = self._fn.info()['bounds']
        l = np.array([l] * dims)
        u = np.array([u] * dims)
        bounds = np.concatenate([l, u], axis=-1)
        del l, u

        rng = np.random.default_rng(self._seed)
        mean = bounds[:, 0] + (rng.random(dims) *
                               (bounds[:, 1] - bounds[:, 0]))
        optimizer = CMA(population_size=self._population_size,
                        mean=mean,
                        sigma=self._sigma,
                        bounds=bounds,
                        seed=self._seed)

        evaluation_count = 0
        popsize0 = optimizer.population_size
        small_n_eval, large_n_eval = 0, 0
        inc_popsize = 2
        poptype = 'small'
        best_solution = None

        while evaluation_count < self._max_evaluation:  # max_evals
            # Evaluate population
            x = np.array([optimizer.ask()
                          for _ in range(optimizer.population_size)],
                         dtype=np.float32)
            value = self._fn.evaluate(x, n_parallel=self._n_workers)
            evaluation_count += optimizer.population_size  # increment FE

            min_value_index = np.argmin(value)
            solution = (x[min_value_index], value[min_value_index])

            if (best_solution is None) or solution[1] < best_solution[1]:
                best_solution = solution

            solutions_and_fitness = [(x[i], value[i])
                                     for i in range(optimizer.population_size)]
            optimizer.tell(solutions_and_fitness)

            # Apply BIPOP-CMA-ES strategy
            if optimizer.should_stop():
                seed_bipop = rng.integers(0, 99999)  # Update the seed
                n_eval = optimizer.population_size * optimizer.generation

                if poptype == 'small':
                    small_n_eval += n_eval
                else:
                    large_n_eval += n_eval

                if small_n_eval < large_n_eval:
                    poptype = 'small'
                    popsize_multiplier = inc_popsize ** evaluation_count
                    popsize = math.floor(
                        popsize0 * popsize_multiplier ** (rng.uniform() ** 2))
                else:
                    poptype = 'large'
                    evaluation_count += 1
                    popsize = popsize0 * (inc_popsize ** evaluation_count)
                mean = lower_bounds + \
                    (rng.random(self._n_dims) * (upper_bounds - lower_bounds))
                optimizer = CMA(mean=mean,
                                sigma=self._sigma,
                                bounds=bounds,
                                seed=seed_bipop,
                                population_size=popsize)

        solution = np.broadcast_to(best_solution[0], (1, dims))
        fitness = np.broadcast_to(best_solution[1], (1, 1))

        return solution, fitness

    def _algorithm_params(self) -> dict:
        """
        Return the configuration of the runner.

        Returns:
            dict: Configuration dictionary.
        """
        config = {
            'algorithm': {
                'name': 'BIPOP',
                'provider': 'CMA-ES',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': {
                    'population_size': self._population_size,
                    'seed': self._seed,
                    'sigma': self._sigma,
                }
            }
        }

        return config
