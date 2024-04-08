"""Esse módulo define um ABM com discretização
do espaço de busca e saída multi-objetiva com
informação da heterogeneidade dos agentes de um
segmento.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn import cluster, metrics

from py_abm.fitness.abm.entities import Market, Objective, OptimizationType

from .discrete import DiscreteABMFitness


class DiscreteClusteringABMFitness(DiscreteABMFitness):
    def __init__(self,
                 market: Market,
                 optimization_type: OptimizationType,
                 time_steps: int,
                 mc_runs: int,
                 r_values_from_individual: Callable[[np.ndarray],
                                                    tuple[np.ndarray,
                                                          np.ndarray]],
                 sql_file_path: Path | None = None,
                 discrete_strategy: Callable[[
                     np.ndarray], np.ndarray] | None = None,
                 objectives: list[Objective] | None = None,
                 input_dims: int | None = None,
                 sql_row_limit: int = int(1e3),
                 minimize: bool = True) -> None:
        assert market in {Market.DAIRIES, Market.FASTFOOD}
        assert optimization_type == OptimizationType.AGENT
        super().__init__(market=market,
                         optimization_type=optimization_type,
                         time_steps=time_steps,
                         mc_runs=mc_runs,
                         r_values_from_individual=r_values_from_individual,
                         sql_file_path=sql_file_path,
                         discrete_strategy=discrete_strategy,
                         objectives=objectives,
                         input_dims=input_dims,
                         sql_row_limit=sql_row_limit)
        cs = np.concatenate([[0],
                             np.cumsum(market.agents_per_segment)])
        self._true_label = np.zeros((market.agents,),
                                    dtype=np.int32)
        self._mininimize = minimize
        for i in range(1, len(cs)):
            self._true_label[cs[i-1]:cs[i]] = i

    def evaluate(self,
                 individuals: np.ndarray,
                 n_parallel: int | None = None,
                 **kwargs) -> np.ndarray:
        base = super().evaluate(individuals,
                                n_parallel,
                                **kwargs)
        cluster = self._get_adj_rand_index(individuals, n_parallel)
        return np.concatenate([base, cluster], axis=-1)

    def evaluate_single(self,
                        individual: np.ndarray,
                        **kwargs) -> np.ndarray:
        base = super().evaluate_single(individual, **kwargs)
        cluster = self._get_adj_rand_index(np.expand_dims(individual, 0))[0]
        return np.concatenate([base, cluster], axis=-1)

    def info(self) -> dict:
        base = super().info()
        obj_name = ('neg_' if self._mininimize else '') + 'adj_rand_score'
        base['n_objectives'] += 1
        base['objectives_names'].append(obj_name)
        return base

    def _get_adj_rand_index(self,
                            individuals: np.ndarray,
                            n_parallel: int | None = None,) -> np.ndarray:
        if n_parallel is None:
            n_parallel = 1

        n_ind = individuals.shape[0]

        # Obtain features
        features = np.zeros((n_ind, self.market.agents, 2),
                            dtype=np.float32)

        for i in range(n_ind):
            r_values = self.r_values_from_individual(individuals[i])
            features[i] = np.stack(r_values, axis=-1)

        # Obtain scores
        # TODO: add parallel evaluation
        scores = []
        for i in range(n_ind):
            scores.append(_adj_rand_score(features[i],
                                          self._true_label))

        scores = np.expand_dims(np.array(scores), axis=-1)
        if self._mininimize:
            scores *= -1

        return scores


def _adj_rand_score(features: np.ndarray,
                    true_label: np.ndarray):
    n_segments = np.unique(true_label).shape[0]
    k_means = cluster.KMeans(n_clusters=n_segments,
                             n_init='auto',
                             random_state=42)

    # Ignore sklearn warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_labels = k_means.fit_predict(features)

    return metrics.adjusted_rand_score(labels_true=true_label,
                                       labels_pred=pred_labels)
