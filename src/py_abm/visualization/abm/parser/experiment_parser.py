"""Módulo com o parser de arquivos
de experimentação envolvendo o ABM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ijson

from py_abm.fitness.abm import Market, OptimizationType


@dataclass(frozen=True)
class Individual:
    """Essa classe representa um indivíduo
    do processo de otimização.
    """
    r1: list[float]
    r2: list[float]
    array: list[float]

    # Valores de fitness associados com esse indivíduo
    fitness: Optional[dict[str, float]] = None

    # Frequência de decisão para cada agente
    decisions: Optional[list[dict[str, float]]] = None

    # Contagem de decisões para cada agente
    decisions_count: Optional[list[dict[str, int]]] = None

    # Probabilidade de escolha das heurísticas para cada agente
    probabilities: Optional[list[dict[str, float]]] = None


@dataclass(frozen=True)
class HistoryEntry:
    milestone: int
    solutions: list[list[float]]
    fitness: list[dict[str, float]]


class ExperimentParser:
    """Essa classe realiza o parse de experimentos
    que produzem como resultado JSONs padronizados.
    """
    _HISTORY = 'history.json'
    _CONFIG = 'config.json'
    _RESULT = 'result.json'
    _FN_KEY = 'fitness'

    def __init__(self,
                 results_path: Path,
                 unnormalize_fitness: bool = True) -> None:
        """Construtor. Recebe o caminho com os
        3 arquivos de resultados gerados e
        expôe propriedades relativas aos resultados.

        Args:
            results_path: caminho para o diretório
                com os arquivos de resultado.
            unnormalize_fitness: se os valores de
                fitness devem ser desnormalizados.
        """
        self._dir = results_path
        self._should_unnormalize = unnormalize_fitness

        # Arquivos JSON potencialmente grandes,
        #   melhor guardar apenas a referência do caminho
        #   e usar o IJSON para parsing on-demand.
        self._history_path = self._dir.joinpath(self._HISTORY)
        self._results_path = self._dir.joinpath(self._RESULT)

        # Config costuma ser pequeno e pode ser carregado
        # diretamente
        self._config_path = self._dir.joinpath(self._CONFIG)
        config_text = self._config_path.read_text('utf-8')
        self._config = json.loads(config_text)

        # Dicionário chave-valor para cacheamento
        #   de propriedades custosas de serem lidas
        #   dos JSON
        self._cache = dict()

        # Indica se foi um problema multi-objetivo
        with self._history_path.open('r') as f:
            self._is_mo = any(map(lambda k: 'pareto' in k,
                                  next(ijson.items(f, 'item'))))

    @property
    def experiment_dir(self) -> Path:
        return self._dir

    @property
    def is_moo(self) -> bool:
        """Se esse experimento é
        multi-objetivo.

        Returns:
            bool: se é multi-objetivo.
        """
        return self._is_mo

    @property
    def best_individuals(self) -> list[Individual]:
        """Retorna os melhores indivíduos encontrados. Os
        r-values do indivíduo são iguais aos passados
        para o simulador.

        Returns:
            Individual: indivíduo.
        """
        def _individual_from_dict(d: dict) -> Individual:
            # Broadcast probabilidades para cada agente
            probabilities = d['probabilities']
            opt = self.optimization_type
            market = self.market

            if opt == OptimizationType.GLOBAL:
                probabilities = probabilities * market.agents
            elif len(probabilities) != market.agents:
                # Caso seja otimização a nível de segmentos
                #   ou agentes, podemos precisar fazer
                #   o broadcast.
                if opt == OptimizationType.SEGMENTS:
                    groups = market.agents_per_segment
                else:
                    groups = ([1] * self.market.agents
                              if not self.groups
                              else self.groups)
                    assert sum(groups) == market.agents

                assert len(probabilities) == len(
                    groups), f'{len(probabilities)} != {len(groups)}'

                probabilities = [repeat
                                 for i, g in enumerate(groups)
                                 for repeat in [probabilities[i]] * g]
            assert len(probabilities) == market.agents
            d['probabilities'] = probabilities
            d['fitness'] = self._maybe_unnormalize_fitness(d['fitness'])
            return Individual(**d)

        with open(self._results_path, 'r') as result_file:
            individuals = list(ijson.items(result_file,
                                           'best_solutions.item',
                                           use_float=True))
        return [_individual_from_dict(i) for i in individuals]

    @property
    def best_individual_per_milestone(self) -> list[HistoryEntry]:
        """Retorna o melhor valor de fitness por
        milestone.

        Returns:
            list[dict[str, float]]: melhor fitness por iteração.
        """
        key = 'best_ind_history'

        if key not in self._cache:
            individuals = []
            with self._history_path.open('r') as hist_file:
                for entry in ijson.items(hist_file,
                                         'item',
                                         use_float=True):
                    if self._is_mo:
                        f_key = 'best_pareto_front'
                        s_key = 'best_pareto_set'
                        def fn(x): return x
                    else:
                        f_key = 'best_fitness'
                        s_key = 'best_individual'
                        def fn(x): return [x]

                    f, s = fn(entry[f_key]), fn(entry[s_key])
                    # NOTE(13/03/2024): currently, the normalization occurs as
                    #   part of the post_evaluation callable in the
                    #   ABMFitnessWithCallback, thus the history
                    #   is not affected by normalization (post_evaluation
                    #   is only applied after the callback).
                    # f = list(map(self._maybe_unnormalize_fitness, f))
                    he = HistoryEntry(milestone=entry['milestone'],
                                      fitness=f,
                                      solutions=s)
                    individuals.append(he)
            self._cache[key] = individuals

        return self._cache[key]

    @property
    def time_steps(self) -> int:
        """Retorna a quantidade de time steps.

        Returns:
            int: time steps.
        """
        return self._config[self._FN_KEY]['time_steps']

    @property
    def mc_runs(self) -> int:
        """Retorna a quantidade de mc runs.

        Returns:
            int: mc runs.
        """
        return self._config[self._FN_KEY]['mc_runs']

    @property
    def market(self) -> Market:
        """Retorna o mercado.

        Returns:
            Market: mercado
        """
        return Market.from_str(self._config[self._FN_KEY]['market'])

    @property
    def optimization_type(self) -> OptimizationType:
        """Retorna o tipo de otimização realizado.

        Returns:
            OptimizationType: tipo de otimização.
        """
        return OptimizationType.from_str(
            self._config[self._FN_KEY]['optimization_type'])

    @property
    def groups(self) -> bool | list[int]:
        """Retorna os grupos (se aplicável) ou False.

        Returns:
            bool | list[int]: grupos ou False.
        """
        return self._config[self._FN_KEY]['groups']

    @property
    def algorithm(self) -> dict:
        """Retorna as configurações do algoritmo
        de otimização utilizado.

        Returns:
            dict: configurações do algoritmo.
        """
        return self._config['algorithm']

    def broadcast_r_value(self, r_value: list[float]) -> list[float]:
        """Faz o broadcast de uma lista de r_values (global, agent,
        segments ou grupo) para sua versão por agente.

        Args:
            r_value (list[float]): r_values originais.

        Returns:
            list[float]: r_value por agente.
        """
        n_repeats = None

        if self.optimization_type == OptimizationType.GLOBAL:
            n_repeats = [self.market.agents]
        elif self.optimization_type == OptimizationType.SEGMENTS:
            n_repeats = self.market.agents_per_segment
        else:
            n_repeats = [1] * len(r_value)

        assert len(n_repeats) == len(r_value)
        result = [r
                  for i, s in enumerate(n_repeats)
                  for r in [r_value[i]] * s]
        assert len(result) == self.market.agents
        return result

    def _maybe_unnormalize_fitness(self,
                                   fitness: dict[str, float]) -> dict[str,
                                                                      float]:
        if (not self._should_unnormalize) or \
                ("normalization" not in self._config):
            return fitness

        normalization = self._config["normalization"]
        assert normalization["kind"] == "MinMax"

        fitness = fitness.copy()
        for key, min, max in zip(fitness,
                                 normalization["minimum"],
                                 normalization["maximum"]):
            fitness[key] = fitness[key] * (max - min) + min

        return fitness
