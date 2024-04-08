"""Esse módulo contém funções para
extração de informações de um dado
experimento do ABM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from py_abm.fitness.abm import Heuristic, Market
from py_abm.visualization.abm.parser import ExperimentParser

from . import utils


def best_fitness_per_milestone(experiment: ExperimentParser,
                               error: str = 'RMSE') -> list[float]:
    """Extrai apenas um dos objetivos do dicionário de fitness.

    Args:
        experiment (ExperimentParser): experimento.
        error (str, opcional): 'RMSE', 'R2' ou 'MAE'. Padrão
            é 'RMSE'.

    Returns:
        list[float]: valores para o erro escolhido.
    """
    return list(map(lambda i: np.array([f[error] for f in i.fitness]).min(),
                    experiment.best_individual_per_milestone))


def chosen_heuristics(experiment: ExperimentParser,
                      idx: int = 0) -> pd.DataFrame:
    """Extrai a quantidade total de heurísticas escolhidas pela
    melhor solução. Retorna uma lista com os nomes selecionadas
    durante a execução do ABM.

    Args:
        experiment (ExperimentParser): experimento.
        idx: indíce do indivíduo a ser analisado.

    Returns:
        pd.DataFrame: DataFrame com coluna 'Heuristic'.
    """
    chosen_heuristics = []
    for entry in experiment.best_individuals[idx].decisions_count:
        mc_run_count = {
            k: sum([v_ for k_, v_ in d.items()
                    if k_ == k.value])
            for k in Heuristic
            for d in entry
        }

        for k in mc_run_count:
            chosen_heuristics.extend([k.value] * mc_run_count[k])

    return pd.DataFrame({'Heuristic': chosen_heuristics})


def max_heuristic(experiment: ExperimentParser,
                  probability: bool = False,
                  idx: int = 0) -> pd.DataFrame:
    """Gera um conjunto de pontos considerando r1 e r2 e a heurística
    com maior valor. O intuito, é obter informações da heurística
    para cada um dos agentes e determinar qual heurística foi mais
    utilizada/possui maior probabilidade.

    Args:
        experiment (ExperimentParser): experimento.
        probability (bool, opcional): se devemos selecionar a heurística
            com base na probabilidade. Por padrão, selecionamos
            com base na frequência.
        idx: indíce do indivíduo a ser analisado.

    Returns:
        pd.DataFrame: conjunto de pontos (r1, r2, heurística, qtd).
    """
    ind = experiment.best_individuals[idx]
    if probability:
        dict_heuristic_value = ind.probabilities
    else:
        dict_heuristic_value = ind.decisions

    best_individual = experiment.best_individuals[idx]
    market = experiment.market
    r1_values = experiment.broadcast_r_value(best_individual.r1)
    r2_values = experiment.broadcast_r_value(best_individual.r2)
    has_segments = market in {Market.DAIRIES, Market.FASTFOOD}

    data = {
        'R1': [],
        'R2': [],
        'Heuristic': [],
    }

    if has_segments:
        data['Segment'] = []

    assert len(r1_values) == len(r2_values)
    assert len(r1_values) == len(dict_heuristic_value)
    for agent_num, (r1, r2, value) in enumerate(zip(r1_values,
                                                    r2_values,
                                                    dict_heuristic_value)):
        data['R1'].append(r1)
        data['R2'].append(r2)
        data['Heuristic'].append(max(value,
                                     key=value.get))

        if has_segments:
            # Adicionando o segmento que esse agente faz parte
            seg_number = utils.get_agent_segment(agent_num, market)
            data['Segment'].append(market.segments_desc[seg_number])

    return pd.DataFrame(data).value_counts().reset_index(name='Size')
