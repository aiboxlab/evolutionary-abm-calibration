"""Testes unitários do ProcessHandler.
"""
from __future__ import annotations

import numpy as np
import pytest

from py_abm.fitness.abm._core.process_handler import ProcessHandler
from py_abm.fitness.abm.entities import (Heuristic, Market, Objective,
                                         OptimizationType)


@pytest.mark.parametrize("optimization_type", OptimizationType)
@pytest.mark.parametrize("market,r1,r2",
                         [(m, 0.5, 0.5) for m in Market])
def test_handler(optimization_type: OptimizationType,
                 market: Market,
                 r1: float,
                 r2: float):
    """Executa testes para o Handler.

    Args:
        market (Market): mercado.
        r1 (float): valor de r1.
        r2 (float): valor de r2.
    """
    # Obtendo saída do simulador
    result = _run(optimization_type,
                  market,
                  r1,
                  r2)

    if result == 'SKIP':
        return

    # Mercado tem que ser o mesmo
    assert result.market == market

    # Precisamos ter um resultado por objetivo
    assert len(result.objectives) == len(Objective)

    # Precisamos ter um conjunto de decisão por agente
    assert len(result.decisions) == market.agents

    # Garantindo que cada conjunto de decisão possui as
    #   4 heurísticas.
    assert all(map(_has_all_heuristics, result.decisions))

    # Garantindo que a contabilização de heurísticas
    #   possui a quantidade certa de mc runs e time steps.
    assert len(result.decisions_count) == 1
    assert len(result.decisions_count[0]) == 1

    # Garantindo que todo time step possui as 4 heurísticas.
    assert all(map(_has_all_heuristics, result.decisions_count[0]))

    # Garantindo que o valor dos objetivos foram corretos
    target = _correct_objectives(market)
    assert all([np.equal(result.objectives[o],
                         target[o])
                for o in Objective])


@pytest.mark.parametrize("optimization_type", OptimizationType)
@pytest.mark.parametrize("market", Market)
@pytest.mark.parametrize("r1", [0, 1])
@pytest.mark.parametrize("r2", [0, 1])
def test_heuristic_distribution(optimization_type: OptimizationType,
                                market: Market,
                                r1: float,
                                r2: float):
    # Calculando a heurística máxima
    max_heuristic = _max_heuristic(r1, r2)

    # Obtendo saída do simulador
    # Colocamos mais timesteps para evitar que algum
    #   agente não tome nenhuma escolha.
    result = _run(optimization_type,
                  market,
                  float(r1),
                  float(r2),
                  ts=1,
                  mcruns=4)

    if result == 'SKIP':
        return

    # A maior decisão de todos os agentes deveria ser
    #   a máxima (todas as demais tem probabilidade 0)
    flatten_decision_count = []
    for ts in result.decisions_count:
        flatten_decision_count += ts

    for d in result.decisions + flatten_decision_count:
        assert max(d, key=d.get) == max_heuristic


def _has_all_heuristics(d: dict) -> bool:
    return set(d.keys()) == set(Heuristic)


def _correct_objectives(market: Market) -> dict[Objective, float]:
    if market == Market.AUTOMAKERS:
        return {
            Objective.MAE: float('2.1888560349006294'),
            Objective.RMSE: float('2.5098968807990047'),
            Objective.R2: float('0.9264143956225697')
        }

    if market == Market.DAIRIES:
        return {
            Objective.MAE: float('5.939788775824854'),
            Objective.RMSE: float('6.078123318843572'),
            Objective.R2: float('0.9268989407032198')
        }

    if market == Market.FASTFOOD:
        return {
            Objective.MAE: float('7.854285714285715'),
            Objective.RMSE: float('9.22974007140582'),
            Objective.R2: float('0.6031086946910227')
        }

    return {
        Objective.MAE: float('4.617941483803553'),
        Objective.RMSE: float('5.486925813529282'),
        Objective.R2: float('0.8142181152730625')
    }


def _broadcast_r_values(opt: OptimizationType,
                        market: Market,
                        r: float) -> np.ndarray:

    if opt == OptimizationType.GLOBAL:
        return np.array([r], dtype=np.float32)

    target_size = market.agents

    if opt != OptimizationType.AGENT:
        target_size = market.segments

    return np.array([r] * target_size,
                    dtype=np.float32)


def _max_heuristic(r1: int, r2: int) -> Heuristic:
    # (r1 ,r2)
    # (0, 0) = SAT
    # (0, 1) = MAJ
    # (1, 0) = EBA
    # (1, 1) = UMAX
    if r1 == 0 and r2 == 0:
        return Heuristic.SAT

    if r1 == 0 and r2 == 1:
        return Heuristic.MAJ

    if r1 == 1 and r2 == 0:
        return Heuristic.EBA

    if r1 == 1 and r2 == 1:
        return Heuristic.UMAX

    return None


def _run(optimization_type: OptimizationType,
         market: Market,
         r1: float,
         r2: float,
         ts: int = 1,
         mcruns: int = 1):
    # Apenas alguns mercados suportam otimização de segmentos
    if optimization_type == OptimizationType.SEGMENTS:
        if market not in [Market.FASTFOOD, Market.DAIRIES]:
            return 'SKIP'

    # Obtendo r-values como NumPy arrays
    r1 = _broadcast_r_values(optimization_type,
                             market,
                             r1)

    r2 = _broadcast_r_values(optimization_type,
                             market,
                             r2)

    # Instanciando handler
    handler = ProcessHandler(market,
                             optimization_type,
                             r1=r1,
                             r2=r2,
                             time_steps=ts,
                             mc_runs=mcruns,
                             show_heuristics=True)
    # Executando o handler e aguardando a finalização
    handler.run(join=True)

    return handler.results()
