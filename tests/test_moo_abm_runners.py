"""Testes do ABM
com diferentes runners
multi-objetivos.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Type

import pytest

from py_abm.entities import Runner
from py_abm.fitness.abm.entities import Market, Objective, OptimizationType
from py_abm.runners.abm import GenericABMRunnerMO
from py_abm.runners.moo.moea import AGEMOEA2Runner, MOEADRunner
from py_abm.runners.moo.nsga import NSGA2Runner, UNSGA3Runner
from py_abm.runners.moo.spea import SPEA2Runner

_RUNNER_KWARGS = {
    "add_adj_rand_score": False
}

_GLOBAL = [NSGA2Runner, SPEA2Runner]
_SEGMENTS = [AGEMOEA2Runner]
_AGENT = [MOEADRunner, UNSGA3Runner]


@pytest.mark.parametrize("market", [Market.AUTOMAKERS])
@pytest.mark.parametrize("runner", _GLOBAL)
def test_global(market: Market,
                runner: Type[Runner]):
    _run_runner(OptimizationType.GLOBAL,
                market,
                runner)


@pytest.mark.parametrize("market", [Market.DAIRIES, Market.FASTFOOD])
@pytest.mark.parametrize("runner", _SEGMENTS)
def test_segments(market: Market,
                  runner: Type[Runner]):
    _run_runner(OptimizationType.SEGMENTS,
                market,
                runner)


@pytest.mark.parametrize("market", [Market.LUXURY])
@pytest.mark.parametrize("runner", _AGENT)
def test_agent(market: Market,
               runner: Type[Runner]):
    _run_runner(OptimizationType.AGENT,
                market,
                runner)


def _run_runner(optimization_type: OptimizationType,
                market: Market,
                runner: Type[Runner]):
    # Some cases are unsupported
    if optimization_type == OptimizationType.SEGMENTS:
        if market not in {Market.DAIRIES, Market.FASTFOOD}:
            return

    with tempfile.TemporaryDirectory() as tmp:
        runner = GenericABMRunnerMO(runner,
                                    market,
                                    optimization_type,
                                    time_steps=1,
                                    mc_runs=1,
                                    objectives=[Objective.RMSE, Objective.MAE],
                                    discretize_search_space=True,
                                    n_workers=4,
                                    population_size=6,
                                    seed=42,
                                    max_fn_evaluation=12,
                                    save_directory=Path(tmp).joinpath('run'),
                                    n_groups=None,
                                    milestone_interval=1,
                                    log_frequency=1,
                                    **_RUNNER_KWARGS)
        runner.run()
