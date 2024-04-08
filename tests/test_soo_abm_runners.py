"""Testes do ABM
com diferentes runners
mono-objetivos.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Type

import pytest

from py_abm.entities import Runner
from py_abm.fitness.abm.entities import Market, Objective, OptimizationType
from py_abm.runners.abm import GenericABMRunnerSO
from py_abm.runners.soo.cc import DECCGRunner, DECCRunner
from py_abm.runners.soo.cmaes import BIPOPRunner, IPOPRunner
from py_abm.runners.soo.de import SHADEILSRunner, SHADERunner
from py_abm.runners.soo.pso import PSORunner, PSOTopologyRunner

_RUNNER_KWARGS = {
    'topology': 'ring',
    'evaluations_gs': 1,
    'evaluations_de': 1,
    'evaluations_ls': 1,
    'de_evaluations': 1,
    'sansde_evaluations': 1,
    'n_subproblems': 2,
    'variant': 'H'
}

_GLOBAL = [SHADEILSRunner, SHADERunner,
           DECCGRunner, DECCRunner]
_SEGMENTS = [BIPOPRunner, IPOPRunner]
_AGENT = [PSORunner, PSOTopologyRunner]


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
        runner = GenericABMRunnerSO(runner,
                                    market,
                                    optimization_type,
                                    time_steps=1,
                                    mc_runs=1,
                                    objective=Objective.RMSE,
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
