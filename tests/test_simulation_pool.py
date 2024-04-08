"""Testes unitários do ProcessHandler.
"""
from __future__ import annotations

import numpy as np

from py_abm.fitness.abm._core.process_handler import ProcessHandler
from py_abm.fitness.abm._core.simulation_pool import SimulationPool
from py_abm.fitness.abm.entities import Market, OptimizationType


def test_run_handler_wait_for_slot():
    """Executa um teste para garantir que o Simulation Pool
    aguarda corretamente até ter um worker disponível.
    """
    pool = SimulationPool(n_workers=1)

    for i in range(3):
        # Instanciação do ProcessHandler para execução da simulação
        abm_jar = ProcessHandler(market=Market.AUTOMAKERS,
                                 run_type=OptimizationType.GLOBAL,
                                 time_steps=1,
                                 mc_runs=1,
                                 r1=np.array([0.5], dtype=np.float32),
                                 r2=np.array([0.5], dtype=np.float32))
        # Executar handler
        pool.run_handler(abm_jar, i)

    pool.wait_all()
