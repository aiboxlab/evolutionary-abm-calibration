"""Testes unitários para o DiscreteABMFitness.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from py_abm.fitness.abm import DiscreteABMFitness
from py_abm.fitness.abm.entities import Market, OptimizationType


def test_abm_output_is_not_none():
    """Executa um teste para garantir que o DiscreteABMFitness
    armazena os indivíduos corretamente até guardar o resultado
    no abm_output.

    Existiam dois erros:
    1. O banco de dados estava sendo limpo após
        salvar todo indivíduo, gerando um miss quando tentávamos
        atualizar o last output e um dos indivíduos calculados naquela
        iteração tinha sido removido;
    2. Quando o conjunto de indivíduos possui 2 indivíduos iguais,
        ocorria um erro no salvamento no Banco de Dados;
    """
    fn = DiscreteABMFitness(market=Market.AUTOMAKERS,
                            optimization_type=OptimizationType.GLOBAL,
                            time_steps=1,
                            mc_runs=1,
                            r_values_from_individual=lambda v: np.split(v, 2),
                            sql_row_limit=1)

    fn.evaluate(np.array([[0.0, 0.0],
                          [0.25, 0.25],
                          [0.0, 0.0],
                          [0.5, 0.5]],
                         dtype=np.float32),
                workers=3)
    assert all(f is not None for f in fn.abm_output)
