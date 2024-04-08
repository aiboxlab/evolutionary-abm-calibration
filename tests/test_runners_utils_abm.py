"""Testes unitários para os códigos utilitários
para runners de ABMs.
"""
from __future__ import annotations

import numpy as np
import pytest

from py_abm.fitness.abm.entities import Market
from py_abm.runners.abm import utils as abm_utils


@pytest.mark.parametrize("market,n_groups,groups",
                         [(Market.AUTOMAKERS,
                          5,
                          [1651, 1651, 1651, 1650, 1650]),
                          (Market.LUXURY,
                          5,
                          [638, 638, 638, 638, 638]),
                          (Market.DAIRIES,
                          10,
                          [2685, 2685, 2647, 2646, 2530,
                           2529, 1894, 1894, 3217, 3217]),
                          (Market.FASTFOOD,
                          8,
                          [4200, 4200, 5000, 5000,
                           6800, 6800, 4000, 4000])])
def test_create_groups(market: Market,
                       n_groups: int,
                       groups: list[int]):
    """Executa testes para a criação de grupos.

    Args:
        market (Market): mercado.
        n_groups (int): quantidade de grupos.
        groups (list[int]): valor esperado para os grupos
    """
    created_groups = abm_utils.create_groups(n_groups,
                                             market)
    expected = np.array(groups, dtype=np.int32)
    assert (created_groups == expected).all()
