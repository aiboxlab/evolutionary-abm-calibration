"""Funções utilitárias para a
análise dos resultados.
"""
from __future__ import annotations

import numpy as np

from py_abm.fitness.abm import Market

_CACHE = {
    m: np.cumsum(m.agents_per_segment)
    for m in {Market.DAIRIES, Market.FASTFOOD}
}


def get_agent_segment(agent_num: int, market: Market) -> int:
    less_than_seg = [agent_num < size for size in _CACHE[market]]
    return less_than_seg.index(True)
