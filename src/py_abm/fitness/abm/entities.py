"""Esse módulo contém classes utilitárias para
implementação do ABM.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass
class ABMJarOutput:
    """Representa a saída de um ABM.
    """
    market: Market
    objectives: dict[Objective, float]
    decisions: list[dict[Heuristic, float]] | None
    decisions_count: list[list[dict[Heuristic, float]]] | None


class Objective(Enum):
    """Representa os objetivos para um ABM.
    """
    RMSE = 'RMSE'
    MAE = 'MAE'
    R2 = 'R2'

    @classmethod
    def from_str(cls, value: str) -> Objective:
        """Retorna o objetivo a partir de uma string.

        Args:
            value (str): valor.

        Returns:
            Objective: objetivo.
        """
        return next(filter(lambda o: o.value == value,
                           Objective))


class Market(Enum):
    """Representa os mercados do ABM.
    """
    AUTOMAKERS = ('automakers', [8253], None, 8253)
    LUXURY = ('luxury', [3190], None, 3190)
    DAIRIES = ('dairies', [5370, 5293, 5059, 3788, 6434],
               ['<35y', '35-44y', '45-54y', '55-64y', '>64y'], 25944)
    FASTFOOD = ('fastfood', [8400, 10000, 13600, 8000],
                ['heavy', 'medium', 'light', 'no consumers'],
                40000)

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, name: str,
                 segments: list[int],
                 segments_desc: list[str],
                 agents: int,
                 ):
        del name
        self._segments = segments
        self._segments_desc = segments_desc
        self._segments_size = len(segments)
        self._agents = agents

    def __str__(self):
        return self.value

    @property
    def segments(self) -> int:
        """Retorna a quantidade de segmentos
        para esse mercado.

        Returns:
            int: quantidade de segmentos.
        """
        return self._segments_size

    @property
    def agents(self) -> int:
        """Retorna a quantidade de agentes para
        esse mercado.

        Returns:
            int: quantidade de agentes.
        """
        return self._agents

    @property
    def agents_per_segment(self) -> list[int]:
        """Retorna a quantidade de agentes para
        cada segmento desse mercado.

        Returns:
            list[int]: quantidade de agentes por
                segmento.
        """
        return self._segments

    @property
    def segments_desc(self) -> list[str]:
        """Retorna uma descrição de cada segmento
        para esse mercado.

        Returns:
            list[str]: descrição de cada segmento.
        """
        return self._segments_desc

    @classmethod
    def from_str(cls, value: str) -> Market:
        """Retorna um mercado a partir de uma string.

        Args:
            value (str): valor.

        Returns:
            Market: mercado.
        """
        return next(filter(lambda o: o.value == value,
                           Market))


class OptimizationType(Enum):
    """Representa as formas de otimização
    do ABM.
    """
    GLOBAL = 'global'
    SEGMENTS = 'segments'
    AGENT = 'agent'

    @classmethod
    def from_str(cls, value: str) -> OptimizationType:
        """Retorna um tipo de otimização a partir de uma string.

        Args:
            value (str): valor

        Returns:
            OptimizationType: tipo de otimização.
        """
        return next(filter(lambda o: o.value == value,
                           OptimizationType))


class Heuristic(Enum):
    """Representação das heurísticas que podem
    ser selecionadas pelos agentes.
    """
    EBA = 'EBA'
    MAJ = 'MAJ'
    SAT = 'SAT'
    UMAX = 'UMAX'

    def probability(self,
                    r1: float,
                    r2: float) -> float:
        """Dados os valores de r1 e r2, retorna
        a probabilidade de escolha dessa heurística.

        Args:
            r1 (float): valor de r1.
            r2 (float): valor de r2.

        Returns:
            float: probabilidade de escolha.
        """
        if self == Heuristic.UMAX:
            return r1 * r2

        if self == Heuristic.MAJ:
            return (1 - r1) * r2

        if self == Heuristic.EBA:
            return r1 * (1 - r2)

        if self == Heuristic.SAT:
            return (1 - r1) * (1 - r2)

    @classmethod
    def from_str(cls, value: str) -> Heuristic:
        """Retorna uma heurística a partir de uma string.

        Args:
            value (str): valor

        Returns:
            Heuristic: heurística.
        """
        return next(filter(lambda o: o.value == value,
                           Heuristic))
