"""Esse módulo contém classes e funções utilitárias
para utilizar em conjunto com a biblioteca pymoo.
"""
from __future__ import annotations

from pymoo.core.problem import Problem

from py_abm.entities import Fitness


class FitnessProblem(Problem):
    """Representa uma função de fitness do py_abm
    como um Problem do pymoo.
    """

    def __init__(self,
                 fn: Fitness,
                 n_workers: int,
                 **kwargs):
        """Construtor.

        Args:
            fn (Fitness): função de fitness.
            n_workers (int): quantidade de workers para execução
                paralela.
        """
        self._fn = fn
        self._n_workers = n_workers
        super().__init__(n_var=self._fn.info()['dims'],
                         n_obj=self._fn.info()['n_objectives'],
                         n_ieq_constr=0,
                         xl=self._fn.info()['bounds'][0],
                         xu=self._fn.info()['bounds'][1],
                         elementwise=False,
                         **kwargs)

    def _evaluate(self,
                  x,
                  out,
                  *args,
                  **kwargs):
        """Método de avaliação de indivíduos.

        Args:
            x: NumPy array em formato de matriz.
            out: dicionário para escrever a saída.
        """
        # Armazenando o resultado da avaliação
        out["F"] = self._fn.evaluate(x,
                                     n_parallel=self._n_workers)
