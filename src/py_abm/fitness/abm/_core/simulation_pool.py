"""Esse é um módulo privado que implementa
uma abstração alto-nível para execução de 
múltiplas simulações simultâneas ordenadas
do ABM.
"""
from __future__ import annotations

from py_abm.fitness.abm import entities

from .process_handler import ProcessHandler


class SimulationPool:
    """Essa classe representa um pool
    de processos de ABMs.
    """

    def __init__(self, n_workers: int) -> None:
        assert n_workers >= 1

        self._n_workers = n_workers
        self._running: dict[int, ProcessHandler] = dict()
        self._finished: dict[int, ProcessHandler] = dict()

    def run_handler(self,
                    abm_handler: ProcessHandler,
                    idx: int) -> None:
        """Inicia a execução da simulação representada
        pelo ProcessHandler. Caso não existam worker disponíveis,
        aguarda até que seja possível executar.

        Args:
            abm_handler (ProcessHandler): simulação do ABM.
            idx (int): índice dessa simulação, usado para
                construir a saída na ordem correta.
        """
        # Caso a quantidade de simulações em execução
        #   seja maior ou superior ao limite, devemos
        #   aguardar.
        if len(self._running) >= self._n_workers:
            free_slots = 0
            remove_keys = []

            while free_slots <= 0:
                for k, p in self._running.items():
                    proc = p.process()
                    if not proc.is_alive():
                        self._finished[k] = p
                        remove_keys.append(k)
                        free_slots += 1

            for k in remove_keys:
                del self._running[k]

        # iniciamos a execução dessa simulação
        abm_handler.run(join=False)

        # Guardamos esse handler na lista dos que
        #   estão em execução.
        self._running[idx] = abm_handler

    def wait_all(self) -> None:
        """Aguarda a finalização de todas as simulações
        presentes nesse pool.
        """
        for i in list(self._running.keys()):
            p = self._running[i]
            proc = p.process()

            if proc.is_alive():
                proc.join()

            self._finished[i] = p
            del self._running[i]

        assert len(self._running) == 0

    def results(self) -> list[entities.ABMJarOutput]:
        """Retorna os resultados ordenados de cada
        uma das simulações executadas.

        Returns:
            list[entities.ABMJarOutput]: resultados.
        """
        return [self._finished[i].results()
                for i in sorted(self._finished.keys())]
