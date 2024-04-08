"""Esse é um módulo privado que implementa
uma abstração alto-nível para execução do ABM.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from py_abm.fitness.abm import entities

from . import jar_handler
from .process import Process

logger = logging.getLogger(__name__)


class ProcessHandler:
    """Essa classe representa um gerenciador
    para o processo de execução do ABM utilizando
    o arquivo jar.
    """
    JAVA_EXECUTABLE = shutil.which("java")

    def __init__(self,
                 market: entities.Market,
                 run_type: entities.OptimizationType,
                 r1: np.ndarray,
                 r2: np.ndarray,
                 time_steps: int = 1,
                 mc_runs: int = 100,
                 show_heuristics: bool = True,
                 tmp_directory: Path | None = None) -> None:
        self._market: entities.Market = market
        self._run_type: entities.OptimizationType = run_type
        self._ts = time_steps
        self._mc_runs = mc_runs
        self._heuristics = show_heuristics
        self._r1: np.ndarray = r1
        self._r2: np.ndarray = r2

        self._tmp: Path = tmp_directory
        self._tmp_directory = None
        self._jar_path: Path = jar_handler.PATH
        self._proc: Process = None
        self._result: entities.ABMJarOutput = None
        self._start = None
        self._end = None

    def run(self, join: bool = False) -> None:
        """Inicia a execução de uma simulação utilizando
        os valores de R1 e R2 passados.


        Args:
            join (bool, optional): Se deve aguardar
                até o fim do processo. Defaults to False.
        """
        p = subprocess.Popen(args=self._build_cmd(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        self._proc = Process(process=p)

        logger.debug('Started ABM simulation process (PID: %s)',
                     p.pid)
        self._start = time.perf_counter()

        if join:
            self._proc.join()

    def results(self) -> entities.ABMJarOutput:
        """Retorna os resultados da execução da simulação.

        Returns:
            entities.ABMJarOutput: resultado da simulação.
        """
        assert self._proc is not None
        assert not self._proc.is_alive()

        self._end = time.perf_counter()
        output = self._proc.output().decode("utf-8")
        err = self._proc.error().decode("utf-8")
        msg = output + err

        market = re.search(r'market = (?P<market>\w+)\n',
                           output)
        objectives = re.search(r'MAE = (?P<MAE>[0-9]+\.[0-9]*)\n'
                               r'RMSE = (?P<RMSE>[0-9]+\.[0-9]*)\n'
                               r'R2 = (?P<R2>[0-9]+\.[0-9]*)\n',
                               output)
        heuristics = None
        heuristics_count = None

        assert objectives is not None, msg
        assert market is not None, msg
        assert market.group('market') == self._market.value, msg
        assert len(objectives.groups()) == len(entities.Objective), msg

        if self._heuristics:
            heuristics = re.search(
                r'heuristics = (?P<heuristics>.+)\n',
                output)
            heuristics_count = re.search(
                r'heuristicCount = \{\n(?P<hcount>[0-9,\{\}\s]+)\}\}\n\}',
                output)

            assert heuristics.group('heuristics') is not None, output
            assert heuristics_count.group('hcount') is not None, output

            def _parse_single(value: str,
                              cvt=float) -> dict[entities.Heuristic,
                                                 float | int]:
                # Exemplo de entrada: 0.0,0.0,0.0,1.0
                # Obtendo valores individuais
                value = re.sub(r"[\{\}]", "", value)
                splits = value.split(',')

                # Garantindo que temos uma porcentagem para cada
                #   heurística
                assert len(splits) == len(entities.Heuristic)

                # Conversão para float
                splits = list(map(cvt, splits))

                # Ordem: {UMAX, EBA, MAJ,SAT}
                return {
                    entities.Heuristic.UMAX: splits[0],
                    entities.Heuristic.EBA: splits[1],
                    entities.Heuristic.MAJ: splits[2],
                    entities.Heuristic.SAT: splits[3]
                }

            # Pré-processamento para parsing
            heuristics = heuristics.group('heuristics').strip().split('},{')
            heuristics_count = heuristics_count.group('hcount').strip()
            heuristics_count = re.sub(r'\s+', '',
                                      heuristics_count).split('},{')
            size = len(heuristics)
            assert size == self._market.agents, size

            size = len(heuristics_count)
            assert size == self._ts * self._mc_runs, size

            # Fazendo o parse individual
            heuristics = list(map(_parse_single, heuristics))
            heuristics_count = [
                [_parse_single(heuristics_count[ts + mc * self._ts],
                               int)
                 for ts in range(self._ts)]
                for mc in range(self._mc_runs)
            ]

        logger.debug('ABM simulation (PID: %s, Return code: %s) '
                     'finished in %f seconds.',
                     self._proc.process.pid,
                     str(self._proc.process.returncode),
                     self._end - self._start)

        return entities.ABMJarOutput(
            market=self._market,
            objectives={
                k: float(objectives.group(k.value))
                for k in entities.Objective
            },
            decisions=heuristics,
            decisions_count=heuristics_count)

    def process(self) -> Process:
        """Retorna a representação do processo
        executando a simulação.

        Returns:
            Process: simulador Java.
        """
        assert self._proc is not None
        return self._proc

    def _build_cmd(self) -> list[str]:
        """Constrói o comando e prepara o
        ambiente para realizar a execução do
        simulador.

        Returns:
            list[str]: comandos.
        """
        assert self._r1.size == self._r2.size

        cmd = [self.JAVA_EXECUTABLE,
               "-jar",
               str(self._jar_path),
               "-market",
               self._market.value,
               "-timesteps",
               str(self._ts),
               "-mcruns",
               str(self._mc_runs)]

        if self._heuristics:
            cmd.append("-showHeuristics")

        if self._run_type == entities.OptimizationType.GLOBAL:
            assert self._r1.size == 1

            cmd = cmd + ['-r1Global',
                         str(self._r1.item()),
                         '-r2Global',
                         str(self._r2.item())]
        elif self._run_type == entities.OptimizationType.SEGMENTS:
            assert self._market in [entities.Market.DAIRIES,
                                    entities.Market.FASTFOOD]
            assert self._r1.size == self._market.segments

            cmd = cmd + ['-r1Segment',
                         np.array2string(self._r1,
                                         separator=',')[1:-1],
                         '-r2Segment',
                         np.array2string(self._r2,
                                         separator=',')[1:-1]]
        elif self._run_type == entities.OptimizationType.AGENT:
            assert self._r1.size == self._market.agents

            def _write_to_file(fpath: Path,
                               v: np.ndarray) -> None:
                fpath.write_text(np.array2string(v,
                                                 separator='\n',
                                                 threshold=np.inf)[1:-1],
                                 encoding='utf-8')

            self._tmp_directory = tempfile.TemporaryDirectory(dir=self._tmp)
            path = Path(self._tmp_directory.name)
            r1_file = path.joinpath('r1.txt')
            _write_to_file(r1_file, self._r1)

            r2_file = path.joinpath('r2.txt')
            _write_to_file(r2_file, self._r2)

            cmd = cmd + ['-r1Agent',
                         str(r1_file),
                         '-r2Agent',
                         str(r2_file)]
        else:
            raise ValueError(f"Modo de otimização inválido: {self._run_type}")

        return cmd
