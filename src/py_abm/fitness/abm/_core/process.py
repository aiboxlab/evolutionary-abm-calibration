"""Essa classe apresenta uma pequena abstração
da classe Popen do módulo subprocess, o objetivo
é manter uma interface similar ao da classe Process
do módulo multiprocessing.
"""
from __future__ import annotations

import io
from subprocess import Popen, TimeoutExpired


class Process:
    """Abstração de uma instância Popen.
    """
    _DEFAULT_TIMEOUT = 1
    _DEFAULT_READ_SIZE = 1024

    def __init__(self,
                 process: Popen) -> None:
        """Construtor. O processo deve possuir
        stdout e stderr precisam ser retornados
        em bytes pelo processo.

        Args:
            process (Popen): processo.
        """
        self._process = process
        self._out = bytearray()
        self._err = bytearray()

    @property
    def process(self) -> Popen:
        """Retorna o objeto Popen.

        Returns:
            Popen: representação baixo
                nível do processo.
        """
        return self._process

    def output(self) -> bytes:
        """Retorna a saída desse processo.

        Returns:
            str: saída (stdout) do processo.
        """
        # Realizar última leitura antes de retornar
        self._read_pipes()
        return bytes(self._out)

    def error(self) -> bytes:
        """Retorna o erro desse processo.

        Returns:
            str: erro (stderr) do processo.
        """
        # Realizar última leitura antes de retornar
        self._read_pipes()
        return bytes(self._err)

    def join(self,
             **kwargs) -> None:
        """Aguarda até a finalização do processo
        ou até o tempo de timeout.

        Args:
            timeout (float | None, optional): Defaults to None.
        """
        while self.is_alive():
            try:
                self._process.wait(timeout=self._DEFAULT_TIMEOUT,
                                   **kwargs)
            except TimeoutExpired:
                self._read_pipes()

    def is_alive(self) -> bool:
        """Retorna se o processo está vivo.

        Returns:
            bool: resultado.
        """
        self._read_pipes()
        return self._process.poll() is None

    def _read_pipes(self):
        streams = [self._process.stdout,
                   self._process.stderr]
        containers = [self._out,
                      self._err]
        for s, c in zip(streams, containers):
            self._read_if_data(s, c)

    @classmethod
    def _read_if_data(cls,
                      stream: io.BytesIO,
                      container: bytearray):
        while True:
            data = stream.read(cls._DEFAULT_READ_SIZE)
            if data:
                # Se possui dados, salvamos
                container.extend(data)
            else:
                # Do contrário, finalizamos o laço
                break
