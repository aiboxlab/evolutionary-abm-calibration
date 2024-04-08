"""Esse é um módulo privado que implementa
o gerenciamento do Jar utilizado para execução
das simulações.
"""
from pathlib import Path

from importlib_resources import files

PATH: Path = files('py_abm.fitness.abm._core').joinpath("abm.jar")
