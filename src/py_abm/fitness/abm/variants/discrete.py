"""Esse módulo define um ABM com discretização
do espaço de busca.
"""
from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from py_abm.entities import Fitness
from py_abm.fitness.abm import ABMFitness
from py_abm.fitness.abm.entities import (ABMJarOutput, Market, Objective,
                                         OptimizationType)


class DiscreteABMFitness(Fitness):
    """Essa classe adiciona a funcionalidade de
    cacheamento e discretização da função de fitness
    base.
    """

    def __init__(self,
                 market: Market,
                 optimization_type: OptimizationType,
                 time_steps: int,
                 mc_runs: int,
                 r_values_from_individual: Callable[[np.ndarray],
                                                    tuple[np.ndarray,
                                                          np.ndarray]],
                 sql_file_path: Path | None = None,
                 discrete_strategy: None | Callable[[np.ndarray],
                                                    np.ndarray] = None,
                 objectives: list[Objective] | None = None,
                 input_dims: int | None = None,
                 sql_row_limit: int = int(1e3)) -> None:
        """Construtor.

        Args:
            market (Market): mercado.
            optimization_type (OptimizationType): tipo de otimização.
            r_values_from_individual: função que converte um indivíduo (vetor)
                para seus valores de R1 e R2.
            discrete_strategy (None | Callable[[np.ndarray],
                                        np.ndarray]): estratégia de
                discretização.
            cache (bool): se uma estratégia de cache deve ser utilizada para
                diminuir o tempo de inferência.
            objectives (list[Objective, optional): lista de erros
                que devem ser calculados. Defaults to None (MAE, R2 e RMSE).
            sql_row_limit (int, optional): quantidade máxima de entradas que
                podem ser armazenadas no banco.
        """
        if discrete_strategy is None:
            discrete_strategy = nearest_hundredth

        self._fn = ABMFitness(
            market=market,
            optimization_type=optimization_type,
            time_steps=time_steps,
            mc_runs=mc_runs,
            r_values_from_individual=r_values_from_individual,
            objectives=objectives,
            input_dims=input_dims)

        self._db = _InMemoryDatabase(size_limit=sql_row_limit)
        self._map = discrete_strategy
        self._last_output = None

    @property
    def market(self) -> Market:
        """Retorna o mercado.

        Returns:
            Market: mercado.
        """
        return self._fn.market

    @property
    def time_steps(self) -> int:
        """Retorna a quantidade de time steps
        da simulação.

        Returns:
            int: time steps.
        """
        return self._fn.time_steps

    @property
    def mc_runs(self) -> int:
        """Retorna a quantidade de mcruns da
        simulação.

        Returns:
            int: mcruns.
        """
        return self._fn.mc_runs

    @property
    def optimization_type(self) -> OptimizationType:
        """Retorna o tipo de otimização.

        Returns:
            OptimizationType: tipo de otimização.
        """
        return self._fn.optimization_type

    @property
    def objectives(self) -> list[Objective]:
        """Retorna os objetivos.

        Returns:
            list[Objective]: objetivos.
        """
        return self._fn.objectives

    @property
    def discrete_mapper(self) -> Callable[[np.ndarray],
                                          np.ndarray]:
        """Retorna o mapeamento discreto utilizado.

        Returns:
            Callable[[np.ndarray], np.ndarray]: mapeador.
        """
        return self._map

    @property
    def r_values_from_individual(self) -> Callable[[np.ndarray],
                                                   tuple[np.ndarray,
                                                         np.ndarray]]:
        """Retorna a função para obter r-values.

        Returns:
            Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]: função.
        """
        return self._fn.r_values_from_individual

    @property
    def abm_output(self) -> list[ABMJarOutput] | None:
        """Retorna a saída do ABM para a última avaliação
        de fitness realizada (None caso nenhuma tenha sido
        realizada). Esse valor é atualizado a cada chamada à
        "evaluate" ou "evaluate_single".

        Returns:
            list[ABMJarOutput] | None: última saída produzida
                pelo ABM.
        """
        return self._last_output

    def evaluate(self,
                 individuals: np.ndarray,
                 n_parallel: int | None = None,
                 **kwargs) -> np.ndarray:
        """Principal método de avaliação. Permite avaliar
        um conjunto de indivíduos (representados como Numpy Arrays)

        Args:
            individuals (np.ndarray): array com os indivíduos.
            n_parallel (int, optional): quantidade de workers.
                Defaults to None (1 worker).

        Returns:
            np.ndarray: resultado da avaliação de fitness.
        """
        non_cached_idx = []
        results: dict[int, np.ndarray] = dict()

        # Aplicação do mapeamento (estratégia de discretização)
        individuals = self._map(individuals)

        # Preliminar: filtro dos indivíduos já na cache (banco)
        for i in range(individuals.shape[0]):
            ind = individuals[i]
            res = self._db.get(ind)

            if res is None:
                non_cached_idx.append(i)
                continue

            results[i] = res.value

        # Avaliação dos indivíduos fora da cache
        non_cached_individuals = individuals[non_cached_idx]
        evaluation = self._fn.evaluate(individuals=non_cached_individuals,
                                       n_parallel=n_parallel)

        # Adicionando resultados para esses indivíduos
        for eval_idx, ind_idx in enumerate(non_cached_idx):
            ind = individuals[ind_idx]
            res = evaluation[eval_idx]

            # Guardando resultado
            results[ind_idx] = res

            # Salvando no banco
            assert self._db.save(ind,
                                 res,
                                 self._fn.abm_output[eval_idx])

        # Atualizar saídas produzidas pelo ABM
        self._update_last_output(individuals)

        return np.array([results[i]
                         for i in sorted(results.keys())],
                        dtype=np.float32)

    def evaluate_single(self,
                        individual: np.ndarray,
                        **kwargs) -> np.ndarray:
        """Realiza a avaliação de um único indivíduo.

        Args:
            individual (np.ndarray): indivíduo.

        Returns:
            np.ndarray: valor da função de fitness.
        """
        # Aplicação do mapeamento
        individual = self._map(individual)

        # Tentando obter valor do banco
        res = self._db.get(individual)

        # Caso não esteja no banco, avaliamos e salvamos
        if res is None:
            res = self._fn.evaluate_single(individual)
            assert self._db.save(individual,
                                 res,
                                 self._fn.abm_output[0])
        else:
            res = res.value

        # Atualizar saídas produzidas pelo ABM
        self._update_last_output(np.expand_dims(individual,
                                                axis=0))

        return res

    def info(self) -> dict:
        return self._fn.info()

    def _update_last_output(self, individuals: np.ndarray):
        # Atualizamos a saída usando o cache mais recente dos
        #   indivíduos.
        self._last_output = [self._db.get(individuals[i]).output
                             for i in range(individuals.shape[0])]

        # Realizamos a limpeza do banco.
        self._db.clear()


@dataclass
class _Entry:
    key: np.ndarray
    value: np.ndarray
    output: ABMJarOutput


class _InMemoryDatabase:
    """Implementação alternativa do Banco de Dados
    que mantém os dados na memória principal.
    """

    def __init__(self,
                 size_limit: int,
                 **kwargs) -> None:
        del kwargs
        self._limit = size_limit
        self._data: dict[bytes, _Entry] = dict()
        self._remove_amount: int = max(self._limit//3, 1)

    def save(self,
             key: np.ndarray,
             value: np.ndarray,
             output: ABMJarOutput) -> bool:
        """Salva um novo par chave-valor no
            banco de dados.

        Args:
            key (np.ndarray): chave.
            value (np.ndarray): valor.

        Returns:
            bool: se a operação foi um sucesso.
        """
        key_bytes = np_array_to_bytes(key)

        if key_bytes not in self._data:
            self._data[key_bytes] = _Entry(key=key,
                                           value=value,
                                           output=output)

        return True

    def clear(self,
              n_rows: int | None = None) -> None:
        """Realiza uma limpeza nas entradas do
        Banco de Dados caso a quantidade de entradas
        seja maior ou superior ao limite, removendo a
        quantidade de linhas recebidas como argumento
        para o método.

        Args:
            n_rows (int | None, optional): quantidade de linhas
                para remover. Defaults to 1/3 do limite de linhas
                do banco.
        """
        if n_rows is None:
            n_rows = self._remove_amount

        if len(self._data) >= self._limit:
            keys_list = list(self._data.keys())
            for i in range(n_rows):
                del self._data[keys_list[i]]

    def get(self,
            key: np.ndarray) -> _Entry | None:
        """Retorna o valor associado com essa chave.
        Caso a chave não exista, retorna None.

        Args:
            key (np.ndarray): chave.

        Returns:
            np.ndarray | None: valor.
        """
        return self._data.get(np_array_to_bytes(key),
                              None)


class _SQLDatabase:
    """Essa é uma classe utilitária que representa um banco
    de dados SQL para armazenamento de NumPy Arrays.
    """

    def __init__(self,
                 sql_file_path: Path | None = None,
                 row_limit: int = int(1e3)) -> None:
        """Construtor

        Args:
            sql_file_path (Path | None, optional): Arquivo do BD.
                Defaults to None.
            row_limit (int, optional): limite máximo de linhas
                que podem ser armazenadas no banco.
        """
        self._fpath = sql_file_path

        if self._fpath is None:
            self._fpath = Path('fitness_cache.db')

        self._n_records = 0
        self._row_limit = row_limit
        self._row_remove_amount = max(row_limit // 3, 1)

        self._table_name = 'data'
        self._key_attr = 'key'
        self._value_attr = 'value'
        self._output_attr = 'output'
        self._conn = sqlite3.connect(self._fpath)
        self._create_table()

    def save(self,
             key: np.ndarray,
             value: np.ndarray,
             output: ABMJarOutput) -> bool:
        """Salva um novo par chave-valor no
            banco de dados.

        Args:
            key (np.ndarray): chave.
            value (np.ndarray): valor.

        Returns:
            bool: se a operação foi um sucesso.
        """
        if self.get(key) is not None:
            # Caso já esteja no banco, podemos
            #   retorna que foi salvo com sucesso.
            return True

        key_bytes = np_array_to_bytes(key)
        value_bytes = np_array_to_bytes(value)
        output_bytes = output_to_bytes(output)

        with self._conn as conn:
            try:
                cursor = conn.cursor()
                sql = f"INSERT INTO {self._table_name} VALUES (?,?,?)"
                cursor.execute(sql,
                               (key_bytes, value_bytes, output_bytes))
                self._n_records += 1
                return True
            except Exception:  # pylint: disable=bare-except
                return False

    def clear(self,
              n_rows: int | None = None) -> None:
        """Realiza uma limpeza nas entradas do
        Banco de Dados caso a quantidade de entradas
        seja maior ou superior ao limite, removendo a
        quantidade de linhas recebidas como argumento
        para o método.

        Args:
            n_rows (int | None, optional): quantidade de linhas
                para remover. Defaults to 1/3 do limite de linhas
                do banco.
        """
        remove_amount = n_rows

        if remove_amount is None:
            remove_amount = self._row_remove_amount

        with self._conn as conn:
            cursor = conn.cursor()
            if self._n_records >= self._row_limit:
                sql = (f"DELETE FROM {self._table_name} "
                       f"WHERE {self._key_attr} IN ("
                       f"SELECT {self._key_attr} FROM {self._table_name} "
                       f"LIMIT {remove_amount})")
                cursor.execute(sql)

    def get(self,
            key: np.ndarray) -> _Entry | None:
        """Retorna o valor associado com essa chave.
        Caso a chave não exista, retorna None.

        Args:
            key (np.ndarray): chave.

        Returns:
            np.ndarray | None: valor.
        """
        key_bytes = np_array_to_bytes(key)

        with self._conn as conn:
            try:
                cursor = conn.cursor()
                sql = (f"SELECT * FROM {self._table_name} "
                       f"WHERE {self._key_attr}=?")
                cursor.execute(sql,
                               (key_bytes,))
                result = cursor.fetchone()

                if result is not None:
                    value = np_array_from_bytes(result[1])
                    output = output_from_bytes(result[2])
                    result = _Entry(key=key,
                                    value=value,
                                    output=output)

                return result
            except Exception:  # pylint: disable=bare-except
                return None

    def _create_table(self):
        with self._conn as conn:
            cursor = conn.cursor()
            cursor.execute(f"CREATE TABLE {self._table_name}("
                           f"{self._key_attr} BLOB PRIMARY KEY,"
                           f"{self._value_attr} BLOB NOT NULL,"
                           f"{self._output_attr} BLOB NOT NULL)")


def nearest_hundredth(values: np.ndarray) -> np.ndarray:
    """Estratégia de discretização para os centésimos
    mais próximos.

    Args:
        individuals (np.ndarray): indivíduos.

    Returns:
        np.ndarray: versão discretizada.
    """
    return values.round(decimals=2)


def np_array_to_bytes(arr: np.ndarray) -> bytes:
    """Transforma uma NumPy Array em um
    objeto bytes, através da geração do
    pickle.

    Args:
        arr (np.ndarray): array.

    Returns:
        bytes: representação em bytes.
    """
    return arr.dumps()


def output_to_bytes(output: ABMJarOutput) -> bytes:
    """Transforma a saída da simulador em um objeto
    bytes através do pickle.

    Args:
        output (ABMJarOutput): saída do ABM.

    Returns:
        bytes: representação em bytes.
    """
    return pickle.dumps(output)


def np_array_from_bytes(value: bytes) -> np.ndarray:
    """Retorna uma NumPy Array de um objeto bytes
    representando o pickle.

    Args:
        value (bytes): bytes.

    Returns:
        np.ndarray: array.
    """
    return pickle.loads(value)


def output_from_bytes(value: bytes) -> ABMJarOutput:
    """Retorna uma saída do simulador de um objeto
    bytes representando o pickle.

    Args:
        value (bytes): bytes.

    Returns:
        ABMJarOutput: saída do simulador.
    """
    return pickle.loads(value)
