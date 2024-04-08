"""Esse módulo contém classes bases
para runners que utilizam o ABM.
"""
from __future__ import annotations

import functools
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np

from py_abm.entities import Runner
from py_abm.fitness.abm import (ABMFitness, DiscreteABMFitness,
                                DiscreteClusteringABMFitness, Heuristic,
                                Market, Objective, OptimizationType)

from . import utils

logger = logging.getLogger(__name__)


class BaseABMRunner(Runner):
    """Classe base para runners
    do ABM.
    """

    def __init__(self,
                 market: Market,
                 optimization_type: OptimizationType,
                 time_steps: int,
                 mc_runs: int,
                 n_groups: int | None,
                 objectives: list[Objective],
                 discretize_search_space: bool,
                 n_workers: int,
                 save_directory: Path,
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 milestone_interval: int = 100,
                 log_frequency: int = 100,
                 ensure_save_directory_is_empty: bool = True) -> None:
        # Variáveis de caminhos
        self._save_dir = save_directory

        # Variáveis relativos ao processo de otimização
        self._n_workers = n_workers
        self._seed = seed
        self._result = None
        self._max_evaluation = max_fn_evaluation
        self._n_evaluations = 0

        # Variáveis de controle de logs/histórico
        self._log_freq = log_frequency
        self._last_milestone = 0
        self._milestone_interval = milestone_interval

        # Garantindo que o diretório de saída não existe inicialmente
        if self._save_dir.exists() and ensure_save_directory_is_empty:
            raise ValueError('Save directory already exists.')

        # Criando o diretório de saída
        self._save_dir.mkdir(parents=True, exist_ok=True)

        # Instanciando função de fitness
        r_values_from_individuals = utils.r_values_split
        self._groups = None

        # A quantidade de dimensões depende do mercado e do tipo de otimização:
        #   Se otimização global, são d = 2
        #   Se segmentos, d = 2 * <quantidade de segmentos>
        #   Se agentes, d = 2 * <quantidade de agentes>
        # A multiplicação por 2 é que temos r1 e r2 para cada "elemento".
        self._n_dims = 2 * (1
                            if optimization_type == OptimizationType.GLOBAL
                            else market.segments
                            if optimization_type == OptimizationType.SEGMENTS
                            else market.agents)

        if n_groups is not None:
            assert utils.valid_n_groups(market,
                                        optimization_type,
                                        n_groups)
            self._groups = utils.create_groups(n_groups, market)

            # Caso seja utilizada uma estratégia de grupos,
            #   d = 2 * <quantidade de grupos>
            self._n_dims = 2 * self._groups.size

            # Atualizando estratégia para obtenção dos r-values:
            #   agora é necessário um broadcast antes do split.
            r_values_from_individuals = functools.partial(
                utils.r_values_split_w_broadcast,
                groups=self._groups)

        if discretize_search_space:
            self._fn = DiscreteABMFitness(
                market=market,
                optimization_type=optimization_type,
                time_steps=time_steps,
                mc_runs=mc_runs,
                r_values_from_individual=r_values_from_individuals,
                discrete_strategy=None,
                objectives=objectives,
                input_dims=self._n_dims)
        else:
            self._fn = ABMFitness(
                market=market,
                optimization_type=optimization_type,
                time_steps=time_steps,
                mc_runs=mc_runs,
                r_values_from_individual=r_values_from_individuals,
                objectives=objectives,
                input_dims=self._n_dims)

    @property
    def abm_fn(self) -> ABMFitness | DiscreteABMFitness:
        """Retorna a função objetiva utilizada por
        esse runner.
        """
        return self._fn

    def run(self,
            *args,
            **kwargs) -> dict:
        """Esse método executa o runner seguindo as
        configurações definidas.
        """
        if self._result is not None:
            raise ValueError("ABMRunners só podem ser executado uma vez.")

        # Salvando as configurações
        self._log_dict_to_file('config.json',
                               self.config(),
                               overwrite=True)

        start = time.perf_counter()
        logger.info('Optimization process started.')

        individual, fitness = self._run(*args, **kwargs)

        end = time.perf_counter()
        duration = end - start
        logger.info('Optimization finished in %f seconds.',
                    duration)

        # Obtendo quantidade de indivíduos na solução
        n_ind = individual.shape[0]

        # Garantindo shape das arrays
        assert individual.shape == (n_ind, self._fn.info()['dims'])
        assert fitness.shape == (n_ind, self._fn.info()['n_objectives'])

        # Função auxiliar para processamento de cada indivíduo
        def _data_from_individual(ind: np.ndarray,
                                  fit: np.ndarray) -> dict:
            if isinstance(self._fn, DiscreteABMFitness):
                # Se for discreto, precisamos obter a representação
                #   real do indivíduo.
                ind = self._fn.discrete_mapper(ind)

            # Convertendo para listas (json-serializable)
            r1, r2 = self._fn.r_values_from_individual(ind)
            r1 = r1.tolist()
            r2 = r2.tolist()

            # Obtendo informações da saida desse indivíduo
            _ = self._fn.evaluate_single(ind)
            abm_output = self._fn.abm_output[0]

            return {
                'fitness': {
                    name: v.item()
                    for name, v in zip(self._fn.info()['objectives_names'],
                                       fit)
                },
                'r1': r1,
                'r2': r2,
                'array': ind.tolist(),
                'decisions': list(map(lambda d: {k.value: v for
                                                 k, v in d.items()},
                                  abm_output.decisions)),
                'decisions_count': [
                    list(map(lambda d: {k_.value: v_
                                        for k_, v_ in d.items()},
                         v))
                    for v in abm_output.decisions_count
                ],
                'probabilities': [{
                    k.value: k.probability(a[0],
                                           a[1])
                    for k in Heuristic
                } for a in zip(r1, r2)]
            }

        self._result = {
            'duration': duration,
            'total_fitness_evaluations': self._n_evaluations,
            'best_solutions': [
                _data_from_individual(individual[i],
                                      fitness[i])
                for i in range(n_ind)
            ]
        }

        # Salva o resultado em um JSON
        self._log_dict_to_file('result.json',
                               self._result,
                               overwrite=True)

        return self._result

    def fitness_callback(self,
                         individuals: np.ndarray,
                         fitness: ABMFitness,
                         objectives: np.ndarray) -> None:
        """Método de callback que realiza o log
        dos indivíduos no histórico de execução do
        runner.

        O callback pode ser utilizado manualmente (i.e., quando
        temos controle do laço evolutivo) ou através de um
        Wrapper (e.g., ABMFitnessWithCallback).

        Todas as classes filhas devem utilizar o callback de alguma
        forma para armazenamento dos resultados.

        Args:
            individuals (np.ndarray): matriz dos indivíduos avaliados
                na função de fitness, com shape (n_individuals, n_dims).
            fitness (ABMFitness): função de fitness
                utilizada.
            objectives (np.ndarray): valor produzido como saída
                pela função de fitness, com shape (n_individuals,
                n_objectives).
        """
        # Pré-condições
        assert objectives.shape[-1] == self._fn.info()['n_objectives']
        assert individuals.shape[-1] == self._fn.info()['dims']
        assert objectives.shape[0] == individuals.shape[0]

        if self._n_evaluations > self._max_evaluation:
            # Caso já tenhamos passado do critério de avaliação,
            #   não salvamos os resultados.
            logger.info('Termination criteria has been reached:'
                        ' %d out of %d fitness evaluations.'
                        ' New results won\'t be saved to the history.',
                        self._n_evaluations,
                        self._max_evaluation)
            self._n_evaluations += individuals.shape[0]
            return

        batch_size = individuals.shape[0]
        total_eval = self._n_evaluations + batch_size
        output = fitness.abm_output
        fname = 'history.json'
        assert output is not None

        # Calculamos se já passamos do intervalo mínimo
        elapsed = (total_eval - self._last_milestone)
        should_save = elapsed >= self._milestone_interval

        # Caso seja tempo de salvar ou chegamos no limite das avaliações
        #   de fitness
        if should_save or (total_eval >= self._max_evaluation):
            # Atualizando o último milestone
            old_milestone = self._last_milestone
            self._last_milestone = total_eval

            if isinstance(fitness, DiscreteABMFitness):
                # Se for discreto, precisamos obter a representação
                #   real dos indivíduos.
                individuals = fitness.discrete_mapper(individuals)

            # Obtain data to log
            data = self._get_callback_data(individuals,
                                           fitness,
                                           objectives)

            if old_milestone == 0:
                self._log_dict_to_file(fname,
                                       data=data,
                                       overwrite=False,
                                       leading='[\n',
                                       trailing=',\n',
                                       indent=None)
            elif total_eval >= self._max_evaluation:
                self._log_dict_to_file(fname,
                                       data=data,
                                       overwrite=False,
                                       trailing='\n]\n',
                                       indent=None)
            else:
                self._log_dict_to_file(fname,
                                       data=data,
                                       overwrite=False,
                                       trailing=',\n',
                                       indent=None)

        # Atualizando o total de avaliações
        self._n_evaluations = total_eval

        # Realizando log do andamento
        if self._n_evaluations % self._log_freq == 0:
            logger.info('Number of fitness evaluations: %d/%d (%f%%)',
                        self._n_evaluations,
                        self._max_evaluation,
                        100 * (self._n_evaluations/self._max_evaluation))

    def result(self,
               *args,
               **kwargs) -> dict | None:
        """Esse método retorna os resultados obtidos
        após execução do Runner.
        """
        return self._result

    def config(self,
               *arg,
               **kwargs) -> dict:
        """Esse método retorna a configuração do
        runner.
        """
        config = self._algorithm_params()
        groups = self._groups.tolist() if self._groups is not None else False
        objectives = list(map(lambda v: v.value,
                              self._fn.objectives))
        config.update({
            'fitness': {
                'name': self._fn.__class__.__name__,
                'n_dims': self._n_dims,
                'market': self._fn.market.value,
                'optimization_type': self._fn.optimization_type.value,
                'time_steps': self._fn.time_steps,
                'mc_runs': self._fn.mc_runs,
                'groups': groups,
                'objectives': objectives,
            }
        })

        return config

    def _log_dict_to_file(self,
                          fname: str,
                          data: dict,
                          overwrite: bool = False,
                          trailing: str = None,
                          leading: str = None,
                          indent: int | None = 2) -> None:
        mode = 'w' if overwrite else 'a'
        fpath = self._save_dir.joinpath(fname)
        with fpath.open(mode, encoding='utf-8') as f:
            if leading is not None:
                f.write(leading)

            json.dump(data,
                      f,
                      ensure_ascii=False,
                      indent=indent,
                      default=str)

            if trailing is not None:
                f.write(trailing)

    @abstractmethod
    def _get_callback_data(self,
                           individuals: np.ndarray,
                           fitness: ABMFitness,
                           objectives: np.ndarray) -> dict:
        """Obtém os dados que devem ser salvos no histórico
        de execução do runner.

         Args:
            individuals (np.ndarray): matriz dos indivíduos avaliados
                na função de fitness, com shape (n_individuals, n_dims).
            fitness (ABMFitness): função de fitness
                utilizada.
            objectives (np.ndarray): valor produzido como saída
                pela função de fitness, com shape (n_individuals,
                n_objectives).

        Returns:
            dict: dados a serem salvos.
        """

    @abstractmethod
    def _algorithm_params(self) -> dict:
        """Retorna os parâmetros do algoritmo
        utilizado por esse runner.

        Returns:
            dict: parâmetros dos algoritmos.
        """

    @abstractmethod
    def _run(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Executa o processo de otimização com
        o algoritmo do runner, retornando como
        resultado uma tupla da melhor solução (indivíduo
        e fitness).

        O indivíduo possui shape (n_ind, n_dims) e o fitness (n_ind,
        n_objectives). Para runners mono-objetivos n_ind = 1, para
        multi-objetivos n_ind >= 1 e a solução representa o Pareto
        front/set.

        Returns:
            tuple[np.ndarray, np.ndarray]: solution (indivíduo),
                fitness.
        """


class BaseABMRunnerSO(BaseABMRunner):
    """Classe base para runners mono-objetivos
    do ABM.
    """

    def __init__(self,
                 market: Market,
                 optimization_type: OptimizationType,
                 time_steps: int,
                 mc_runs: int,
                 n_groups: int | None,
                 objective: Objective,
                 discretize_search_space: bool,
                 n_workers: int,
                 save_directory: Path,
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 milestone_interval: int = 100,
                 log_frequency: int = 100,
                 ensure_save_directory_is_empty: bool = True) -> None:
        super().__init__(
            market=market,
            optimization_type=optimization_type,
            time_steps=time_steps,
            mc_runs=mc_runs,
            n_groups=n_groups,
            objectives=[objective],
            discretize_search_space=discretize_search_space,
            n_workers=n_workers,
            save_directory=save_directory,
            seed=seed,
            max_fn_evaluation=max_fn_evaluation,
            milestone_interval=milestone_interval,
            log_frequency=log_frequency,
            ensure_save_directory_is_empty=ensure_save_directory_is_empty)

        self._history_manager = utils.HistoryManagerSO()

    def _get_callback_data(self,
                           individuals: np.ndarray,
                           fitness: ABMFitness,
                           objectives: np.ndarray) -> dict:
        # Update history
        self._history_manager.update(individuals=individuals,
                                     fitness=objectives)

        # Obtiain current state
        entry = self._history_manager.get_current()

        # Return best and average results
        objective_name = self._fn.info()['objectives_names'][0]
        return {
            'milestone': self._last_milestone,
            'best_individual': entry.best_solution.tolist(),
            'best_fitness': {
                objective_name: entry.best_fitness.item()
            },
            'average_individual': entry.average_solution.tolist(),
            'average_fitness': entry.average_fitness.tolist()
        }


class BaseABMRunnerMO(BaseABMRunner):
    """Classe base para runners multi-objetivos
    do ABM.
    """

    def __init__(self,
                 market: Market,
                 optimization_type: OptimizationType,
                 time_steps: int,
                 mc_runs: int,
                 n_groups: int | None,
                 objectives: list[Objective],
                 discretize_search_space: bool,
                 add_adj_rand_score: bool,
                 n_workers: int,
                 save_directory: Path,
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 milestone_interval: int = 100,
                 log_frequency: int = 100,
                 ensure_save_directory_is_empty: bool = True) -> None:
        super().__init__(
            market=market,
            optimization_type=optimization_type,
            time_steps=time_steps,
            mc_runs=mc_runs,
            n_groups=n_groups,
            objectives=objectives,
            discretize_search_space=discretize_search_space,
            n_workers=n_workers,
            save_directory=save_directory,
            seed=seed,
            max_fn_evaluation=max_fn_evaluation,
            milestone_interval=milestone_interval,
            log_frequency=log_frequency,
            ensure_save_directory_is_empty=ensure_save_directory_is_empty)

        if add_adj_rand_score:
            # Atualizar função de otimização do runner
            #   para versão que adiciona novo objetivo
            self._fn = DiscreteClusteringABMFitness(
                market=self._fn.market,
                optimization_type=self._fn.optimization_type,
                time_steps=self._fn.time_steps,
                mc_runs=self._fn.mc_runs,
                r_values_from_individual=self._fn.r_values_from_individual,
                discrete_strategy=self._fn.discrete_mapper,
                objectives=self._fn.objectives,
                input_dims=self._fn.info()['dims'])

        self._history_manager = utils.HistoryManagerMO()

    def _get_callback_data(self,
                           individuals: np.ndarray,
                           fitness: ABMFitness,
                           objectives: np.ndarray) -> None:
        # Update history
        self._history_manager.update(individuals=individuals,
                                     fitness=objectives)

        # Obtiain current state
        entry = self._history_manager.get_current()

        # Return best and average results
        return {
            'milestone': self._last_milestone,
            'best_pareto_set': entry.pareto_set.tolist(),
            'best_pareto_front': [
                {
                    name: value.item()
                    for name, value in zip(self._fn.info()['objectives_names'],
                                           entry.pareto_front[i])
                }
                for i in range(entry.pareto_front.shape[0])
            ]
        }
