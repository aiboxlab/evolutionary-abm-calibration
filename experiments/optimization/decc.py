"""Esse script executa o DECC
para otimização do ABM em sua versão
discreta com parâmetros pré-definidos.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from py_abm.fitness.abm.entities import Market, Objective, OptimizationType
from py_abm.runners.abm import GenericABMRunnerSO
from py_abm.runners.soo.cc import DECCGRunner, DECCRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="py_abm",
                                     description="ABM calibration with DECC.")
    parser.add_argument('-m',
                        '--market',
                        dest='market',
                        help='Market to optimize',
                        required=True,
                        type=str)
    parser.add_argument('-w',
                        '--workers',
                        dest='workers',
                        help='Number of parallel workers',
                        required=True,
                        type=int)
    parser.add_argument('-o',
                        '--optimization-type',
                        dest='opt',
                        help='Optimization type.',
                        required=False,
                        default='agent',
                        type=str)
    parser.add_argument('-f',
                        '--fitness_fn',
                        dest='objective',
                        help=('Which objective to use (i.e., '
                              'RMSE, MAE or R2)'),
                        required=False,
                        default='RMSE',
                        type=str)
    parser.add_argument('-g',
                        '--use-groups',
                        dest='groups',
                        help='Whether to use groups or not.',
                        action='store_true')
    parser.add_argument('-p',
                        '--population',
                        dest='pop_size',
                        help='Population size',
                        required=False,
                        default=30,
                        type=int)
    parser.add_argument('--variant',
                        dest='variant',
                        help='Which variant to use',
                        required=False,
                        default='O',
                        type=str)
    parser.add_argument('--F',
                        dest='F',
                        help='F value to use.',
                        required=False,
                        default=0.5,
                        type=float)
    parser.add_argument('--CR',
                        dest='CR',
                        help='CR value to use.',
                        required=False,
                        default=0.7,
                        type=float)
    parser.add_argument('--n-subproblems',
                        dest='n_subproblems',
                        help='Number of subproblems for DECC-G.',
                        required=False,
                        default=100,
                        type=int)
    parser.add_argument('--max-fn',
                        dest='max_fn',
                        help='Number of fitness evaluations',
                        required=False,
                        default=5000,
                        type=int)
    parser.add_argument('--n-milestones',
                        dest='n_milestones',
                        help='Number of milestones to write in history',
                        required=False,
                        default=100,
                        type=int)
    parser.add_argument('--time-steps',
                        dest='ts',
                        help='Number of time steps for simulation',
                        required=False,
                        default=10,
                        type=int)
    parser.add_argument('--mcruns',
                        dest='mcruns',
                        help='Number of mcruns for simulation',
                        required=False,
                        default=10,
                        type=int)
    parser.add_argument('--save_directory',
                        dest='save_dir',
                        help='Directory to save output',
                        required=False,
                        type=str)
    parser.add_argument('-s',
                        '--seed',
                        dest='seed',
                        help='Random seed for DE',
                        required=False,
                        type=int)
    args = parser.parse_args()

    pop_size = args.pop_size
    n_groups = 100 if args.groups else None
    market = Market.from_str(args.market.lower())
    opt = OptimizationType.from_str(args.opt)
    ts = args.ts
    objective = Objective.from_str(args.objective.upper())
    mcruns = args.mcruns
    workers = args.workers
    max_fn = args.max_fn
    seed = args.seed
    save_dir = args.save_dir
    F = args.F
    CR = args.CR
    variant = args.variant
    n_subproblems = args.n_subproblems
    milestone_interval = max(max_fn//args.n_milestones, 1)
    log_frequency = max(max_fn//100, 1)

    if seed is None:
        seed = random.randint(0, 999999)

    if save_dir is None:
        directories = []

        for p in Path('.').iterdir():
            if p.is_dir():
                try:
                    v = int(p.name)
                    directories.append(v)
                except ValueError:
                    continue

        directories = sorted(directories)
        base = 0 if len(directories) <= 0 else directories[-1]
        save_dir = str(base + 1)

    if variant.upper() == 'G':
        max_subproblems = 2

        if opt == OptimizationType.AGENT:
            max_subproblems = market.agents
        elif opt == OptimizationType.SEGMENTS:
            max_subproblems = market.segments

        n_subproblems = min(n_subproblems, max_subproblems)
        evaluations_de = max(min(max_fn // 12, 1000), 1)
        evaluations_sansde = max(min(max_fn // (4 * n_subproblems), 1000), 1)
        runner = GenericABMRunnerSO(DECCGRunner,
                                    market,
                                    opt,
                                    time_steps=ts,
                                    mc_runs=mcruns,
                                    objective=objective,
                                    discretize_search_space=True,
                                    n_workers=workers,
                                    population_size=pop_size,
                                    n_subproblems=n_subproblems,
                                    de_evaluations=evaluations_de,
                                    sansde_evaluations=evaluations_sansde,
                                    F=F,
                                    CR=CR,
                                    seed=seed,
                                    max_fn_evaluation=max_fn,
                                    save_directory=Path(save_dir),
                                    n_groups=n_groups,
                                    milestone_interval=milestone_interval,
                                    log_frequency=log_frequency)
    else:
        runner = GenericABMRunnerSO(DECCRunner,
                                    market,
                                    opt,
                                    time_steps=ts,
                                    mc_runs=mcruns,
                                    objective=objective,
                                    discretize_search_space=True,
                                    n_workers=workers,
                                    population_size=pop_size,
                                    variant=variant,
                                    F=F,
                                    CR=CR,
                                    seed=seed,
                                    max_fn_evaluation=max_fn,
                                    save_directory=Path(save_dir),
                                    n_groups=n_groups,
                                    milestone_interval=milestone_interval,
                                    log_frequency=log_frequency)

    runner.run()
