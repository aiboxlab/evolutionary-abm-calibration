"""Esse script executa algoritmos
baseados no MOEA para otimização do
ABM.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from py_abm.fitness.abm.entities import Market, Objective, OptimizationType
from py_abm.runners.abm import GenericABMRunnerMO
from py_abm.runners.moo.moea import AGEMOEA2Runner, MOEADRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="py_abm",
                                     description="ABM calibration with MOEA.")
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
                        dest='objectives',
                        help=('Which objectives to use (i.e., '
                              'RMSE, MAE or R2)'),
                        metavar='N',
                        nargs='+',
                        default=['RMSE'],
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
    parser.add_argument('--no-adj-rand-score',
                        dest='adj_rand_score',
                        help='Disable extra clustering objective.',
                        action='store_false')
    parser.add_argument('--moead',
                        dest='moead',
                        help='Whether to use MOEA/D or not.',
                        action='store_true')
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
                        help='Random seed for NSGA2',
                        required=False,
                        type=int)
    args = parser.parse_args()

    pop_size = args.pop_size
    n_groups = 100 if args.groups else None
    market = Market.from_str(args.market.lower())
    opt = OptimizationType.from_str(args.opt)
    ts = args.ts
    objectives = [Objective.from_str(o.upper())
                  for o in args.objectives]
    mcruns = args.mcruns
    workers = args.workers
    max_fn = args.max_fn
    seed = args.seed
    save_dir = args.save_dir
    milestone_interval = max(max_fn//args.n_milestones, 1)
    log_frequency = max(max_fn//100, 1)
    moead = args.moead
    adj_rand_score = args.adj_rand_score
    min_max_bounds = None
    if adj_rand_score:
        assert objectives == [Objective.RMSE]
        min_max_bounds = np.array([[5, -1], [9, 0.5]],
                                  dtype=np.float32)

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

    target = MOEADRunner if moead else AGEMOEA2Runner
    runner = GenericABMRunnerMO(target,
                                market,
                                opt,
                                time_steps=ts,
                                mc_runs=mcruns,
                                objectives=objectives,
                                discretize_search_space=True,
                                n_workers=workers,
                                population_size=pop_size,
                                seed=seed,
                                max_fn_evaluation=max_fn,
                                save_directory=Path(save_dir),
                                n_groups=n_groups,
                                add_adj_rand_score=adj_rand_score,
                                min_max_bounds=min_max_bounds,
                                milestone_interval=milestone_interval,
                                log_frequency=log_frequency)

    runner.run()
