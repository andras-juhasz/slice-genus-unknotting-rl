import os
from multiprocessing import freeze_support
from bayes_opt import BayesianOptimization

from dataset import Dataset
from agent_py import (RandomWalkParam,
                      band_operations_random_walk_parallel)
from seeding import derive_seed, resolve_seed

if __name__ == '__main__':
    freeze_support()

    # the master seed of the run: SEED from the environment,
    # or a fresh one; print it so the run can be repeated
    seed = resolve_seed(os.environ.get('SEED'))
    print(f'SEED: {seed}')

    framework = 'unknotting'
    sample = Dataset.read_csv('../datasets/rolfsen.csv')[:25]
    # every evaluation gets its own seed, so repeated
    # parameters are independent draws of the objective
    evaluations = [0]

    def bayesian_optimization_objective(
            p_twist: float,
            p_end: float,
            _max_twists: float,
            _max_actions: float
    ) -> float:
        evaluations[0] += 1
        params = RandomWalkParam(p_twist, p_end,
                                 int(_max_twists),
                                 int(_max_actions))
        return sample.evaluate_invariant(
            framework, band_operations_random_walk_parallel(
                framework, params, sample.links, 50000, 12,
                seed=derive_seed(seed, 'bo', evaluations[0])
            ), False
        )

    pbounds = {'p_twist': (0.1, 5.0), 'p_end': (0.5, 10.0),
               '_max_twists': (1, 5), '_max_actions': (10, 50)}
    optimizer = BayesianOptimization(
        f=bayesian_optimization_objective, pbounds=pbounds,
        verbose=2, random_state=1
    )
    optimizer.maximize(init_points=10, n_iter=40)
    print(optimizer.max)
