import os
from multiprocessing import set_start_method

from dataset import Dataset
from experiment import (RLParam, train_rl,
                        MixedStrategyParam,
                        band_operations_rl)
from seeding import derive_seed, resolve_seed

if __name__ == '__main__':
    set_start_method('spawn')

    # the master seed of the run: SEED from the environment,
    # or a fresh one; print it so the run can be repeated
    seed = resolve_seed(os.environ.get('SEED'))
    print(f'SEED: {seed}')

    framework = 'slice'
    dataset = Dataset.read_csv('../datasets/linkinfo-to-9.csv')[:3]

    params = RLParam(
        max_actions = 30,
        shape = (20, 4),
        sampling_method = 'random',
        features_cls=['DeterminantAndSignature', 'LinkingMatrix'],
        # old_reward = False selects the reward that charges
        # the answer, which needs gamma = 1, and mdp = True
        # makes the observation Markov. Their defaults are
        # the reward and the observation of the models
        # trained before them.
        gamma = 1.0,
        mdp = True,
        old_reward = False,
        # a record of the run; create_env also records in
        # these parameters the reward settings in force
        seed = seed
    )
    policy_kwargs = dict(
        net_arch=dict(pi=[64] * 2, vf=[64] * 2)
    )
    model, env = train_rl(framework, dataset.links, 8192,
                          params, policy_kwargs,
                          seed=derive_seed(seed, 'train'))
    mixed_params = MixedStrategyParam.default(framework)
    output = band_operations_rl(model, env, dataset.links, 2000,
                                mixed_params,
                                seed=derive_seed(seed, 'test'))
    dataset.evaluate_invariant(framework, output)
