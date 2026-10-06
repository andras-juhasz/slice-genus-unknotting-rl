from typing import List, Optional, Tuple
import json
from multiprocessing import Pool

from spherogram import Link

from agent import (band_operations_random_walk_interface,
                   nim_set_seed)
from seeding import derive_seed, resolve_seed
from utils import split_testing_workload, encode_link, decode_link


class RandomWalkParam:
    """Parameters for the random walk."""
    p_twist: float
    p_end: float
    max_twists: int
    max_actions: int
    sampling_method: str
    p_birth: float

    def __init__(self,
                 p_twist: float,
                 p_end: float,
                 max_twists: int,
                 max_actions: int,
                 sampling_method: str = 'consecutive',
                 p_birth: float = 0.0):
        """
        Initializes an instance of RandomWalkParam with all
        the parameters filled in. The start, over, under
        probabilities are always set to 1.0.

        Args:
            p_twist: the twist probability
            p_end: the end probability
            max_twists: the maximum number of total twists
                allowed across all the faces
            max_actions: the maximum number of actions per
                episode
            sampling_method: the sampling method; can be
                either "consecutive" or "random"
            p_birth: the probability of an unknot_birth
                action when no band is in progress; only
                meaningful for the "slice" and "strong
                slice" frameworks

        Raises:
            ValueError: if sampling_method is not
                "consecutive" or "random"
        """
        if sampling_method not in ['consecutive', 'random']:
            raise ValueError(f'"{sampling_method}" is not '
                             f'a valid sampling method')
        self.p_twist = p_twist
        self.p_end = p_end
        self.max_twists = max_twists
        self.max_actions = max_actions
        self.sampling_method = sampling_method
        self.p_birth = p_birth

    @staticmethod
    def default(framework: str) -> 'RandomWalkParam':
        # Bayesian-optimization-tuned per-framework defaults
        # (p_twist, p_end, max_twists, max_actions[, p_birth]).
        p_twist, p_end, max_twists, max_actions, p_birth = {
            'unknotting':    (0.0606, 90.4377, 1, 10, 0.0),
            'ribbon':        (0.3508, 98.3104, 1, 20, 0.0),
            'strong ribbon': (0.0234, 90.0185, 1, 17, 0.0),
            'slice':         (0.0119, 86.6341, 1, 21,
                              0.0053),
            'strong slice':  (0.0222, 89.7971, 4, 17,
                              0.0014),
        }[framework]
        return RandomWalkParam(p_twist, p_end, max_twists, max_actions,
                               p_birth=p_birth)


def encode_param(params: RandomWalkParam) -> str:
    return json.dumps({'p_twist': params.p_twist,
                       'p_end': params.p_end,
                       'max_twists': params.max_twists,
                       'max_actions': params.max_actions,
                       'sampling_method': params.sampling_method,
                       'p_birth': params.p_birth})


def band_operations_random_walk(
        framework: str,
        params: RandomWalkParam,
        dataset: List[Link],
        total_steps: int,
        verbose: bool = False,
        seed: Optional[int] = None
) -> List[Tuple[int, List[str], List[Link], List[str]]]:
    """
    Given a list of links, calculates upper bounds of their
    link invariants by attempting the random walk over the
    dataset multiple times and taking the minimum number
    of crossing changes or genera of ribbon / slice surfaces for each
    link.

    Args:
        framework: the type of link invariant that we are
            calculating; for a detailed explanation, see the
            docstring of agent.band_operation_random_walk
        params: the parameters of the random walk
        dataset: the list of links whose link invariants we
            are calculating
        total_steps: the total number of steps allowed,
            shared by all the links in the dataset
        verbose: whether we are outputting the resets of the
            current link and the actions taken by the random
            walk agent
        seed: if given, the seed of the random walk, which
            draws only from the Nim generator of agent.so;
            if None, that generator continues its stream

    Returns:
        A list of 4-tuples for each link in dataset. Their
        meaning are explained in the docstring of
        agent.band_operation_random_walk.
    """
    if seed is not None:
        nim_set_seed(derive_seed(seed, 'agent'))
    ans_encoded = band_operations_random_walk_interface(
        framework, encode_param(params),
        [encode_link(link) for link in dataset],
        total_steps, verbose
    )
    return [(c1, c2,
             [decode_link(link_json) for link_json in c3],
             c4) for (c1, c2, c3, c4) in ans_encoded]


def band_operations_random_walk_parallel(
        framework: str,
        params: RandomWalkParam,
        dataset: List[Link],
        total_steps: int,
        num_workers: int,
        verbose: bool = False,
        seed: Optional[int] = None
) -> List[Tuple[int, List[str], List[Link], List[str]]]:
    """
    Given a list of links, calculates upper bounds of their
    link invariants by attempting the random walk over the
    dataset multiple times, with the dataset and total steps
    split between multiple worker processes, and taking the
    minimum number of crossing changes or genera of
    ribbon / slice surfaces for each link.

    Args:
        framework: the type of link invariant that we are
            calculating; for a detailed explanation, see the
            docstring of agent.band_operation_random_walk
        params: the parameters of the random walk
        dataset: the list of links whose link invariants we
            are calculating
        total_steps: the total number of steps allowed,
            shared by all the links in the dataset
        num_workers: the number of worker processes; its
            recommended value is the number of cores that
            can be utilized
        verbose: whether we are outputting the resets of the
            current link and the actions taken by the random
            walk agent
        seed: shard i of the split runs with
            derive_seed(seed, "shard", i); if None, a seed is
            drawn, so the shards still never share a random
            stream

    Returns:
        A list of 4-tuples for each link in dataset. Their
        meaning are explained in the docstring of
        agent.band_operation_random_walk.
    """
    master_seed = resolve_seed(seed)
    dataset_steps_per_worker = split_testing_workload(dataset, total_steps, num_workers)
    with Pool(num_workers) as p:
        worker_outputs = p.starmap(
            band_operations_random_walk,
            [(framework, params, worker_dataset,
              worker_steps, verbose,
              derive_seed(master_seed, 'shard', i))
             for i, (worker_dataset, worker_steps)
             in enumerate(dataset_steps_per_worker)]
        )
    return [c for worker_output in worker_outputs
            for c in worker_output]
