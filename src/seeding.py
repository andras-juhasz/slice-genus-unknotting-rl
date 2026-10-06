import hashlib
import importlib
import random
import secrets
from typing import Dict, List, Optional, Union

import numpy as np

# Seeds stay below 2**63, so that every seed fits a Nim int
SEED_LIMIT = 2**63


def resolve_seed(value: Optional[Union[int, str]] = None) -> int:
    """
    Returns the master seed of a run: the given value, or a
    value drawn from the operating system when none is
    given.

    Args:
        value: an integer in [0, SEED_LIMIT=2**63), a string
            holding one, or None or an empty string

    Returns:
        The master seed.

    Raises:
        ValueError: if value is not an integer in
            [0, SEED_LIMIT=2**63)
    """
    if value is None or (isinstance(value, str)
                         and not value.strip()):
        return secrets.randbelow(SEED_LIMIT)
    seed = int(value)
    if not 0 <= seed < SEED_LIMIT:
        raise ValueError(f'seed {seed} is not in [0, 2**63=SEED_LIMIT={SEED_LIMIT})')
    return seed


def derive_seed(seed: int, *keys: Union[int, str]) -> int:
    """
    Derives a seed from a master seed and keys naming a unit
    of work or a random number generator, with NumPy's
    SeedSequence. Equal arguments give equal seeds in every
    process, and different keys give independent seeds.

    Args:
        seed: the master seed, an integer in [0, SEED_LIMIT=2**63)
        keys: non-negative integers or strings; the integer
            1 and the string "1" are different keys

    Returns:
        A seed in [1, SEED_LIMIT=2**63).

    Raises:
        ValueError: if seed or an integer key is negative
    """
    if seed < 0:
        raise ValueError(f'seed {seed} is negative')
    words: List[int] = []
    for key in keys:
        if isinstance(key, str):
            digest = hashlib.sha256(key.encode()).digest()
            words += [1, int.from_bytes(digest[:4], 'little')]
        elif key >= 0:
            words += [0, int(key)]
        else:
            raise ValueError(f'key {key} is negative')
    sequence = np.random.SeedSequence(entropy=seed,
                                      spawn_key=words)
    state = sequence.generate_state(1, dtype=np.uint64)
    return max(int(state[0]) >> 1, 1)


def set_all_seeds(seed: int) -> Dict[str, int]:
    """
    Seeds every random number generator a run draws from:
    Python's random module, NumPy's global generator,
    PyTorch, and the Nim generators of agent.so and
    representable.so, each with its own seed derived from
    the given one. A module that cannot be imported is
    skipped, since nothing in the process can draw from it.

    Args:
        seed: the seed, an integer in [0, SEED_LIMIT=2**63)

    Returns:
        The derived seed of every generator seeded, by name.

    Raises:
        RuntimeError: if agent.so or representable.so has no
            nim_set_seed, which means it was built from older
            sources
    """
    seeds = {'python': derive_seed(seed, 'python'),
             'numpy': derive_seed(seed, 'numpy')}
    random.seed(seeds['python'])
    # the legacy global generator takes 32 bits
    np.random.seed(seeds['numpy'] % 2**32)
    # imported here so that importing this module needs
    # neither PyTorch nor the compiled modules
    for name in ('torch', 'agent', 'representable'):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        seeds[name] = derive_seed(seed, name)
        if name == 'torch':
            module.manual_seed(seeds[name])
        elif hasattr(module, 'nim_set_seed'):
            module.nim_set_seed(seeds[name])
        else:
            raise RuntimeError(f'{name}.so has no nim_set_seed; '
                               f'rebuild it from the current '
                               f'sources')
    return seeds
