"""Global seeding for reproducible experiment runs."""
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    """
    Seeds Python, NumPy and PyTorch (CPU and CUDA) random number generators.

    Call this once at the top of a script's ``__main__`` block -- never at
    module import time. Library modules used to call ``set_seed()`` on import,
    which silently reset the RNG state of any caller that imported them midway
    through a run.

    Returns:
        int: the seed that was applied.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
