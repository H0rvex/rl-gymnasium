import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed torch, numpy, and Python's global `random`.

    Callers should additionally seed the env (`env.reset(seed=seed)`,
    `env.action_space.seed(seed)`) and any per-use-site `random.Random(seed)`
    instances — this function only covers the global state.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
