"""Deterministic seeding for reproducibility (roadmap §9).

Always call ``set_seed`` at the top of any script or experiment driver. The seed
is the one declared in the YAML config, so swapping it requires editing config,
not code.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    ``deterministic_torch=True`` flips cudnn into deterministic mode. We default
    it on so dissertation results are reproducible end-to-end; switch it off
    only when debugging GPU performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            # Older torch or kernels without deterministic implementations.
            pass
