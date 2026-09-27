"""core/seed.py — global deterministic seeding.

Single entry point for reproducibility. ``set_all`` seeds numpy, python ``random``
and (if present) torch, so that every downstream module observes the same RNG state
for a given seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

_SEED_HOLDER = {"seed": 42}


def set_all(seed: int = 42) -> int:
    """Seed every available RNG deterministically. Returns the active seed."""
    seed = int(seed)
    _SEED_HOLDER["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional; never required for RankForge
        import torch

        torch.manual_seed(seed)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def get_seed() -> int:
    return _SEED_HOLDER["seed"]


@dataclass
class RngBundle:
    """Convenience bundle exposing fresh, labelled RNG streams from one master seed."""

    seed: int

    def numpy(self, label: int = 0) -> np.random.Generator:
        return np.random.default_rng((self.seed * 1000003 + label) & 0xFFFFFFFF)

    def python(self, label: int = 0) -> random.Random:
        return random.Random((self.seed * 1000003 + label) & 0xFFFFFFFF)
