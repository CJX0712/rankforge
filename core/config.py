"""core/config.py — runtime configuration with ENV_XXX_* overrides + schema validation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import List, Tuple

from .errors import ConfigError


@dataclass
class Config:
    seed: int = 42
    n_queries: int = 100
    docs_min: int = 8
    docs_max: int = 20
    n_features: int = 16
    label_noise: float = 0.5
    # fraction of queries that are "hard" (low signal) — creates difficulty gradient
    hard_query_frac: float = 0.35
    n_seeds: int = 3
    seeds: Tuple[int, ...] = (42, 123, 2026)
    # train/val/test split fractions (by query)
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    standardize: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        def ei(key: str, default: int) -> int:
            v = os.environ.get(key)
            return int(v) if v not in (None, "") else default

        def ef(key: str, default: float) -> float:
            v = os.environ.get(key)
            return float(v) if v not in (None, "") else default

        return cls(
            seed=ei("RANKFORGE_SEED", 42),
            n_queries=ei("RANKFORGE_N_QUERIES", 100),
            docs_min=ei("RANKFORGE_DOCS_MIN", 8),
            docs_max=ei("RANKFORGE_DOCS_MAX", 20),
            n_features=ei("RANKFORGE_N_FEATURES", 16),
            label_noise=ef("RANKFORGE_LABEL_NOISE", 0.6),
            hard_query_frac=ef("RANKFORGE_HARD_QUERY_FRAC", 0.35),
            n_seeds=ei("RANKFORGE_N_SEEDS", 3),
            train_frac=ef("RANKFORGE_TRAIN_FRAC", 0.6),
            val_frac=ef("RANKFORGE_VAL_FRAC", 0.2),
            test_frac=ef("RANKFORGE_TEST_FRAC", 0.2),
            standardize=os.environ.get("RANKFORGE_STANDARDIZE", "1") != "0",
        )

    def validate(self) -> None:
        if self.docs_min < 2:
            raise ConfigError("docs_min must be >= 2")
        if self.docs_max < self.docs_min:
            raise ConfigError("docs_max must be >= docs_min")
        if self.n_features < 1:
            raise ConfigError("n_features must be >= 1")
        if not (0.0 <= self.label_noise <= 1.0):
            raise ConfigError("label_noise must be in [0, 1]")
        s = self.train_frac + self.val_frac + self.test_frac
        if abs(s - 1.0) > 1e-6:
            raise ConfigError(f"split fractions must sum to 1.0 (got {s})")
        if self.n_seeds < 1 or len(self.seeds) < 1:
            raise ConfigError("n_seeds must be >= 1")

    def as_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out
