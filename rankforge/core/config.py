"""全局配置：支持 ENV_RANKFORGE_* 环境变量覆盖 +  schema 校验。"""
from __future__ import annotations

import os

from .errors import ConfigError


class RankForgeConfig:
    """系统配置。环境变量 RANKFORGE_<KEY> 可覆盖任意字段。"""

    def __init__(
        self,
        seed: int = 42,
        n_features: int = 10,
        n_queries: int = 200,
        docs_per_query: int = 15,
        relevance_snr: float = 3.0,
        nonlinear: float = 1.2,
        test_size: float = 0.2,
        val_size: float = 0.2,
        default_backends: tuple = ("lambdamart", "xgboost", "ranknet", "pointwise", "random", "heuristic"),
        ks: tuple = (1, 3, 5, 10),
        win_rel_threshold: float = 0.05,
    ) -> None:
        self.seed = seed
        self.n_features = n_features
        self.n_queries = n_queries
        self.docs_per_query = docs_per_query
        self.relevance_snr = relevance_snr
        self.nonlinear = nonlinear
        self.test_size = test_size
        self.val_size = val_size
        self.default_backends = tuple(default_backends)
        self.ks = tuple(ks)
        self.win_rel_threshold = win_rel_threshold

    @classmethod
    def from_env(cls) -> RankForgeConfig:
        def gi(k: str, d: int) -> int:
            v = os.environ.get(f"RANKFORGE_{k}")
            return int(v) if v is not None else d

        def gf(k: str, d: float) -> float:
            v = os.environ.get(f"RANKFORGE_{k}")
            return float(v) if v is not None else d

        return cls(
            seed=gi("SEED", 42),
            n_features=gi("N_FEATURES", 10),
            n_queries=gi("N_QUERIES", 200),
            docs_per_query=gi("DOCS_PER_QUERY", 15),
            relevance_snr=gf("RELEVANCE_SNR", 3.0),
            nonlinear=gf("NONLINEAR", 1.2),
            test_size=gf("TEST_SIZE", 0.2),
            val_size=gf("VAL_SIZE", 0.2),
            win_rel_threshold=gf("WIN_REL_THRESHOLD", 0.05),
        )

    def validate(self) -> None:
        if self.n_features < 1:
            raise ConfigError("n_features must be >= 1")
        if self.n_queries < 10:
            raise ConfigError("n_queries must be >= 10")
        if not (0.0 < self.test_size < 1.0):
            raise ConfigError("test_size must be in (0, 1)")
        if not (0.0 < self.val_size < 1.0):
            raise ConfigError("val_size must be in (0, 1)")
        if self.test_size + self.val_size >= 1.0:
            raise ConfigError("test_size + val_size must be < 1")
        if self.win_rel_threshold <= 0:
            raise ConfigError("win_rel_threshold must be > 0")

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_features": self.n_features,
            "n_queries": self.n_queries,
            "docs_per_query": self.docs_per_query,
            "relevance_snr": self.relevance_snr,
            "nonlinear": self.nonlinear,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "ks": list(self.ks),
            "default_backends": list(self.default_backends),
            "win_rel_threshold": self.win_rel_threshold,
        }
