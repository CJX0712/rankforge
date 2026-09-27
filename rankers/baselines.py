"""rankers/baselines.py — naive/strong baselines (no learning)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.seed import RngBundle
from ..core.types import RankingDataset, RankerResult
from .base import BaseRanker


class RandomScorer(BaseRanker):
    name = "Random"
    backend = "baseline"

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)

    def is_available(self) -> bool:
        return True

    def _fit(self, ds: RankingDataset) -> None:
        return None

    def _scores(self, ds: RankingDataset) -> Dict[str, np.ndarray]:
        qids, Xs, _ = self._per_query(ds)
        rng = RngBundle(self.seed).numpy(777)
        out: Dict[str, np.ndarray] = {}
        for qid, X in zip(qids, Xs):
            out[qid] = rng.uniform(-1.0, 1.0, size=X.shape[0])
        return out


class ConstantScorer(BaseRanker):
    """Predicts a constant score; ranking collapses to input order. Lower bound baseline."""

    name = "Constant"
    backend = "baseline"

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)

    def is_available(self) -> bool:
        return True

    def _fit(self, ds: RankingDataset) -> None:
        return None

    def _scores(self, ds: RankingDataset) -> Dict[str, np.ndarray]:
        qids, Xs, _ = self._per_query(ds)
        out: Dict[str, np.ndarray] = {}
        for qid, X in zip(qids, Xs):
            out[qid] = np.zeros(X.shape[0], dtype=np.float64)
        return out
