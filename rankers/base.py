"""rankers/base.py — abstract base + registry for all ranking backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from ..core.types import RankingDataset, RankerResult


class BaseRanker(ABC):
    name: str = "base"
    backend: str = "tier-1"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self._fitted = False

    @abstractmethod
    def _fit(self, ds: RankingDataset) -> None:
        ...

    @abstractmethod
    def _scores(self, ds: RankingDataset) -> Dict[str, np.ndarray]:
        ...

    def fit(self, ds: RankingDataset) -> "BaseRanker":
        self._fit(ds)
        self._fitted = True
        return self

    def predict_scores(self, ds: RankingDataset) -> RankerResult:
        if not self._fitted:
            raise RuntimeError(f"{self.name} used before fit()")
        scores = self._scores(ds)
        return RankerResult(name=self.name, scores=scores, available=True)

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _per_query(ds: RankingDataset):
        Xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        qids: List[str] = []
        for q in ds.queries:
            Xs.append(q.features_matrix())
            ys.append(q.relevance_vector())
            qids.append(q.query_id)
        return qids, Xs, ys
