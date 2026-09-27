"""rankers/lambdamart.py — LambdaMART SOTA backends (Tier-0).

Two tree-boosting backends that optimize a listwise NDCG objective directly:
  * ``lightgbm`` LGBMRanker (objective="lambdarank")  — the canonical LambdaMART
  * ``xgboost``  XGBRanker   (objective="rank:ndcg")

Both are imported lazily and reported unavailable (not crashed) if the wheel is
missing, so the offline numpy fallbacks still run and the benchmark stays green.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.errors import BackendUnavailableError
from ..core.types import RankingDataset, RankerResult
from .base import BaseRanker


class LambdaMARTRanker(BaseRanker):
    name = "LambdaMART"
    backend = "lightgbm"

    def __init__(
        self,
        seed: int = 42,
        backend: str = "lightgbm",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
    ) -> None:
        super().__init__(seed)
        self.backend_name = backend
        self.name = "LambdaMART" if backend == "lightgbm" else "XGBRanker"
        self.backend = backend
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self._model = None

    def is_available(self) -> bool:
        if self._model is not None:
            return True
        try:
            if self.backend_name == "lightgbm":
                import lightgbm  # noqa: F401
            else:
                import xgboost  # noqa: F401
            return True
        except Exception:
            return False

    def _fit(self, ds: RankingDataset) -> None:
        if not self.is_available():
            raise BackendUnavailableError(f"{self.backend_name} backend unavailable")
        X = ds.features_matrix()
        y = ds.relevance_vector()
        groups = ds.group_sizes()
        if self.backend_name == "lightgbm":
            from lightgbm import LGBMRanker

            self._model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.seed,
                verbosity=-1,
                n_jobs=1,
            )
            self._model.fit(X, y, group=groups)
        else:
            from xgboost import XGBRanker

            self._model = XGBRanker(
                objective="rank:ndcg",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.seed,
                verbosity=0,
                n_jobs=1,
            )
            self._model.fit(X, y, group=groups)

    def _scores(self, ds: RankingDataset) -> Dict[str, np.ndarray]:
        qids, Xs, _ = self._per_query(ds)
        out: Dict[str, np.ndarray] = {}
        for qid, X in zip(qids, Xs):
            out[qid] = np.asarray(self._model.predict(X), dtype=np.float64)
        return out
