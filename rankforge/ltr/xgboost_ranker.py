"""XGBoost rank:ndcg 后端（业界 SOTA 排序模型）。"""
from __future__ import annotations

import numpy as np

from ..core.types import LTRDataset, RankResult
from .metrics import rank_result_from


def available_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


class XgbRanker:
    name = "xgboost"
    available = available_xgboost()

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 120,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        objective: str = "rank:ndcg",
        normalize: bool = True,
        **kw,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.normalize = normalize
        self.kw = kw
        self.model = None
        self.scaler = None

    def fit(self, train: LTRDataset, val: LTRDataset | None = None) -> XgbRanker:
        import xgboost as xgb

        X, y, g = train.Xy_group()
        if self.normalize:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        self.model = xgb.XGBRanker(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            random_state=self.seed,
            n_jobs=1,
            verbosity=0,
            **self.kw,
        )
        self.model.fit(X, y, group=g)
        return self

    def predict(self, data: LTRDataset) -> np.ndarray:
        X, _, _ = data.Xy_group()
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, data: LTRDataset, ks) -> RankResult:
        _, y, g = data.Xy_group()
        return rank_result_from(self.name, y, self.predict(data), g, ks)
