"""LightGBM lambdamart 后端（业界 SOTA 排序模型）。"""
from __future__ import annotations

import numpy as np

from ..core.types import LTRDataset, RankResult
from .metrics import rank_result_from


def available_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


class LgbmRanker:
    name = "lambdamart"
    available = available_lightgbm()

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 120,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        normalize: bool = True,
        **kw,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.kw = kw
        self.model = None
        self.scaler = None

    def fit(self, train: LTRDataset, val: LTRDataset | None = None) -> LgbmRanker:
        import lightgbm as lgb

        X, y, g = train.Xy_group()
        if self.normalize:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        params = dict(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            n_jobs=1,
            verbose=-1,
            **self.kw,
        )
        self.model = lgb.LGBMRanker(**params)
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
