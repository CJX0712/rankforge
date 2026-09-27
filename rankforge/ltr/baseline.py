"""强基线集合：Random / Pointwise(LR) / Heuristic。

用于对照 SOTA 与离线旗舰 RankNet 的增益。均为离线零依赖实现。
"""
from __future__ import annotations

import numpy as np

from ..core.types import LTRDataset, RankResult
from .metrics import rank_result_from


class RandomRanker:
    name = "random"
    available = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, train: LTRDataset, val: LTRDataset | None = None) -> RandomRanker:
        return self

    def predict(self, data: LTRDataset) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        scores = np.empty(data.n_docs, dtype=np.float64)
        pos = 0
        for q in data.queries:
            scores[pos : pos + q.n_docs] = rng.standard_normal(q.n_docs)
            pos += q.n_docs
        return scores

    def evaluate(self, data: LTRDataset, ks) -> RankResult:
        _, y, g = data.Xy_group()
        return rank_result_from(self.name, y, self.predict(data), g, ks)


class PointwiseRanker:
    """逐点回归基线：把相关度等级当作回归目标，忽略序结构。"""

    name = "pointwise"
    available = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model = None

    def fit(self, train: LTRDataset, val: LTRDataset | None = None) -> PointwiseRanker:
        from sklearn.linear_model import LinearRegression

        X, y, _ = train.Xy_group()
        self.model = LinearRegression()
        self.model.fit(X, y.astype(np.float64))
        return self

    def predict(self, data: LTRDataset) -> np.ndarray:
        X, _, _ = data.Xy_group()
        return self.model.predict(X)

    def evaluate(self, data: LTRDataset, ks) -> RankResult:
        _, y, g = data.Xy_group()
        return rank_result_from(self.name, y, self.predict(data), g, ks)


class HeuristicRanker:
    """朴素启发式：正特征之和作为分数（无训练）。"""

    name = "heuristic"
    available = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, train: LTRDataset, val: LTRDataset | None = None) -> HeuristicRanker:
        return self

    def predict(self, data: LTRDataset) -> np.ndarray:
        X, _, _ = data.Xy_group()
        return X.sum(axis=1)

    def evaluate(self, data: LTRDataset, ks) -> RankResult:
        _, y, g = data.Xy_group()
        return rank_result_from(self.name, y, self.predict(data), g, ks)
