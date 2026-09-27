"""数据类型定义（dataclass）。

排序以 query 为分组单位：每个 query 含若干文档，文档带特征向量与相关度等级
（relevance grade，通常 0..4）。评测只在 query 内部进行。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuerySample:
    """单个 query 下的文档集合。"""

    qid: int
    X: np.ndarray  # (n_docs, n_features) float64
    y: np.ndarray  # (n_docs,) int64 相关度等级

    @property
    def n_docs(self) -> int:
        return int(self.X.shape[0])


@dataclass
class LTRDataset:
    """按 query 分组的排序数据集。"""

    queries: list[QuerySample]
    name: str = "synthetic"

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_docs(self) -> int:
        return sum(q.n_docs for q in self.queries)

    @property
    def n_features(self) -> int:
        return int(self.queries[0].X.shape[1])

    def Xy_group(self):
        """拼平为全局 X / y 与 group 长度列表（LightGBM / XGBoost 所需）。"""
        Xs, ys, group = [], [], []
        for q in self.queries:
            Xs.append(q.X)
            ys.append(np.asarray(q.y, dtype=np.float64))
            group.append(q.n_docs)
        return np.vstack(Xs), np.concatenate(ys), group


@dataclass
class RankResult:
    """某后端在某数据集上的评测结果。"""

    backend: str
    ndcg: dict[int, float] = field(default_factory=dict)  # k -> NDCG@k
    map: float = 0.0
    skipped: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "ndcg": {str(k): float(v) for k, v in self.ndcg.items()},
            "map": float(self.map),
            "skipped": bool(self.skipped),
            "note": self.note,
        }


@dataclass
class SeedResult:
    """单 seed 下所有后端的评测结果。"""

    seed: int
    results: dict[str, RankResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"seed": int(self.seed), "results": {n: r.to_dict() for n, r in self.results.items()}}
