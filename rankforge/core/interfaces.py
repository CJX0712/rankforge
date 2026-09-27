"""接口契约（Protocol）。模块间只依赖抽象，不依赖具体实现。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import LTRDataset, RankResult


@runtime_checkable
class Ranker(Protocol):
    """排序后端契约。"""

    name: str
    available: bool

    def fit(self, train: LTRDataset, val: LTRDataset | None = None) -> Ranker: ...

    def predict(self, data: LTRDataset) -> np.ndarray: ...

    def evaluate(self, data: LTRDataset, ks) -> RankResult: ...
