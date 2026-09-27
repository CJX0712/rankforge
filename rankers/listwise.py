"""rankers/listwise.py — ListNet (listwise, pure numpy, Tier-1 offline fallback).

Linear scoring ``s(x) = x @ w`` trained with the ListNet top-1 probability matching
loss (Cao et al., 2007). The target distribution uses DCG gains ``2^rel - 1`` so the
model focuses on ranking the top of each list — a listwise objective that directly
targets NDCG, unlike pointwise or pairwise formulations.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.seed import RngBundle
from ..core.types import RankingDataset, RankerResult
from .base import BaseRanker


class ListNetRanker(BaseRanker):
    name = "ListNet"
    backend = "tier-1-numpy"

    def __init__(
        self,
        seed: int = 42,
        lr: float = 0.05,
        epochs: int = 35,
        max_grad_norm: float = 5.0,
        target_mode: str = "gain",
    ) -> None:
        super().__init__(seed)
        self.lr = lr
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.target_mode = target_mode  # "gain" (DCG 2^rel-1) or "raw" (relevance)
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def _target(self, y: np.ndarray) -> np.ndarray:
        if self.target_mode == "raw":
            gains = y.astype(np.float64) + 1.0
        else:  # DCG gain, emphasizes the top of the list
            gains = np.power(2.0, y) - 1.0
        gains = np.clip(gains, 1e-6, None)
        return gains / gains.sum()

    def _fit(self, ds: RankingDataset) -> None:
        qids, Xs, ys = self._per_query(ds)
        n_feat = Xs[0].shape[1]
        rng = RngBundle(self.seed).numpy(13)
        w = rng.normal(0.0, 0.01, size=n_feat)
        b = 0.0
        for _ in range(self.epochs):
            grad_w = np.zeros(n_feat)
            grad_b = 0.0
            n_q = 0
            for X, y in zip(Xs, ys):
                if len(y) < 2:
                    continue
                s = X @ w + b
                s = s - s.max()
                p = np.exp(s)
                p = p / p.sum()
                t = self._target(y)
                delta = p - t  # (n,)
                grad_w += X.T @ delta
                grad_b += delta.sum()
                n_q += 1
            if n_q == 0:
                continue
            gw = grad_w / n_q
            gb = grad_b / n_q
            norm = np.sqrt(np.sum(gw**2) + gb**2)
            if norm > self.max_grad_norm:
                gw = gw * self.max_grad_norm / norm
                gb = gb * self.max_grad_norm / norm
            w -= self.lr * gw
            b -= self.lr * gb
        self.w = w
        self.b = b

    def _scores(self, ds: RankingDataset) -> Dict[str, np.ndarray]:
        qids, Xs, _ = self._per_query(ds)
        out: Dict[str, np.ndarray] = {}
        for qid, X in zip(qids, Xs):
            out[qid] = X @ self.w + self.b
        return out
