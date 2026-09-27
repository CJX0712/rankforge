"""rankers/pairwise.py — RankNet (pairwise, pure numpy, Tier-1 offline fallback).

Linear scoring ``s(x) = x @ w`` trained with the RankNet pairwise sigmoid loss
(Burges et al., 2005). Listwise structure enters through within-query preference
pairs. Gradients are clipped for stability and pair sampling is seeded.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..core.seed import RngBundle
from ..core.types import RankingDataset, RankerResult
from .base import BaseRanker


class RankNetRanker(BaseRanker):
    name = "RankNet"
    backend = "tier-1-numpy"

    def __init__(
        self,
        seed: int = 42,
        lr: float = 0.05,
        epochs: int = 30,
        max_pairs: int = 400,
        max_grad_norm: float = 5.0,
    ) -> None:
        super().__init__(seed)
        self.lr = lr
        self.epochs = epochs
        self.max_pairs = max_pairs
        self.max_grad_norm = max_grad_norm
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def _build_pairs(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> List[Tuple[int, int, float]]:
        n = len(y)
        pairs: List[Tuple[int, int, float]] = []
        idx = np.arange(n)
        # sample candidate pairs, keep only those with differing relevance
        attempts = 0
        max_attempts = max(self.max_pairs * 4, 50)
        while len(pairs) < self.max_pairs and attempts < max_attempts:
            attempts += 1
            i, j = rng.integers(0, n, size=2).tolist()
            if i == j:
                continue
            s = np.sign(y[i] - y[j])
            if s == 0:
                continue
            pairs.append((i, j, float(s)))
        return pairs

    def _fit(self, ds: RankingDataset) -> None:
        qids, Xs, ys = self._per_query(ds)
        n_feat = Xs[0].shape[1]
        rng = RngBundle(self.seed).numpy(11)
        w = rng.normal(0.0, 0.01, size=n_feat)
        b = 0.0
        for epoch in range(self.epochs):
            grad_w = np.zeros(n_feat)
            grad_b = 0.0
            pair_count = 0
            for X, y in zip(Xs, ys):
                if len(y) < 2:
                    continue
                pairs = self._build_pairs(X, y, rng)
                s = X @ w + b
                for i, j, Sij in pairs:
                    diff = s[i] - s[j]
                    sigma = 1.0 / (1.0 + np.exp(-diff))
                    lam = -(Sij - sigma)
                    grad_w += lam * (X[i] - X[j])
                    grad_b += lam
                    pair_count += 1
            if pair_count == 0:
                continue
            gw = grad_w / pair_count
            gb = grad_b / pair_count
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
