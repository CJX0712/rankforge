"""rankers/pointwise.py — pointwise ranker (classic method / strong baseline).

Reuses scikit-learn ``LogisticRegression`` (multiclass grades 0-4) when available;
falls back to a pure-numpy multinomial logistic regression if sklearn is missing.
Ranking score = expected grade (probabilities @ grades), a standard pointwise proxy.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.errors import BackendUnavailableError
from ..core.types import RankingDataset, RankerResult
from .base import BaseRanker


class PointwiseLogisticRanker(BaseRanker):
    name = "PointwiseLogistic"
    backend = "sklearn"

    def __init__(self, seed: int = 42, max_iter: int = 400, fallback: bool = True) -> None:
        super().__init__(seed)
        self.max_iter = max_iter
        self.fallback = fallback
        self._model = None
        self._grades = np.arange(5, dtype=np.float64)
        self._use_sklearn = True

    def is_available(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: F401

            return True
        except Exception:
            return self.fallback

    def _fit(self, ds: RankingDataset) -> None:
        X = ds.features_matrix()
        y = ds.relevance_vector().astype(int)
        try:
            from sklearn.linear_model import LogisticRegression

            self._use_sklearn = True
            self._model = LogisticRegression(
                max_iter=self.max_iter,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=self.seed,
            )
            self._model.fit(X, y)
        except Exception:
            if not self.fallback:
                raise BackendUnavailableError("sklearn unavailable and fallback disabled")
            self._use_sklearn = False
            self._model = _NumpyMultinomialLogistic(seed=self.seed, max_iter=self.max_iter)
            self._model.fit(X, y)

    def _scores(self, ds: RankingDataset) -> Dict[str, np.ndarray]:
        qids, Xs, _ = self._per_query(ds)
        out: Dict[str, np.ndarray] = {}
        for qid, X in zip(qids, Xs):
            if self._use_sklearn:
                proba = self._model.predict_proba(X)
            else:
                proba = self._model.predict_proba(X)
            out[qid] = proba @ self._grades
        return out


class _NumpyMultinomialLogistic:
    """Minimal pure-numpy multinomial logistic regression (one-vs-rest, Newton-lite SGD)."""

    def __init__(self, seed: int = 42, max_iter: int = 400, lr: float = 0.5) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self.lr = lr
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyMultinomialLogistic":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        classes = np.unique(y)
        self.classes_ = classes
        n_feat = Xb.shape[1]
        n_cls = len(classes)
        W = rng.normal(0.0, 0.01, size=(n_cls, n_feat))
        yidx = np.array([int(np.where(classes == v)[0][0]) for v in y])
        for _ in range(self.max_iter):
            logits = W @ Xb.T  # (n_cls, n)
            logits -= logits.max(axis=0, keepdims=True)
            exp = np.exp(logits)
            prob = exp / exp.sum(axis=0, keepdims=True)
            grad = (prob - np.eye(n_cls)[:, yidx].T.T) @ Xb  # (n_cls, n_feat)
            W -= self.lr / X.shape[0] * grad
        self.coef_ = W[:, 1:]
        self.intercept_ = W[:, 0]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        logits = self.coef_ @ X.T + self.intercept_[:, None]
        logits -= logits.max(axis=0, keepdims=True)
        exp = np.exp(logits)
        return (exp / exp.sum(axis=0, keepdims=True)).T
