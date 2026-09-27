"""preprocess/standardize.py — leakage-safe feature standardization.

The scaler is fit ONLY on the training split, then applied to val/test. This is the
single place where statistics touch training data, so the leakage guard lives here.
"""
from __future__ import annotations

import numpy as np

from ..core.errors import DataLeakError
from ..core.types import RankingDataset


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._fitted = False

    def fit(self, dataset: RankingDataset) -> "Standardizer":
        X = dataset.features_matrix()
        if X.shape[0] == 0:
            raise DataLeakError("cannot fit Standardizer on empty train set")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        self._fitted = True
        return self

    def transform(self, dataset: RankingDataset) -> RankingDataset:
        if not self._fitted:
            raise DataLeakError("Standardizer used before fit()")
        out = RankingDataset(name=dataset.name, queries=[])
        for q in dataset.queries:
            docs = [
                type(d)(
                    doc_id=d.doc_id,
                    features=(d.features - self.mean_) / self.std_,
                    relevance=d.relevance,
                )
                for d in q.docs
            ]
            out.queries.append(type(q)(query_id=q.query_id, docs=docs))
        return out

    def fit_transform(self, dataset: RankingDataset) -> RankingDataset:
        return self.fit(dataset).transform(dataset)
