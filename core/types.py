"""core/types.py — shared dataclasses for the ranking domain."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Document:
    doc_id: str
    features: np.ndarray
    relevance: float = 0.0


@dataclass
class Query:
    query_id: str
    docs: List[Document] = field(default_factory=list)

    def features_matrix(self) -> np.ndarray:
        return np.asarray([d.features for d in self.docs], dtype=np.float64)

    def relevance_vector(self) -> np.ndarray:
        return np.asarray([d.relevance for d in self.docs], dtype=np.float64)


@dataclass
class RankingDataset:
    name: str
    queries: List[Query] = field(default_factory=list)

    def features_matrix(self) -> np.ndarray:
        return np.vstack([q.features_matrix() for q in self.queries])

    def relevance_vector(self) -> np.ndarray:
        return np.concatenate([q.relevance_vector() for q in self.queries])

    def group_sizes(self) -> List[int]:
        return [len(q.docs) for q in self.queries]

    def n_docs(self) -> int:
        return sum(len(q.docs) for q in self.queries)

    def n_queries(self) -> int:
        return len(self.queries)


@dataclass
class MetricResult:
    name: str
    value: float


@dataclass
class RankerResult:
    name: str
    scores: Dict[str, np.ndarray]  # query_id -> score vector (higher = better)
    available: bool = True
    note: str = ""


@dataclass
class BenchmarkRow:
    ranker: str
    available: bool
    ndcg_at_1: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    map: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    note: str = ""
