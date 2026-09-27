"""core/interfaces.py — Protocol contracts for pluggable rankers."""
from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

from .types import RankingDataset, RankerResult


@runtime_checkable
class Ranker(Protocol):
    """Every ranking backend implements this contract.

    Score semantics are unified: ``predict_scores`` returns a dict mapping
    query_id -> score vector where *higher score means better rank*.
    """

    name: str

    def fit(self, dataset: RankingDataset) -> "Ranker":
        ...

    def predict_scores(self, dataset: RankingDataset) -> RankerResult:
        ...

    def is_available(self) -> bool:
        ...
