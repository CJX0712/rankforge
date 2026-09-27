"""eval/benchmark.py — per-ranker evaluation orchestration + aggregation."""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from ..core.types import BenchmarkRow, RankingDataset, RankerResult
from ..rankers.base import BaseRanker
from .metrics import evaluate_dataset


def evaluate_ranker(
    ranker: BaseRanker, train: RankingDataset, test: RankingDataset
) -> Tuple[BenchmarkRow, Dict[str, float]]:
    if not ranker.is_available():
        return (
            BenchmarkRow(ranker=ranker.name, available=False, note="backend unavailable"),
            {},
        )
    t0 = time.perf_counter()
    ranker.fit(train)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    res: RankerResult = ranker.predict_scores(test)
    t_pred = time.perf_counter() - t1

    y_true = [q.relevance_vector() for q in test.queries]
    y_score = [res.scores[q.query_id] for q in test.queries]
    metrics = evaluate_dataset(y_true, y_score)

    row = BenchmarkRow(
        ranker=ranker.name,
        available=True,
        ndcg_at_1=metrics["ndcg@1"],
        ndcg_at_3=metrics["ndcg@3"],
        ndcg_at_5=metrics["ndcg@5"],
        ndcg_at_10=metrics["ndcg@10"],
        map=metrics["map"],
        precision_at_5=metrics["p@5"],
        mrr=metrics["mrr"],
        train_seconds=t_train,
        predict_seconds=t_pred,
    )
    return row, metrics


def aggregate(per_ranker_seed: Dict[str, List[Dict[str, float]]]) -> Dict[str, dict]:
    """Aggregate per-seed metric dicts into mean ± std per ranker."""
    keys = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "map", "p@5", "mrr"]
    out: Dict[str, dict] = {}
    for name, lst in per_ranker_seed.items():
        if not lst:
            out[name] = {"available": False}
            continue
        out[name] = {"available": True, "n_seeds": len(lst)}
        for k in keys:
            vals = np.array([m[k] for m in lst], dtype=np.float64)
            out[name][k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out
