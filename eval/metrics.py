"""eval/metrics.py — ranking metrics (pure numpy, cross-validated).

Score semantics: higher predicted score = better rank. All metrics yield values in
[0, 1]. Each metric has an independent *reference* implementation used by the unit
tests to cross-validate the vectorized production code (no shared code path).
"""
from __future__ import annotations

import math
from typing import List

import numpy as np


def _dcg_gain(rel: float) -> float:
    return (2.0 ** rel - 1.0)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    order = np.argsort(y_score)[::-1][:k]
    dcg = 0.0
    for i, idx in enumerate(order):
        dcg += _dcg_gain(y_true[idx]) / math.log2(i + 2)
    ideal = np.sort(y_true)[::-1][:k]
    idcg = 0.0
    for i, r in enumerate(ideal):
        idcg += _dcg_gain(r) / math.log2(i + 2)
    return float(dcg / idcg) if idcg > 0 else 0.0


def ndcg_at_k_reference(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Independent loop-based reference (no vectorized argsort reuse)."""
    y_true = list(np.asarray(y_true, dtype=np.float64))
    y_score = list(np.asarray(y_score, dtype=np.float64))
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    pairs = sorted(range(n), key=lambda i: y_score[i], reverse=True)[:k]
    dcg = 0.0
    for pos, idx in enumerate(pairs):
        dcg += (_dcg_gain(y_true[idx])) / math.log2(pos + 2)
    order_true = sorted(range(n), key=lambda i: y_true[i], reverse=True)[:k]
    idcg = 0.0
    for pos, idx in enumerate(order_true):
        idcg += (_dcg_gain(y_true[idx])) / math.log2(pos + 2)
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0
    order = np.argsort(y_score)[::-1]
    if k is not None:
        order = order[:k]
    hits = 0
    sum_p = 0.0
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] > 0:
            hits += 1
            sum_p += hits / rank
    n_rel = int(np.sum(y_true > 0))
    return float(sum_p / n_rel) if n_rel > 0 else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    order = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[order] > 0))


def reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] > 0:
            return 1.0 / rank
    return 0.0


def evaluate_query(y_true: np.ndarray, y_score: np.ndarray, ks=(1, 3, 5, 10)) -> dict:
    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(y_true, y_score, k)
    out["map"] = average_precision(y_true, y_score)
    out["p@5"] = precision_at_k(y_true, y_score, 5)
    out["mrr"] = reciprocal_rank(y_true, y_score)
    return out


def evaluate_dataset(
    y_true_per_query: List[np.ndarray],
    y_score_per_query: List[np.ndarray],
    ks=(1, 3, 5, 10),
) -> dict:
    if len(y_true_per_query) != len(y_score_per_query):
        raise ValueError("mismatched query counts")
    agg = {f"ndcg@{k}": [] for k in ks}
    agg["map"] = []
    agg["p@5"] = []
    agg["mrr"] = []
    for yt, ys in zip(y_true_per_query, y_score_per_query):
        r = evaluate_query(yt, ys, ks=ks)
        for key in agg:
            agg[key].append(r[key])
    return {key: float(np.mean(vals)) for key, vals in agg.items()}
