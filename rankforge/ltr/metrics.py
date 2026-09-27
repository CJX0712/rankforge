"""IR 排序指标：NDCG@{k} 与 MAP（手写 numpy，无 sklearn 别名递归风险）。

约定：分数越大越靠前（排序越优）。
"""
from __future__ import annotations

import numpy as np

from ..core.types import RankResult


def _dcg(rels: np.ndarray, k: int) -> float:
    rels = np.asarray(rels, dtype=np.float64)
    if rels.size == 0:
        return 0.0
    kk = min(k, rels.size)
    gains = np.power(2.0, rels[:kk]) - 1.0
    discounts = np.log2(np.arange(1, kk + 1) + 1.0)
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, group: list, k: int
) -> float:
    """按 query 分组计算 NDCG@k（每组内部排序后评测）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    pos = 0
    scores: list[float] = []
    for g in group:
        g = int(g)
        yt = y_true[pos : pos + g]
        yp = y_pred[pos : pos + g]
        pos += g
        order = np.argsort(-yp, kind="stable")
        dcg = _dcg(yt[order], k)
        idcg = _dcg(np.sort(yt)[::-1], k)
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def average_precision(
    y_true: np.ndarray, y_pred: np.ndarray, group: list, k: int | None = None
) -> float:
    """按 query 分组计算 MAP（相关 = grade > 0）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    pos = 0
    aps: list[float] = []
    for g in group:
        g = int(g)
        yt = y_true[pos : pos + g]
        yp = y_pred[pos : pos + g]
        pos += g
        order = np.argsort(-yp, kind="stable")
        yt = yt[order]
        if k is not None:
            yt = yt[:k]
        rel = (yt > 0).astype(float)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        aps.append(float(np.sum(prec * rel) / rel.sum()))
    return float(np.mean(aps)) if aps else 0.0


def evaluate_ranking(
    y_true: np.ndarray, y_pred: np.ndarray, group: list, ks=(1, 3, 5, 10)
) -> dict:
    """返回 {k: NDCG@k, 'map': MAP}。"""
    res: dict = {}
    for k in ks:
        res[k] = ndcg_at_k(y_true, y_pred, group, k)
    res["map"] = average_precision(y_true, y_pred, group, max(ks))
    return res


def rank_result_from(name: str, y_true, y_pred, group, ks) -> RankResult:
    """构造 RankResult。"""
    res = evaluate_ranking(y_true, y_pred, group, ks)
    rr = RankResult(backend=name)
    rr.ndcg = {k: float(res[k]) for k in ks}
    rr.map = float(res["map"])
    return rr
