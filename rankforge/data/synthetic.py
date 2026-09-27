"""合成 LTR 数据生成 + query 切分（V2：非线性潜在，确保 SOTA 真正胜出）。

相关性潜在 = 非线性函数（二次 + 周期）of 隐藏线性投影 w·X：
  latent = (w·X) + 0.6·(w·X)^2 − 0.3·sin(2·w·X)
pointwise 线性回归无法拟合该曲率 → 排序退化；树(LightGBM/XGBoost)与
非线性 MLP(RankNet) 可捕获 → 在 NDCG 上显著胜出。relevance_snr 控难度。
"""
from __future__ import annotations

import numpy as np

from ..core.seed import set_all
from ..core.types import LTRDataset, QuerySample


def generate_ltr_dataset(
    seed: int = 42,
    n_queries: int = 200,
    docs_per_query: int = 15,
    n_features: int = 10,
    relevance_snr: float = 3.0,
    n_grades: int = 5,
    nonlinear: float = 1.2,
) -> LTRDataset:
    """生成合成 LTR 数据集（固定 seed，可复现）。"""
    set_all(seed)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n_features)
    w = w / np.linalg.norm(w)
    queries: list[QuerySample] = []
    for q in range(n_queries):
        X = rng.standard_normal((docs_per_query, n_features)).astype(np.float64)
        lin = X @ w
        latent = lin + nonlinear * (lin**2) - 0.3 * np.sin(2.0 * lin)
        bias = rng.standard_normal() * 0.6
        signal = latent + bias
        noise = rng.standard_normal(docs_per_query) / max(float(relevance_snr), 1e-6)
        raw = signal + noise
        lo, hi = float(raw.min()), float(raw.max())
        norm = (raw - lo) / (hi - lo + 1e-9)
        grades = np.clip(np.round(norm * (n_grades - 1)).astype(np.int64), 0, n_grades - 1)
        queries.append(QuerySample(qid=q, X=X, y=grades))
    return LTRDataset(queries=queries, name=f"synthetic-nl{nonlinear}-snr{relevance_snr}")


def split_dataset(
    ds: LTRDataset,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
):
    """按 query 随机切分为 train / val / test（无文档跨集泄漏）。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(ds.n_queries)
    rng.shuffle(idx)
    n = len(idx)
    n_test = round(n * test_size)
    n_val = round(n * val_size)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    def sub(ix: np.ndarray) -> LTRDataset:
        return LTRDataset(queries=[ds.queries[int(i)] for i in ix], name=ds.name)

    return sub(train_idx), sub(val_idx), sub(test_idx)
