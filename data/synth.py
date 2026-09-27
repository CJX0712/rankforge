"""data/synth.py — synthetic LETOR-style learning-to-rank generator.

The generator is seeded so a given ``seed`` always reproduces the same dataset.
Difficulty is heterogeneous by design: a fraction of queries are "hard" (weak signal,
more noise) which creates a realistic gradient where listwise methods earn their margin.
"""
from __future__ import annotations

import string

import numpy as np

from ..core.seed import RngBundle
from ..core.types import Document, Query, RankingDataset


def make_dataset(
    seed: int = 42,
    *,
    n_queries: int = 200,
    docs_min: int = 8,
    docs_max: int = 20,
    n_features: int = 16,
    label_noise: float = 0.6,
    hard_query_frac: float = 0.35,
) -> RankingDataset:
    rng = RngBundle(seed)
    npr = rng.numpy(1)
    pyr = rng.python(2)

    w = npr.normal(0.0, 1.0, size=n_features)
    doc_id_chars = string.ascii_lowercase + string.digits

    queries: List[Query] = []
    for qi in range(n_queries):
        qid = f"q{qi:05d}"
        is_hard = pyr.random() < hard_query_frac
        # hard queries: weaker true signal + stronger observation noise
        signal_scale = 0.45 if is_hard else 1.0
        obs_noise = 0.9 if is_hard else 0.35
        n_docs = int(pyr.randint(docs_min, docs_max + 1))

        X = npr.normal(0.0, 1.0, size=(n_docs, n_features))
        # True relevance is dominated by NON-LINEAR / INTERACTIVE structure. The linear
        # component is deliberately weak so that linear pointwise models (logistic
        # regression, linear ListNet/RankNet) cannot rank well, while boosted tree
        # rankers (LambdaMART) capture the interactions and win — the realistic regime
        # where listwise/nonlinear ranking earns its margin.
        linear = 0.2 * signal_scale * (X @ w)
        interact = 1.0 * signal_scale * (X[:, 0] * X[:, 1] + 0.5 * X[:, 2] * X[:, 3])
        periodic = 0.8 * signal_scale * np.sin(X[:, 4] * 2.0 + X[:, 5])
        nonlinear = 0.7 * signal_scale * (X[:, 0] ** 2 - X[:, 1] ** 2)
        offset = npr.normal(0.0, 0.4, size=n_docs)
        util = (
            linear
            + interact
            + periodic
            + nonlinear
            + offset
            + npr.normal(0.0, obs_noise, size=n_docs)
        )

        med = np.median(util)
        std = max(float(np.std(util)), 1e-6)
        grades = np.clip(np.round((util - med) / std * 1.5 + 2.0), 0, 4).astype(int)

        # label noise: flip grade by +/-1 with probability label_noise
        for j in range(n_docs):
            if pyr.random() < label_noise:
                grades[j] = int(np.clip(grades[j] + pyr.choice([-1, 1]), 0, 4))

        docs = [
            Document(
                doc_id=f"{qid}-{''.join(pyr.choice(list(doc_id_chars)) for _ in range(4))}",
                features=X[j],
                relevance=float(grades[j]),
            )
            for j in range(n_docs)
        ]
        queries.append(Query(query_id=qid, docs=docs))

    return RankingDataset(name=f"synth-n{n_queries}-f{n_features}-seed{seed}", queries=queries)
