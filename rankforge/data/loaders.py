"""文件载入：支持 libsvm ranking 格式（含 qid）。"""
from __future__ import annotations

import numpy as np

from ..core.types import LTRDataset, QuerySample


def load_libsvm(path: str) -> LTRDataset:
    """载入 svmlight / libsvm ranking 文件（按 qid 分组）。"""
    from sklearn.datasets import load_svmlight_file

    X, y, qid = load_svmlight_file(path, query_id=True)
    X = X.toarray()
    queries: list[QuerySample] = []
    for q in np.unique(qid):
        m = qid == q
        queries.append(
            QuerySample(
                qid=int(q),
                X=X[m].astype(np.float64),
                y=np.asarray(y[m]).astype(np.int64),
            )
        )
    return LTRDataset(queries=queries, name=path)
