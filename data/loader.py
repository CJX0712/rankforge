"""data/loader.py — group-aware (query-level) train/val/test split + LETOR file loader."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.errors import DataError, DataLeakError
from ..core.seed import RngBundle
from ..core.types import Query, RankingDataset


@dataclass
class Split:
    train: RankingDataset
    val: RankingDataset
    test: RankingDataset


def split_by_query(
    dataset: RankingDataset,
    *,
    seed: int = 42,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Split:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise DataError("split fractions must sum to 1.0")
    if dataset.n_queries() == 0:
        raise DataError("empty dataset")

    npr = RngBundle(seed).numpy(7)
    idx = np.arange(dataset.n_queries())
    npr.shuffle(idx)

    n = len(idx)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(0, int(round(n * val_frac)))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    def sub(ids: List[int], name: str) -> RankingDataset:
        return RankingDataset(name=name, queries=[dataset.queries[i] for i in ids])

    return Split(
        train=sub(list(train_idx), f"{dataset.name}[train]"),
        val=sub(list(val_idx), f"{dataset.name}[val]"),
        test=sub(list(test_idx), f"{dataset.name}[test]"),
    )


def load_letor(path: str, name: str = "letor") -> RankingDataset:
    """Parse standard LETOR format lines: ``<grade> qid:XX feat:val ... # docid``."""
    queries: dict = {}
    order: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                grade = float(parts[0])
            except ValueError:
                raise DataError(f"bad label in LETOR line: {line[:40]!r}")
            qid = None
            feats: dict = {}
            docid = ""
            for p in parts[1:]:
                if p.startswith("qid:"):
                    qid = p[4:]
                elif p.startswith("#"):
                    docid = p[1:].strip()
                    break
                elif ":" in p:
                    k, v = p.split(":", 1)
                    try:
                        feats[int(float(k))] = float(v)
                    except ValueError:
                        pass
            if qid is None:
                raise DataError("LETOR line missing qid:")
            if qid not in queries:
                queries[qid] = []
                order.append(qid)
            max_feat = max(feats) if feats else 0
            vec = np.zeros(max_feat, dtype=np.float64)
            for k, v in feats.items():
                vec[k - 1] = v
            queries[qid].append(Document(doc_id=docid or f"{qid}-{len(queries[qid])}", features=vec, relevance=grade))
    return RankingDataset(name=name, queries=[Query(query_id=q, docs=queries[q]) for q in order])
