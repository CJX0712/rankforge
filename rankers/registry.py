"""rankers/registry.py — build the full ranker roster for a seed."""
from __future__ import annotations

from typing import Dict, List

from ..core.config import Config
from .base import BaseRanker
from .baselines import ConstantScorer, RandomScorer
from .lambdamart import LambdaMARTRanker
from .listwise import ListNetRanker
from .pairwise import RankNetRanker
from .pointwise import PointwiseLogisticRanker


def build_rankers(seed: int, config: Config | None = None) -> List[BaseRanker]:
    """SOTA backends first, then offline numpy fallbacks, then baselines."""
    return [
        LambdaMARTRanker(seed=seed, backend="lightgbm"),
        LambdaMARTRanker(seed=seed, backend="xgboost"),
        PointwiseLogisticRanker(seed=seed),
        RankNetRanker(seed=seed),
        ListNetRanker(seed=seed),
        RandomScorer(seed=seed),
        ConstantScorer(seed=seed),
    ]


def build_by_name(seed: int, name: str) -> BaseRanker:
    mapping: Dict[str, BaseRanker] = {
        "LambdaMART": LambdaMARTRanker(seed=seed, backend="lightgbm"),
        "XGBRanker": LambdaMARTRanker(seed=seed, backend="xgboost"),
        "PointwiseLogistic": PointwiseLogisticRanker(seed=seed),
        "RankNet": RankNetRanker(seed=seed),
        "ListNet": ListNetRanker(seed=seed),
        "Random": RandomScorer(seed=seed),
        "Constant": ConstantScorer(seed=seed),
    }
    if name not in mapping:
        raise KeyError(f"unknown ranker: {name}")
    return mapping[name]
