"""后端注册表：名称 -> 工厂，支持 available_*() 探测与 auto 解析。"""
from __future__ import annotations

from ..core.errors import BackendError
from .baseline import HeuristicRanker, PointwiseRanker, RandomRanker
from .lightgbm_ranker import LgbmRanker
from .ranknet import RankNet
from .xgboost_ranker import XgbRanker

_REGISTRY = {
    "random": RandomRanker,
    "pointwise": PointwiseRanker,
    "heuristic": HeuristicRanker,
    "ranknet": RankNet,
    "lambdamart": LgbmRanker,
    "xgboost": XgbRanker,
}


def list_backends() -> list:
    return list(_REGISTRY.keys())


def available_backends() -> list:
    out = []
    for n, cls in _REGISTRY.items():
        try:
            out.append((n, bool(cls.available)))
        except Exception:  # noqa: BLE001
            out.append((n, False))
    return out


def resolve_backends(spec) -> list:
    if spec is None or list(spec) in (["auto"], []):
        spec = list(_REGISTRY.keys())
    names = list(spec)
    resolved = []
    for n in names:
        if n not in _REGISTRY:
            raise BackendError(f"unknown backend: {n}")
        resolved.append(n)
    return resolved


def make_backend(name: str, **kw):
    cls = _REGISTRY[name]
    return cls(**kw)
