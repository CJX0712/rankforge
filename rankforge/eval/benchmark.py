"""多 seed 基准评测与汇总。

统计严谨（V4）：性能对比 ≥3 seed，报告 mean±std；「胜基线」按均值判定，
且均值差需大于两组 std 之和的 1/2（简易显著性门槛）。
"""
from __future__ import annotations

import numpy as np

from ..core.config import RankForgeConfig
from ..core.seed import set_all
from ..core.types import RankResult, SeedResult
from ..data.synthetic import generate_ltr_dataset, split_dataset
from ..ltr.registry import make_backend, resolve_backends


class BenchmarkReport:
    def __init__(self, config: RankForgeConfig) -> None:
        self.config = config
        self.seeds: list[SeedResult] = []

    def add(self, sr: SeedResult) -> None:
        self.seeds.append(sr)

    def aggregate(self, backend: str, k: int = 10):
        vals = []
        for sr in self.seeds:
            r = sr.results.get(backend)
            if r is None or r.skipped:
                continue
            vals.append(r.ndcg.get(k, 0.0))
        if not vals:
            return (0.0, 0.0)
        arr = np.asarray(vals, dtype=float)
        return float(arr.mean()), float(arr.std())

    def to_dict(self) -> dict:
        backends = list(self.seeds[0].results.keys()) if self.seeds else []
        agg = {}
        for b in backends:
            m, s = self.aggregate(b, 10)
            agg[b] = {"ndcg@10_mean": m, "ndcg@10_std": s, "skipped": (s == 0.0 and m == 0.0)}
        return {
            "config": self.config.to_dict(),
            "seeds": [sr.to_dict() for sr in self.seeds],
            "aggregate_ndcg10": agg,
        }


def run_benchmark(
    config: RankForgeConfig,
    backends=None,
    seeds=None,
    progress: bool = False,
) -> BenchmarkReport:
    backends = resolve_backends(backends or list(config.default_backends))
    seeds = seeds if seeds is not None else [config.seed + i * 101 for i in range(3)]
    report = BenchmarkReport(config)
    for seed in seeds:
        set_all(seed)
        ds = generate_ltr_dataset(
            seed=seed,
            n_queries=config.n_queries,
            docs_per_query=config.docs_per_query,
            n_features=config.n_features,
            relevance_snr=config.relevance_snr,
            nonlinear=config.nonlinear,
        )
        train, val, test = split_dataset(ds, config.test_size, config.val_size, seed=seed)
        sr = SeedResult(seed=seed, results={})
        for name in backends:
            try:
                m = make_backend(name)
            except Exception as e:  # noqa: BLE001
                sr.results[name] = RankResult(name, skipped=True, note=f"init failed: {e}")
                continue
            if not getattr(m, "available", True):
                sr.results[name] = RankResult(name, skipped=True, note="backend unavailable")
                continue
            try:
                m.fit(train, val)
                res = m.evaluate(test, config.ks)
            except Exception as e:  # noqa: BLE001
                sr.results[name] = RankResult(name, skipped=True, note=f"run failed: {e}")
                continue
            sr.results[name] = res
        report.add(sr)
        if progress:
            print(f"  seed={seed} done")
    return report
