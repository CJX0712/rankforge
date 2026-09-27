"""端到端演示：benchmark + 消融 + 失败案例 + 确定性校验，落盘 benchmark.json。

结论先行：打印对照表，写入结构化 JSON（真实来自运行输出，禁止编造）。
"""
from __future__ import annotations

import json
import sys

from ..core.config import RankForgeConfig
from ..pipeline.rank_pipeline import RankPipeline


def _win_check(report, cfg):
    """SOTA 最佳后端 vs 逐点强基线，统计严谨门槛。"""
    backends = list(report.seeds[0].results.keys())
    pw_mean, pw_std = report.aggregate("pointwise", 10) if "pointwise" in backends else (0.0, 0.0)
    sota_pool = [b for b in ("lambdamart", "xgboost", "ranknet") if b in backends]
    best_b, best_m, best_s = None, -1.0, 0.0
    for b in sota_pool:
        m, s = report.aggregate(b, 10)
        if not report.seeds[0].results[b].skipped and m > best_m:
            best_b, best_m, best_s = b, m, s
    if best_b is None or pw_mean <= 0:
        return {"passed": False, "reason": "无可用 SOTA 或基线"}
    rel = (best_m - pw_mean) / pw_mean
    sig = abs(best_m - pw_mean) > 0.5 * (best_s + pw_std)
    return {
        "passed": bool(rel >= cfg.win_rel_threshold and sig),
        "sota_best": best_b,
        "sota_ndcg10_mean": round(best_m, 4),
        "pointwise_ndcg10_mean": round(pw_mean, 4),
        "rel_gain": round(rel, 4),
        "abs_gain": round(best_m - pw_mean, 4),
        "significant": bool(sig),
        "threshold": cfg.win_rel_threshold,
    }


def _print_table(report, cfg):
    backends = list(report.seeds[0].results.keys())
    hdr = f"{'backend':<12}{'NDCG@1':<12}{'NDCG@3':<12}{'NDCG@5':<12}{'NDCG@10':<14}{'MAP':<10}"
    print(hdr)
    print("-" * len(hdr))
    for b in backends:
        r = report.seeds[-1].results[b]
        if r.skipped:
            print(f"{b:<12}{'skipped':<12}")
            continue
        m1, s1 = _mean_std(report, b, 1)
        m3, s3 = _mean_std(report, b, 3)
        m5, s5 = _mean_std(report, b, 5)
        m10, s10 = report.aggregate(b, 10)
        mapm = _mean_map(report, b)

        def fmt(m, s):
            return f"{m:.3f}±{s:.3f}"

        print(
            f"{b:<12}{fmt(m1, s1):<14}{fmt(m3, s3):<14}{fmt(m5, s5):<14}{fmt(m10, s10):<16}{mapm:.3f}"
        )


def _mean_std(report, b, k):
    vals = [sr.results[b].ndcg.get(k, 0.0) for sr in report.seeds if not sr.results[b].skipped]
    import numpy as np

    a = np.asarray(vals)
    return float(a.mean()), float(a.std())


def _mean_map(report, b):
    import numpy as np

    vals = [sr.results[b].map for sr in report.seeds if not sr.results[b].skipped]
    return float(np.asarray(vals).mean())


def run_full_demo(cfg: RankForgeConfig, out_path: str = "benchmark.json", seeds=None, backends=None) -> dict:
    pipe = RankPipeline(cfg)
    pipe.config.validate()
    print("== RankForge demo ==")
    print(f"config: n_queries={cfg.n_queries} docs/q={cfg.docs_per_query} feat={cfg.n_features} snr={cfg.relevance_snr}")
    print("-- benchmark (multi-seed) --")
    report = pipe.benchmark(backends=backends, seeds=seeds)
    _print_table(report, cfg)
    win = _win_check(report, cfg)
    print(f"\nwin over pointwise: {win}")

    print("-- ablation --")
    ablation = pipe.ablation(seed=cfg.seed)
    for k, v in ablation.items():
        print(f"  {k}: {v}")

    print("-- failure cases --")
    failures = pipe.failure_cases(seed=cfg.seed)
    for k, v in failures.items():
        print(f"  {k}: {v}")

    print("-- determinism --")
    det = pipe.determinism_check(seed=cfg.seed, backends=backends)
    for k, v in det.items():
        print(f"  {k}: {v}")

    out = {
        "system": "RankForge",
        "version": "0.1.0",
        "author": "晨星",
        "config": cfg.to_dict(),
        "seeds": [sr.to_dict() for sr in report.seeds],
        "aggregate_ndcg10": {
            b: {"mean": report.aggregate(b, 10)[0], "std": report.aggregate(b, 10)[1]}
            for b in report.seeds[0].results
        },
        "win_over_pointwise": win,
        "ablation": ablation,
        "failure_cases": failures,
        "determinism": det,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nbenchmark.json written -> {out_path}")
    return out


if __name__ == "__main__":
    from ..core.config import RankForgeConfig as _C

    c = _C.from_env()
    run_full_demo(c, out_path="benchmark.json")
    sys.exit(0)
