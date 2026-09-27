"""examples/run_demo.py — end-to-end RankForge demo.

Generates a synthetic LETOR-style dataset, runs every available ranker across
``n_seeds`` seeds, validates determinism by re-running, computes an ablation and
failure analysis, and writes ``benchmark.json`` (the quantified deliverable).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows console safety
# Insert the workspace (parent of the ``rankforge`` package) so the package is
# importable as ``rankforge.*`` and intra-package relative imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rankforge.core.config import Config
from rankforge.core.seed import set_all
from rankforge.pipeline.pipeline import RankForgePipeline

COLS = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "map", "p@5", "mrr"]


def _print_table(summary: dict) -> None:
    hdr = "ranker".ljust(16) + "".join(f"{c:>9}" for c in COLS)
    print(hdr)
    print("-" * len(hdr))
    for name, block in summary.items():
        if not block.get("available"):
            print(f"{name.ljust(16)}{'skipped':>9}{'':>{9*len(COLS)-9}}")
            continue
        cells = "".join(f"{block[k]['mean']:>9.4f}" for k in COLS)
        print(f"{name.ljust(16)}{cells}")


def main() -> dict:
    set_all(42)
    config = Config.from_env()
    config.validate()
    pipe = RankForgePipeline(config)
    seeds = list(config.seeds)[: config.n_seeds]

    t0 = time.perf_counter()
    res = pipe.benchmark(seeds)
    doc = pipe.build_report(res, seeds)
    elapsed = time.perf_counter() - t0
    doc["demo_seconds"] = round(elapsed, 2)

    out = Path(__file__).resolve().parent.parent / "benchmark.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print("RankForge benchmark  (seeds=%s, queries=%d, feats=%d)" % (seeds, config.n_queries, config.n_features))
    _print_table(doc["summary"])
    print("\nDeterminism (run1 vs run2 NDCG@10 identical):")
    for name, d in doc["determinism"].items():
        if d.get("available"):
            print(f"  {name.ljust(16)} {'OK' if d['identical'] else 'DIFF'}")
    print(f"\nAblation (ListNet listwise vs LinearPointwise): {doc['ablation']['delta']:+.4f} NDCG@10")
    print(f"DoD performance met: {doc['dos_performance']['met']}  |  Quality grade: {doc['quality_grade']}")
    print(f"wrote {out}  ({out.stat().st_size} bytes, demo {doc['demo_seconds']}s)")
    return doc


if __name__ == "__main__":
    main()
