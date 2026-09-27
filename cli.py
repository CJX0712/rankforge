"""cli.py — argparse entry point for RankForge.

Usage:
    python -m rankforge.cli demo      # full benchmark -> benchmark.json + table
    python -m rankforge.cli bench     # print benchmark table only
    python -m rankforge.cli rank ...   # (reserved) rank a feature vector
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rankforge.core.config import Config
from rankforge.core.seed import set_all
from rankforge.pipeline.pipeline import RankForgePipeline

COLS = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "map", "p@5", "mrr"]


def _print_table(summary: dict) -> None:
    hdr = "ranker".ljust(16) + "".join(f"{c:>10}" for c in COLS) + "avail"
    print(hdr)
    print("-" * len(hdr))
    for name, block in summary.items():
        if not block.get("available"):
            print(f"{name.ljust(16)}{'skipped':>10}{'':>60}")
            continue
        cells = "".join(f"{block[k]['mean']:>10.4f}" for k in COLS)
        print(f"{name.ljust(16)}{cells}  ok")


def cmd_demo(args: argparse.Namespace) -> int:
    set_all(args.seed)
    config = Config.from_env()
    if args.queries:
        config.n_queries = args.queries
    if args.seeds:
        config.n_seeds = args.seeds
    config.validate()
    pipe = RankForgePipeline(config)
    seeds = list(config.seeds)[: config.n_seeds]
    res = pipe.benchmark(seeds)
    _print_table(res["summary"])
    out = Path(args.out)
    doc = pipe.build_report(res, seeds)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out} ({out.stat().st_size} bytes)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    return cmd_demo(args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rankforge", description="RankForge learning-to-rank CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("demo", help="run full benchmark + write benchmark.json")
    pd.add_argument("--seed", type=int, default=42)
    pd.add_argument("--queries", type=int, default=0)
    pd.add_argument("--seeds", type=int, default=0)
    pd.add_argument("--out", default="benchmark.json")
    pd.set_defaults(func=cmd_demo)

    pb = sub.add_parser("bench", help="print benchmark table")
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--queries", type=int, default=0)
    pb.add_argument("--seeds", type=int, default=0)
    pb.add_argument("--out", default="benchmark.json")
    pb.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
