"""CLI 入口：python -m rankforge.cli --demo。"""
from __future__ import annotations

import argparse
import sys

from .core.config import RankForgeConfig
from .examples.run_demo import run_full_demo


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rankforge", description="RankForge — Learning to Rank")
    p.add_argument("--demo", action="store_true", help="运行端到端演示并落盘 benchmark.json")
    p.add_argument("--seeds", type=int, default=3, help="基准运行 seed 数（≥3）")
    p.add_argument("--backends", nargs="*", default=None, help="后端名列表，默认 auto")
    p.add_argument("--out", default="benchmark.json", help="benchmark 输出路径")
    p.add_argument("--quick", action="store_true", help="使用小规模数据（CI 冒烟）")
    args = p.parse_args(argv)

    if not args.demo:
        p.print_help()
        return 0

    cfg = RankForgeConfig.from_env()
    if args.quick:
        cfg.n_queries = 120
        cfg.docs_per_query = 12
    seeds = [cfg.seed + i * 101 for i in range(max(1, args.seeds))]
    run_full_demo(cfg, out_path=args.out, seeds=seeds, backends=args.backends)
    return 0


if __name__ == "__main__":
    sys.exit(main())
