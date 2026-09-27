import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rankforge.core.config import Config
from rankforge.core.seed import set_all
from rankforge.pipeline.pipeline import RankForgePipeline

set_all(42)
c = Config(n_queries=40, docs_min=8, docs_max=20, n_features=16, label_noise=0.5, n_seeds=2, seeds=(42, 123))
p = RankForgePipeline(c)
t0 = time.perf_counter()
res = p.benchmark([42, 123])
print("elapsed", round(time.perf_counter() - t0, 1), "s")
for name, b in res["summary"].items():
    if b.get("available"):
        print(f"{name.ljust(16)} ndcg@10={b['ndcg@10']['mean']:.4f}  ndcg@5={b['ndcg@5']['mean']:.4f}  map={b['map']['mean']:.4f}")
