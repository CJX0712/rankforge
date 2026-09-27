from rankforge.core.config import Config
from rankforge.pipeline.pipeline import RankForgePipeline


def _pipe():
    c = Config(
        n_queries=20,
        docs_min=6,
        docs_max=10,
        n_features=10,
        label_noise=0.4,
        n_seeds=2,
        seeds=(11, 22),
    )
    return RankForgePipeline(c)


def test_single_seed_rows_present():
    p = _pipe()
    res = p.run_single_seed(11)
    names = [r.ranker for r in res["rows"]]
    assert "LambdaMART" in names and "ListNet" in names and "PointwiseLogistic" in names


def test_determinism_two_runs_identical():
    p = _pipe()
    r1 = p.benchmark([11, 22])["summary"]
    r2 = p.benchmark([11, 22])["summary"]
    for n in r1:
        if r1[n].get("available"):
            assert abs(r1[n]["ndcg@10"]["mean"] - r2[n]["ndcg@10"]["mean"]) < 1e-9


def test_ablation_runs():
    p = _pipe()
    ab = p.ablation_listwise_vs_pointwise(11)
    assert "delta" in ab


def test_split_is_leakage_free():
    p = _pipe()
    res = p.run_single_seed(11)
    sp = res["split"]
    tr = {q.query_id for q in sp.train.queries}
    va = {q.query_id for q in sp.val.queries}
    te = {q.query_id for q in sp.test.queries}
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)


def test_benchmark_aggregates_seeds():
    p = _pipe()
    res = p.benchmark([11, 22])
    assert res["summary"]["LambdaMART"]["n_seeds"] == 2
