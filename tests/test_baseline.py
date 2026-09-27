"""基线单测：Random / Pointwise / Heuristic。"""
from rankforge.core.seed import set_all
from rankforge.data.synthetic import generate_ltr_dataset, split_dataset
from rankforge.ltr.baseline import HeuristicRanker, PointwiseRanker, RandomRanker


def _ds():
    set_all(5)
    return generate_ltr_dataset(seed=5, n_queries=40, docs_per_query=10, n_features=6)


def test_baselines_run():
    ds = _ds()
    train, _, test = split_dataset(ds, 0.2, 0.2, seed=5)
    for cls in (RandomRanker, PointwiseRanker, HeuristicRanker):
        m = cls(seed=5)
        if cls is not RandomRanker and cls is not HeuristicRanker:
            m.fit(train)
        res = m.evaluate(test, (1, 3, 5, 10))
        assert not res.skipped
        assert 0.0 <= res.ndcg[10] <= 1.0


def test_random_nondeterministic_across_seed():
    ds = _ds()
    _, _, test = split_dataset(ds, 0.2, 0.2, seed=5)
    r1 = RandomRanker(seed=1).predict(test)
    r2 = RandomRanker(seed=2).predict(test)
    assert (r1 != r2).any()
