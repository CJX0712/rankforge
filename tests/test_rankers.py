import numpy as np

from rankforge.data.loader import split_by_query
from rankforge.data.synth import make_dataset
from rankforge.eval.benchmark import evaluate_ranker
from rankforge.rankers.lambdamart import LambdaMARTRanker
from rankforge.rankers.listwise import ListNetRanker
from rankforge.rankers.pairwise import RankNetRanker
from rankforge.rankers.registry import build_rankers


def _tiny():
    ds = make_dataset(seed=5, n_queries=20, docs_min=6, docs_max=10, n_features=10)
    return split_by_query(ds, seed=5)


def test_all_rankers_run_and_finite():
    sp = _tiny()
    for r in build_rankers(5):
        if not r.is_available():
            continue
        row, m = evaluate_ranker(r, sp.train, sp.test)
        assert row.available
        assert 0.0 <= m["ndcg@10"] <= 1.0 + 1e-9
        res = r.predict_scores(sp.test)
        for v in res.scores.values():
            assert np.all(np.isfinite(v))


def test_sota_backends_available():
    assert LambdaMARTRanker(seed=1, backend="lightgbm").is_available()
    assert LambdaMARTRanker(seed=1, backend="xgboost").is_available()


def test_numpy_fallbacks_run():
    sp = _tiny()
    for r in [ListNetRanker(seed=5), RankNetRanker(seed=5)]:
        row, m = evaluate_ranker(r, sp.train, sp.test)
        assert row.available
        assert np.isfinite(m["ndcg@10"])


def test_unavailable_ranker_reports_skipped():
    from rankforge.core.types import RankingDataset
    from rankforge.rankers.base import BaseRanker

    class Broken(BaseRanker):
        name = "Broken"

        def is_available(self):
            return False

        def _fit(self, ds):
            pass

        def _scores(self, ds):
            return {}

    row, m = evaluate_ranker(Broken(seed=1), _tiny().train, _tiny().test)
    assert not row.available
    assert m == {}


def test_listnet_beats_random_on_average():
    from rankforge.core.config import Config
    from rankforge.pipeline.pipeline import RankForgePipeline

    # Use a small multi-seed benchmark: the offline linear fallback must beat naive
    # ranking in expectation (variance on a single tiny seed can otherwise mislead).
    cfg = Config(n_queries=40, n_features=16, n_seeds=3, seeds=(11, 22, 33))
    pipe = RankForgePipeline(cfg)
    res = pipe.benchmark([11, 22, 33])
    list_mean = res["summary"]["ListNet"]["ndcg@10"]["mean"]
    rand_mean = res["summary"]["Random"]["ndcg@10"]["mean"]
    assert list_mean >= rand_mean - 1e-3
