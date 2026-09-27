"""RankNet 离线旗舰单测：可用 / 可跑 / 确定性。"""
import numpy as np

from rankforge.core.seed import set_all
from rankforge.data.synthetic import generate_ltr_dataset, split_dataset
from rankforge.ltr.ranknet import RankNet


def _tiny_ds():
    set_all(3)
    ds = generate_ltr_dataset(seed=3, n_queries=40, docs_per_query=10, n_features=6)
    return ds


def test_ranknet_available_and_runs():
    ds = _tiny_ds()
    train, _, test = split_dataset(ds, 0.2, 0.2, seed=3)
    m = RankNet(seed=3, epochs=10)
    m.fit(train)
    res = m.evaluate(test, (1, 3, 5, 10))
    assert not res.skipped
    assert 0.0 <= res.ndcg[10] <= 1.0


def test_ranknet_deterministic():
    ds = _tiny_ds()
    train, _, test = split_dataset(ds, 0.2, 0.2, seed=3)
    m1 = RankNet(seed=3, epochs=10).fit(train)
    m2 = RankNet(seed=3, epochs=10).fit(train)
    s1 = m1.predict(test)
    s2 = m2.predict(test)
    assert np.array_equal(s1, s2), "同 seed 两次训练应逐位一致"


def test_ranknet_linear_mode_runs():
    ds = _tiny_ds()
    train, _, test = split_dataset(ds, 0.2, 0.2, seed=3)
    m = RankNet(linear=True, seed=3, epochs=10).fit(train)
    res = m.evaluate(test, (10,))
    assert not res.skipped
