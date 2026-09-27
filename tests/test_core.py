import numpy as np

from rankforge.core.seed import set_all
from rankforge.data.loader import split_by_query
from rankforge.data.synth import make_dataset
from rankforge.preprocess.standardize import Standardizer


def test_set_all_deterministic():
    set_all(7)
    a = np.random.randn(5)
    set_all(7)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_synth_shape_and_range():
    ds = make_dataset(seed=1, n_queries=10, docs_min=5, docs_max=8, n_features=8, label_noise=0.3)
    assert ds.n_queries() == 10
    for q in ds.queries:
        assert q.features_matrix().shape[1] == 8
        rv = q.relevance_vector()
        assert rv.min() >= 0 and rv.max() <= 4


def test_synth_reproducible():
    a = make_dataset(seed=3, n_queries=10)
    b = make_dataset(seed=3, n_queries=10)
    assert np.array_equal(a.features_matrix(), b.features_matrix())


def test_standardize_fit_train_only():
    ds = make_dataset(seed=2, n_queries=30, docs_min=6, docs_max=10, n_features=8)
    sp = split_by_query(ds, seed=2)
    scaler = Standardizer().fit(sp.train)
    tr = scaler.transform(sp.train)
    X = tr.features_matrix()
    assert np.allclose(X.mean(0), 0, atol=1e-9)
    assert np.allclose(X.std(0), 1, atol=1e-6)


def test_leakage_guard_requires_fit():
    ds = make_dataset(seed=2, n_queries=30)
    scaler = Standardizer()
    raised = False
    try:
        scaler.transform(ds)
    except Exception:
        raised = True
    assert raised
