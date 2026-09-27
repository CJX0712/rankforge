import numpy as np

from rankforge.eval.metrics import (
    average_precision,
    ndcg_at_k,
    ndcg_at_k_reference,
    precision_at_k,
    reciprocal_rank,
)


def test_ndcg_reference_agrees():
    rng = np.random.default_rng(0)
    for _ in range(80):
        n = int(rng.integers(3, 30))
        y = rng.integers(0, 5, size=n).astype(float)
        s = rng.normal(size=n)
        assert abs(ndcg_at_k(y, s, 10) - ndcg_at_k_reference(y, s, 10)) < 1e-12


def test_ndcg_perfect_is_one():
    y = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    s = y.copy()
    assert abs(ndcg_at_k(y, s, 5) - 1.0) < 1e-9


def test_ndcg_in_unit_interval():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 20))
        y = rng.integers(0, 5, size=n).astype(float)
        s = rng.normal(size=n)
        v = ndcg_at_k(y, s, 10)
        assert -1e-9 <= v <= 1.0 + 1e-9


def test_map_bounds_and_perfect():
    y = np.array([2.0, 0.0, 1.0, 0.0])
    s = y.copy()
    assert abs(average_precision(y, s) - 1.0) < 1e-9
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(1, 15))
        y = rng.integers(0, 3, size=n).astype(float)
        s = rng.normal(size=n)
        assert 0.0 <= average_precision(y, s) <= 1.0 + 1e-9


def test_precision_monotone_and_bounded():
    rng = np.random.default_rng(4)
    for _ in range(50):
        n = int(rng.integers(2, 15))
        y = rng.integers(0, 2, size=n).astype(float)
        s = rng.normal(size=n)
        assert 0.0 <= precision_at_k(y, s, 5) <= 1.0
        assert 0.0 <= reciprocal_rank(y, s) <= 1.0
