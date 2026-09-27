"""core 层单测。"""
from rankforge.core.config import RankForgeConfig
from rankforge.core.errors import ConfigError
from rankforge.core.seed import set_all
from rankforge.data.synthetic import generate_ltr_dataset, split_dataset


def test_config_validate_ok():
    cfg = RankForgeConfig()
    cfg.validate()  # 不应抛异常


def test_config_validate_bad():
    cfg = RankForgeConfig(test_size=0.8, val_size=0.8)
    try:
        cfg.validate()
        assert False, "expected ConfigError"
    except ConfigError:
        pass


def test_generate_and_split_sizes():
    set_all(1)
    ds = generate_ltr_dataset(seed=1, n_queries=60, docs_per_query=10, n_features=6)
    assert ds.n_queries == 60
    assert ds.n_docs == 600
    train, val, test = split_dataset(ds, 0.2, 0.2, seed=1)
    assert train.n_queries + val.n_queries + test.n_queries == 60
    # query 间独立，无文档跨集泄漏：总文档数守恒
    assert train.n_docs + val.n_docs + test.n_docs == ds.n_docs


def test_determinism_generate():
    a = generate_ltr_dataset(seed=7, n_queries=20, docs_per_query=8, n_features=5)
    b = generate_ltr_dataset(seed=7, n_queries=20, docs_per_query=8, n_features=5)
    import numpy as np

    assert np.array_equal(a.queries[0].X, b.queries[0].X)
    assert np.array_equal(a.queries[0].y, b.queries[0].y)
