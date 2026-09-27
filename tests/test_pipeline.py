"""pipeline 单测：run / 确定性 / 离线兜底路径。"""

from rankforge.core.config import RankForgeConfig
from rankforge.pipeline.rank_pipeline import RankPipeline


def _cfg():
    return RankForgeConfig(n_queries=60, docs_per_query=10, n_features=6, seed=11)


def test_pipeline_run_single_seed():
    pipe = RankPipeline(_cfg())
    rep = pipe.run(seed=11, backends=["ranknet", "pointwise", "random"])
    assert len(rep.seeds) == 1
    assert "ranknet" in rep.seeds[0].results
    assert not rep.seeds[0].results["ranknet"].skipped


def test_pipeline_determinism_ranknet_bitwise():
    pipe = RankPipeline(_cfg())
    det = pipe.determinism_check(seed=11, backends=["ranknet"])
    assert det["ranknet"]["bitwise"] is True


def test_offline_fallback_runs_without_sota():
    # 强制只用纯 numpy 后端（离线兜底），不依赖 lightgbm/xgboost
    pipe = RankPipeline(_cfg())
    rep = pipe.run(seed=11, backends=["ranknet", "pointwise", "random", "heuristic"])
    for b in ("ranknet", "pointwise", "random", "heuristic"):
        assert not rep.seeds[0].results[b].skipped
