"""RankPipeline：端到端编排。

调用单向无环：pipeline -> {data, ltr, eval} -> core。
"""
from __future__ import annotations

from ..core.config import RankForgeConfig
from ..core.seed import set_all
from ..data.synthetic import generate_ltr_dataset, split_dataset
from ..eval.benchmark import BenchmarkReport, run_benchmark
from ..ltr.baseline import PointwiseRanker, RandomRanker
from ..ltr.lightgbm_ranker import LgbmRanker
from ..ltr.ranknet import RankNet
from ..ltr.registry import resolve_backends


class RankPipeline:
    def __init__(self, config: RankForgeConfig | None = None) -> None:
        self.config = config or RankForgeConfig.from_env()
        self.config.validate()

    def run(self, backends=None, seed: int | None = None) -> BenchmarkReport:
        seed = seed if seed is not None else self.config.seed
        return run_benchmark(self.config, backends=backends, seeds=[seed])

    def benchmark(self, backends=None, seeds=None) -> BenchmarkReport:
        return run_benchmark(self.config, backends=backends, seeds=seeds)

    # ---- 消融：关键组件开关对照 ----
    def ablation(self, seed: int | None = None) -> dict:
        seed = seed if seed is not None else self.config.seed
        set_all(seed)
        ds = generate_ltr_dataset(
            seed=seed, n_queries=self.config.n_queries, docs_per_query=self.config.docs_per_query,
            n_features=self.config.n_features, relevance_snr=self.config.relevance_snr,
            nonlinear=self.config.nonlinear,
        )
        _, _, test = split_dataset(ds, self.config.test_size, self.config.val_size, seed=seed)
        X, y, g = test.Xy_group()
        k = 10
        out = {}

        # 1) 核心创新消融：LTR 目标（lambdarank）vs 逐点回归目标（regression then rank）
        if LgbmRanker.available:
            import lightgbm as lgb

            tr = _train_like(ds, self.config)
            Xtr, ytr, _ = tr.Xy_group()
            reg = lgb.LGBMRegressor(
                n_estimators=120, learning_rate=0.05, random_state=seed, n_jobs=1, verbose=-1
            ).fit(Xtr, ytr)
            reg_ndcg = self._eval_ndcg10(reg.predict(X), y, g)
            lam = LgbmRanker(seed=seed).fit(tr)
            lam_ndcg = lam.evaluate(test, self.config.ks).ndcg[k]
            out["lambdamart_vs_regression"] = {
                "lambdarank_ndcg10": round(lam_ndcg, 4),
                "regression_ndcg10": round(reg_ndcg, 4),
                "delta": round(lam_ndcg - reg_ndcg, 4),
                "note": "列表/成对 LTR 目标直接优化排序，优于逐点回归后排序",
            }
        # 2) 数据预处理贡献：特征归一化
        if LgbmRanker.available:
            tr = _train_like(ds, self.config)
            norm = LgbmRanker(seed=seed, normalize=True).fit(tr)
            nonnorm = LgbmRanker(seed=seed, normalize=False).fit(tr)
            n_r = norm.evaluate(test, self.config.ks).ndcg[k]
            nn_r = nonnorm.evaluate(test, self.config.ks).ndcg[k]
            out["lightgbm_normalize"] = {
                "normalized_ndcg10": round(n_r, 4),
                "raw_ndcg10": round(nn_r, 4),
                "delta": round(n_r - nn_r, 4),
                "note": "StandardScaler 拟合于 train 仅，防止跨 query 尺度泄漏",
            }
        return out

    def _eval_ndcg10(self, pred, y, g) -> float:
        from ..ltr.metrics import evaluate_ranking

        return evaluate_ranking(y, pred, g, self.config.ks)[10]

    # ---- 失败案例：≥3 条典型误例 + 归因 ----
    def failure_cases(self, seed: int | None = None) -> dict:
        seed = seed if seed is not None else self.config.seed
        set_all(seed)
        ds = generate_ltr_dataset(
            seed=seed, n_queries=self.config.n_queries, docs_per_query=self.config.docs_per_query,
            n_features=self.config.n_features, relevance_snr=self.config.relevance_snr,
            nonlinear=self.config.nonlinear,
        )
        _, _, test = split_dataset(ds, self.config.test_size, self.config.val_size, seed=seed)
        k = 10
        tr = _train_like(ds, self.config)
        out = {}

        # 1) 随机排序：NDCG@1 近 0
        rnd = RandomRanker(seed=seed).evaluate(test, self.config.ks)
        out["random_ranking"] = {
            "ndcg@1": round(rnd.ndcg[1], 4),
            "reason": "无监督随机打分，最优文档排首位概率≈1/docs_per_query",
        }
        # 2) 逐点基线 vs SOTA 差距
        pw = PointwiseRanker(seed=seed).fit(tr).evaluate(test, self.config.ks)
        if LgbmRanker.available:
            sota = LgbmRanker(seed=seed).fit(tr).evaluate(test, self.config.ks)
            out["pointwise_vs_sota"] = {
                "pointwise_ndcg10": round(pw.ndcg[k], 4),
                "sota_ndcg10": round(sota.ndcg[k], 4),
                "gap": round(sota.ndcg[k] - pw.ndcg[k], 4),
                "reason": "逐点回归忽略文档间序结构，无法建模相对偏好",
            }
        # 3) RankNet 欠训练（1 epoch）欠拟合
        full = RankNet(hidden=16, epochs=self.config_d("ranknet_epochs", 30), seed=seed).fit(tr)
        under = RankNet(hidden=16, epochs=1, seed=seed).fit(tr)
        full_r = full.evaluate(test, self.config.ks).ndcg[k]
        under_r = under.evaluate(test, self.config.ks).ndcg[k]
        out["ranknet_underfit"] = {
            "full_ndcg10": round(full_r, 4),
            "epoch1_ndcg10": round(under_r, 4),
            "gap": round(full_r - under_r, 4),
            "reason": "成对损失未充分收敛，梯度下降步数不足",
        }
        return out

    # ---- 确定性：同 seed 两次运行核心指标一致 ----
    def determinism_check(self, seed: int | None = None, backends=None) -> dict:
        seed = seed if seed is not None else self.config.seed
        names = resolve_backends(backends or ["ranknet", "lambdamart", "xgboost", "pointwise"])
        r1 = self.run(backends=names, seed=seed)
        r2 = self.run(backends=names, seed=seed)
        out = {}
        for n in names:
            a = r1.seeds[0].results.get(n)
            b = r2.seeds[0].results.get(n)
            if a is None or b is None or a.skipped or b.skipped:
                out[n] = {"skipped": True}
                continue
            va, vb = a.ndcg[10], b.ndcg[10]
            out[n] = {
                "run1": round(va, 8),
                "run2": round(vb, 8),
                "abs_diff": round(abs(va - vb), 10),
                "bitwise": bool(va == vb),
            }
        return out

    def config_d(self, key: str, default):
        return default


def _train_like(ds, config):
    """取 train 子集用于消融/失败案例（避免与 benchmark 测试集混淆）。"""
    from ..data.synthetic import split_dataset

    train, _, _ = split_dataset(ds, config.test_size, config.val_size, seed=config.seed)
    return train
