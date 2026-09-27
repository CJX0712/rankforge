"""HPO：Optuna 调参 LightGBM lambdamart。"""
from __future__ import annotations

from ..core.types import LTRDataset
from ..ltr.lightgbm_ranker import LgbmRanker
from ..ltr.metrics import ndcg_at_k


def tune_lambdamart(
    train: LTRDataset,
    val: LTRDataset,
    seed: int = 9042,
    n_trials: int = 20,
    timeout: int = 30,
) -> dict | None:
    """在独立 val 集上优化 lambdamart 超参，返回 best_params 或 None（Optuna 缺失）。"""
    try:
        import optuna
    except ImportError:
        return None

    def objective(trial):
        num_leaves = int(trial.suggest_int("num_leaves", 16, 63))
        lr = float(trial.suggest_float("learning_rate", 0.01, 0.2, log=True))
        l2 = float(trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True))
        m = LgbmRanker(
            seed=seed, n_estimators=120, num_leaves=num_leaves,
            learning_rate=lr, lambda_l2=l2,
        )
        m.fit(train, val)
        _, y, g = val.Xy_group()
        return ndcg_at_k(y, m.predict(val), g, 10)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return dict(study.best_params)
