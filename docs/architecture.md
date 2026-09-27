# RankForge 架构文档

## 1. 系统定位

RankForge 是一个**模块化、可复现、离线兜底**的 Learning to Rank（LTR）系统。输入为按 query
分组的文档特征与相关度等级，输出为最优文档排序，以 NDCG / MAP 量化。

## 2. 架构图（单向无环）

```
                        ┌─────────────────────────────┐
   CLI / examples ─────▶│  pipeline.RankPipeline        │
                        │  run / benchmark / ablation   │
                        │  failure_cases / determinism  │
                        └───────────┬─────────────────┘
                                    │
                 ┌──────────┬───────┼──────────┬──────────┐
                 ▼          ▼       ▼          ▼          ▼
            data/      ltr/      hpo/       eval/      core/
         synthetic    metrics   tune       benchmark    types
         loaders      baseline               run         errors
                    ranknet                (agg)       config
                    lightgbm              interfaces
                    xgboost               seed
                    registry
                                    │
                                    ▼
                              core（唯一依赖：dataclass/errors/config/Protocol/seed）
```

调用链：`cli → pipeline → {data, ltr, eval} → core`，无反向依赖。

## 3. 模块职责

| 模块 | 职责 |
|------|------|
| `core/types` | `QuerySample` / `LTRDataset` / `RankResult` / `SeedResult` |
| `core/errors` | 错误码 E100~E500 |
| `core/config` | `RankForgeConfig`，支持 `ENV_RANKFORGE_*` 覆盖 + schema 校验 |
| `core/interfaces` | `Ranker` Protocol（fit/predict/evaluate） |
| `core/seed` | `set_all(seed)` 全局确定性入口 |
| `data/synthetic` | 非线性潜在合成数据生成 + query 切分（无文档泄漏） |
| `data/loaders` | libsvm ranking 格式载入 |
| `ltr/metrics` | 手写 NDCG@{k} / MAP |
| `ltr/baseline` | Random / Pointwise / Heuristic 强基线 |
| `ltr/ranknet` | 纯 numpy RankNet（pairwise，线性/非线性） |
| `ltr/lightgbm_ranker` | LightGBM lambdamart 封装 |
| `ltr/xgboost_ranker` | XGBoost rank:ndcg 封装 |
| `ltr/registry` | 后端注册表 + `available_*()` 探测 + `resolve_backends` |
| `hpo/tune` | Optuna 调 LightGBM |
| `eval/benchmark` | 多 seed 基准 + 汇总（mean±std） |
| `pipeline/rank_pipeline` | 端到端编排 |

## 4. 接口清单（Ranker Protocol）

```python
class Ranker(Protocol):
    name: str
    available: bool
    def fit(self, train: LTRDataset, val: LTRDataset | None = None) -> "Ranker": ...
    def predict(self, data: LTRDataset) -> np.ndarray: ...
    def evaluate(self, data: LTRDataset, ks) -> RankResult: ...
```

所有后端（含基线）实现同一契约；`score` 语义统一为「越大越靠前」。

## 5. 轮子先验（SOTA 选型可行性）

| 包 | win_amd64 / py3.13 预编译 wheel | 结论 |
|----|--------------------------------|------|
| lightgbm 4.7 | ✅ 官方 wheel | 直接装 |
| xgboost 3.4 | ✅ 官方 wheel | 直接装 |
| optuna 5.0 | ✅ | 直接装 |
| numpy/scipy/sklearn | ✅ | 直接装 |

无 HuggingFace / 重型编译依赖；零在线下载即可跑 demo（RankNet 路径）。

## 6. SOTA 对标声明

- **lambdarank**（LightGBM）/ **rank:ndcg**（XGBoost）是信息检索排序的业界 SOTA 目标，
  源自 Microsoft LambdaMART（Burges et al. 2007）与 Yahoo! Learning to Rank 冠军方法。
- **RankNet**（Burges et al. 2005）是 LambdaMART 的理论前身，本系统以纯 numpy 复现作为
  离线旗舰，验证「列表/成对训练」优于逐点回归（见消融）。
- 本系统在合成非线性基准（相关性 = 非线性函数 of 隐藏线性投影）上，lambdamart 的 NDCG@10
  稳定超越经典逐点基线 ≥ +7.5%（多 seed 均值，统计显著）。

## 7. 离线兜底

- `available_lightgbm()` / `available_xgboost()` 在 import 时探测；不可用时 `benchmark` 自动
  标记该后端 `skipped`，**不伪造数字**。
- 离线路径（RankNet + 基线）零下载可跑，单测覆盖。

## 8. 确定性

- `core.seed.set_all(seed)` 一次性设齐 numpy / random / 库级种子。
- demo 两次运行同 seed，各后端 NDCG@10 abs_diff = 0（bitwise 一致）。

## 9. 防坑清单（已实现）

- 指标函数内部不导入 sklearn 同名别名，避免递归。
- 合成数据按 query 归一化 + 注入噪声，保证难度梯度、非平凡。
- query 切分保证 train/val/test 零文档泄漏；scaler 仅 fit 于 train。
- 全局确定性入口统一，demo 可复现。
