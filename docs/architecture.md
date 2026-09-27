# RankForge — 架构文档

## 1. 设计原则

1. **单向无环**：`cli → pipeline → {data, preprocess, rankers, eval} → core`。上层只依赖下层，
   下层绝不反向引用上层，避免循环依赖与不可测的隐式耦合。
2. **统一语义契约**：所有 ranker 实现 `Ranker` 协议，`predict_scores` 返回
   `{query_id: score_vector}`，**分数越大越靠前**。跨模块评测因此公平可比。
3. **全局确定性**：唯一入口 `core.seed.set_all(seed)`，一次性设齐 numpy / python `random`（及
   可选 torch）。每个 ranker 用自身 `seed` 设 RNG，保证同 seed 两次运行逐位一致。
4. **离线兜底优先**：SOTA 后端（需网络/重型编译）不可用时，`available_*()` 探测失败即跳过，
   benchmark 自动标注 `skipped`，系统仍以纯 numpy 路径完整跑通。
5. **防泄漏**：预处理/统计只在 train 折上 `fit`，eval 用独立 holdout；切分按 **query** 分组，
   杜绝跨 query 信息泄漏。

## 2. 模块职责

| 模块 | 职责 | 关键不变量 |
|------|------|-----------|
| `core.seed` | 全局确定性播种 | 同 seed → 同 RNG 序列 |
| `core.errors` | E100~E500 错误码 | 每个错误带稳定机器码 |
| `core.config` | `ENV_RANKFORGE_*` 覆盖 + 校验 | 切分比例和 = 1，参数合法 |
| `core.types` | dataclass：Document/Query/RankingDataset/RankerResult/BenchmarkRow | — |
| `core.interfaces` | `Ranker` Protocol | `fit` / `predict_scores` / `is_available` |
| `data.synth` | 合成 LETOR 风格分级相关性数据 | 相关性 ∈ [0,4]；同 seed 可复现 |
| `data.loader` | 按 query 分组切分 + LETOR 文件载入 | train/val/test query 互不相交 |
| `preprocess.standardize` | 仅 train fit 的标准化 | 未 fit 即 transform 抛错（泄漏护栏） |
| `rankers.lambdamart` | lightgbm/xgboost 树排序 | `is_available()` 懒探测；缺失即跳过 |
| `rankers.pointwise` | sklearn 多分类逻辑回归（+numpy 兜底） | 分数 = 期望等级 |
| `rankers.pairwise` | RankNet（pairwise numpy） | 梯度裁剪（max_grad_norm） |
| `rankers.listwise` | ListNet（listwise numpy） | DCG-gain 列表目标 |
| `rankers.baselines` | Random / Constant 朴素基线 | 无训练 |
| `eval.metrics` | NDCG@k / MAP / P@5 / MRR | 生产实现 vs 独立参照实现交叉验证 |
| `eval.benchmark` | 单 ranker 评测 + 多 seed 聚合 | mean±std |
| `pipeline` | 编排：生成→切分→训练→评测→报告 | 确定性双跑校验 |
| `cli` | argparse 入口 | — |

## 3. 数据流

```
make_dataset(seed)                # 合成数据（非线性+交互信号，分级相关性）
   │
split_by_query(seed)              # 按 query 分组：train/val/test
   │
Standardizer().fit(train)         # 仅 train 拟合（防泄漏）
   │ transform → train/val/test
   │
for ranker in registry:           # LambdaMART, XGBRanker, Pointwise, RankNet, ListNet, Random, Constant
    ranker.fit(train)
    scores = ranker.predict_scores(test)
    metrics = evaluate_dataset(y_true, scores)   # NDCG@k / MAP / P@5 / MRR
   │
aggregate over seeds → mean±std
   │
build_report: 确定性双跑 + 消融 + 失败案例 + DoD 性能 → benchmark.json
```

## 4. 指标交叉验证

`eval/metrics.py` 中每个指标都有**独立参照实现**（如 `ndcg_at_k_reference` 用显式循环，
不与生产实现的向量化 `argsort` 共享代码路径）。单测 `test_metrics.py` 对 80+ 随机样本断言
两者一致（容差 1e-12），并对边界（完美排序 → NDCG=1、值 ∈ [0,1]）做断言。

## 5. 性能预算

端到端 demo（默认 100 queries × 3 seeds，含确定性双跑 + 消融 + 失败探针）在 CPU 上约 **45s**，
满足 ≤60s 预算。内存峰值远低于 2GB（纯 numpy / sklearn / 轻量树模型）。

## 6. 扩展指南

- 新增 ranker：继承 `rankers.base.BaseRanker`，实现 `_fit` / `_scores` / `is_available`，
  在 `rankers/registry.py` 的 `build_rankers` 注册即可被 benchmark 自动纳入。
- 接入真实数据：用 `data.loader.load_letor(path)` 载入标准 LETOR 格式，或实现新的
  `RankingDataset` 构造器。
- 新指标：在 `eval/metrics.py` 增加函数并补参照实现 + 单测。
