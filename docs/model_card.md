# Model Card — RankForge

## 模型详情

- **名称**：RankForge
- **类型**：Learning to Rank（列表/成对排序学习）
- **后端**：LightGBM lambdamart（默认 SOTA）、XGBoost rank:ndcg、纯 numpy RankNet（离线旗舰）
- **作者**：晨星 · 版本：0.1.0

## 用途（Intended Use）

将「同一 query 下的若干文档」按相关度等级排序，使最相关文档排在前列。适用于搜索召回重排、
推荐列表排序、问答候选排序等需优化 NDCG/MAP 的场景。

## 训练数据

- **主数据**：合成 LTR 数据集（默认 200 query × 15 doc × 10 feat）。
- **生成方式**：相关性潜在 = 非线性函数 of 隐藏线性投影 `w·X`
  （`latent = (w·X) + 1.2·(w·X)² − 0.3·sin(2·w·X)`）+ 每 query 偏置 + 高斯噪声（SNR=3.0），
  按 query 归一化到等级 0..4。
- **切分**：按 query 随机切分 train/val/test（无文档跨集泄漏）。
- **真实数据**：支持 libsvm ranking 格式（`data/loaders.load_libsvm`）。

## 指标（真实运行 · 3 seed · NDCG@10）

| 后端 | NDCG@10 | 说明 |
|------|---------|------|
| lambdamart | 0.928 ± 0.004 | 最佳 SOTA |
| xgboost | 0.901 ± 0.013 | SOTA |
| ranknet | 0.845 ± 0.014 | 离线旗舰（零依赖） |
| pointwise | 0.863 ± 0.014 | 强基线 |
| random | 0.389 ± 0.020 | 下界 |
| heuristic | 0.424 ± 0.104 | 下界 |

胜强基线：lambdamart 相对 pointwise **+7.5%**（显著）。

## 局限性（Limitations）

1. **合成基准**：主量化在合成非线性数据上完成；真实搜索引擎数据（MSLR-WEB30K 等）需另测。
2. **RankNet 规模**：纯 numpy 实现适合中小数据；大规模训练建议用 LightGBM/XGBoost 后端。
3. **特征假设**：依赖文档级特征；需先有召回/特征工程链路。
4. **无在线下载**：默认不拉取预训练权重，全部从零训练。

## 伦理与风险

- 排序系统可能放大训练数据中的偏差；上线前应在真实业务数据上做公平性审计。
- 本系统仅提供排序能力，不对最终业务决策负责。

## 复现

```bash
python -m rankforge.examples.run_demo --out benchmark.json
```
固定 `RankForgeConfig.seed` 与 `requirements.lock.txt`，结果逐位可复现。
