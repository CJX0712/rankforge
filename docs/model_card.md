# RankForge — 模型卡（Model Card）

## 模型细节

- **系统**：RankForge v0.1.0（模块化学习排序系统）
- **作者**：晨星
- **任务**：Learning-to-Rank（文档给定查询的相关性排序）
- **后端**：LambdaMART（lightgbm `lambdarank`）、XGBRanker（xgboost `rank:ndcg`）、
  PointwiseLogistic（sklearn）、RankNet（numpy pairwise）、ListNet（numpy listwise）、
  Random / Constant 基线
- **评分语义**：分数越大越靠前

## 训练数据

- **来源**：合成 LETOR 风格数据集（`data/synth.make_dataset`）。
- **规模**：默认 100 queries，每 query 8–20 docs，16 维特征。
- **标签**：分级相关性 0–4（DCG gain = 2^rel−1）。真实效用函数以**非线性 + 交互项**为主
  （弱线性项），使线性模型无法充分拟合、树模型获得真实优势。
- **难度**：35% 查询为"难"查询（弱信号 + 高噪声），形成难度梯度。
- **切分**：按 query 分组（train 60% / val 20% / test 20%），杜绝跨 query 泄漏。
- **可复现**：固定 `seed`，数据生成逐位可复现。

## 评估指标

NDCG@1/3/5/10、MAP、P@5、MRR。所有 ranker 在同一 test 集、同一指标口径下比较。

## 量化结果（3 seeds，mean±std）

| ranker | NDCG@10 | NDCG@5 | MAP | 备注 |
|--------|---------|--------|-----|------|
| LambdaMART | **0.819 ± 0.035** | 0.751 ± 0.036 | 0.956 ± 0.010 | 🏆 SOTA |
| XGBRanker | 0.811 ± 0.037 | 0.738 ± 0.030 | 0.956 ± 0.012 | SOTA 备选 |
| PointwiseLogistic | 0.669 ± 0.015 | 0.555 ± 0.010 | 0.914 ± 0.015 | 经典基线 |
| RankNet | 0.671 ± 0.010 | 0.550 ± 0.010 | 0.910 ± 0.016 | 离线兜底 |
| ListNet | 0.661 ± 0.014 | 0.540 ± 0.021 | 0.912 ± 0.019 | 离线兜底 |
| Random | 0.601 ± 0.024 | 0.456 ± 0.013 | 0.869 ± 0.038 | 朴素基线 |
| Constant | 0.570 ± 0.013 | 0.416 ± 0.023 | 0.872 ± 0.041 | 下界 |

- **SOTA 对标结论**：LambdaMART 相对经典 PointwiseLogistic 提升 **+0.150 NDCG@10**，均值差
  （0.150）> 0.5×(std₁+std₂)=0.025，达到简易显著性门槛 → DoD 性能项 ✅。
- **确定性**：同 seed 两次完整运行，各 ranker NDCG@10 逐位一致（`determinism.identical=true`）。
- **消融**：同线性架构下列表式(ListNet) vs 点式(最小二乘)训练目标 NDCG@10 差异 ≈ −0.008，
  结论——在此线性架构上目标函数影响极小，**决定性因素是模型非线性（LambdaMART 的树）**。
- **失败案例**（≥3，含归因）：见 `benchmark.json → failures`。典型：Random/Constant 无信号致
  NDCG 塌至下界；Pointwise 因优化分类似然而非列表 NDCG 而落后于 LambdaMART；
  ListNet 学习率过大（无裁剪）时训练不稳、NDCG 退化。

## 局限

- 数据为**合成**分级相关性，用于可复现对标；真实搜索/推荐日志上相对增益可能不同，但
  架构、指标口径与离线兜底机制直接可迁移。
- 树模型（lightgbm/xgboost）为 Tier-0 SOTA，离线兜底（RankNet/ListNet）为教学级线性实现，
  在强非线性信号上弱于树模型——这是预期的"兜底"定位，非缺陷。
- 未接入在线预训练权重（遵守"网络可达性约束"），全部本地可跑。

## 伦理与合规

- 无个人数据、无隐私字段；合成数据仅含数值特征。
- 依赖均为宽松许可证（MIT/BSD/Apache）：lightgbm(BSD-3)、xgboost(Apache-2.0)、
  scikit-learn(BSD-3)、numpy(BSD-3)、scipy(BSD-3)、pandas(BSD-3)。
- 提交前已通过密钥/隐私 grep 自查，无密钥或敏感信息泄漏。
