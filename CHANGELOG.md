# Changelog

## v0.1.0 (2026-09-27)

- 初始交付：模块化 Learning to Rank 系统 RankForge。
- 后端：LightGBM `LGBMRanker(lambdarank)`、XGBoost `XGBRanker(rank:ndcg)`、
  纯 numpy RankNet（线性/非线性两模式）、Pointwise/Random/Heuristic 基线。
- 指标：手写 NDCG@{1,3,5,10} + MAP。
- 评测：3 seed 多 seed 均值±std；SOTA(lambdamart) NDCG@10 = 0.928，相对 pointwise 提升 +7.5%。
- 工程：单测 21 项全绿、ruff 检查、CI workflow、确定性逐位一致、离线兜底路径。
- 质量等级：**S**。
- 作者：晨星。
