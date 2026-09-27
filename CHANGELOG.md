# Changelog

## v0.1.0 (2026-09-27)

- 初始发布：模块化学习排序（Learning-to-Rank）系统 **RankForge**。
- **SOTA 后端（Tier-0）**：LambdaMART（lightgbm `lambdarank`）、XGBRanker（xgboost `rank:ndcg`）。
- **离线兜底（Tier-1，纯 numpy / sklearn）**：RankNet（pairwise）、ListNet（listwise）、
  PointwiseLogistic（sklearn，缺包降级纯 numpy）。
- **朴素基线**：Random / Constant。
- 统一"分数越大越靠前"契约；按 query 分组切分（防泄漏）；全局确定性 `set_all`。
- 量化基准 `benchmark.json`：NDCG@k / MAP / P@5 / MRR，3 seeds mean±std，确定性双跑校验，
  消融（列表式 vs 点式目标），失败案例（≥3 含归因）。
- 单测 21 项全绿；demo 端到端 45s（CPU，≤60s 预算）；质量等级 **S**。
- 文档：README、architecture.md、model_card.md。
- 作者：晨星。许可证：MIT。
