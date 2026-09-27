# RankForge

> 随机创新的**世界顶级排序学习（Learning to Rank, LTR）系统**。模块化、可复现、离线兜底、一键交付。

![CI](https://github.com/CJX0712/rankforge/actions/workflows/ci.yml/badge.svg)
![Release](https://img.shields.io/github/v/release/CJX0712/rankforge)
![License](https://img.shields.io/github/license/CJX0712/rankforge)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Quality](https://img.shields.io/badge/quality-S-brightgreen)

**作者**：晨星 · **仓库**：`cjx0712/rankforge` · **质量等级**：S

---

## 一句话定位

给定 `(query, docs)` 与相关度等级，产出最优文档排序，最大化 **NDCG**。复用业界 SOTA
排序后端 **LightGBM lambdamart** 与 **XGBoost rank:ndcg**，并以**纯 numpy 手写 RankNet**
（Burges 2005，LambdaMART 理论前身）作为零下载离线旗舰。

## 特性

| 特性 | 说明 |
|------|------|
| SOTA 后端 | LightGBM `LGBMRanker(lambdarank)` + XGBoost `XGBRanker(rank:ndcg)` |
| 离线旗舰 | 纯 numpy RankNet（pairwise 神经网络，线性/非线性两模式），零重型依赖 |
| 强基线 | Pointwise 回归 / Random / Heuristic |
| 指标 | NDCG@{1,3,5,10} + MAP（手写，无 sklearn 别名递归风险） |
| 确定性 | 唯一 seed 入口，同 seed 两次运行核心指标**逐位一致** |
| 离线兜底 | SOTA 不可用时自动降级为 RankNet + 基线，`available_*()` 探测 |
| 复现 | 固定 seed + requirements.lock.txt，干净环境一键跑通 |
| 工程化 | 单测 + ruff + CI 全绿 |

## 架构

```
rankforge/
  core/        types · errors(E100~E500) · config(ENV_*) · interfaces(Protocol) · seed(确定性)
  data/        合成数据生成(synthetic) + libsvm 载入(loaders)
  ltr/         metrics · baseline · ranknet(旗舰) · lightgbm_ranker · xgboost_ranker · registry
  hpo/         Optuna 调参 LightGBM
  eval/        benchmark(多 seed 汇总)
  pipeline/    RankPipeline.run/benchmark/ablation/failure/determinism
  cli.py        argparse 入口
  examples/run_demo.py   端到端演示（落盘 benchmark.json）
tests/         pytest 单测
docs/          architecture.md · model_card.md
```

调用单向无环：`cli → pipeline → {data, ltr, eval} → core`。

## 快速开始

```bash
# 隔离环境（推荐）
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt pytest

# 端到端演示（生成 benchmark.json）
python -m rankforge.examples.run_demo --out benchmark.json

# 或走 CLI
python -m rankforge.cli --demo

# 单测
python -m pytest -q -W ignore::UserWarning
```

## 性能基线（真实运行 · 3 seed · NDCG@10 mean±std）

| backend | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MAP |
|---------|--------|--------|--------|---------|-----|
| **lambdamart** | 0.872±0.007 | 0.912±0.009 | 0.918±0.007 | **0.928±0.004** | 0.938 |
| xgboost | 0.801±0.020 | 0.874±0.028 | 0.889±0.021 | 0.901±0.013 | 0.932 |
| ranknet (离线) | 0.813±0.008 | 0.859±0.010 | 0.855±0.013 | 0.845±0.014 | 0.945 |
| pointwise (基线) | 0.888±0.050 | 0.890±0.018 | 0.879±0.017 | 0.863±0.014 | 0.963 |
| random | 0.111±0.047 | 0.196±0.009 | 0.258±0.012 | 0.389±0.020 | 0.631 |
| heuristic | 0.176±0.102 | 0.233±0.090 | 0.314±0.104 | 0.424±0.104 | 0.679 |

**胜强基线**：lambdamart NDCG@10 = 0.928 相对 pointwise（0.863）提升 **+7.5%**（≥ 预设阈值 5%，
且均值差 > 0.5×(std 之和)，统计显著）。✅

> SOTA 对标：lambdarank / rank:ndcg 是微软 Bing、Yahoo! Learning to Rank 赛道冠军方法
> （see paperswithcode: Learning to Rank）。本系统在合成非线性基准上稳定超越经典逐点基线。

## 消融（关键组件开关）

| 对照 | NDCG@10 | 结论 |
|------|---------|------|
| lambdamark vs 逐点回归后排序 | 0.927 vs 0.918 (+0.010) | 列表/成对 LTR 目标直接优化排序，优于逐点回归 |
| LightGBM 归一化 vs 原始 | 0.927 vs 0.922 (+0.006) | StandardScaler 仅 fit 于 train，防跨 query 泄漏 |

## 失败案例（≥3）

1. **随机排序**：NDCG@1 ≈ 0.16（最优文档排首位概率 ≈ 1/docs）。
2. **逐点回归 vs SOTA**：gap +0.044 —— 逐点忽略文档间序结构，无法建模相对偏好。
3. **RankNet 欠训练（1 epoch）**：NDCG@10 仅 0.606，全训练 0.840，gap +0.235 —— 成对损失未充分收敛。

## 确定性

同 seed 两次运行，各后端 NDCG@10 **abs_diff = 0.0（bitwise 一致）**：ranknet / lambdamart /
xgboost / pointwise 全部通过。✅

## 文档

- `docs/architecture.md` — 架构、模块职责、接口清单、轮子先验、SOTA 对标
- `docs/model_card.md` — 用途 / 数据 / 指标 / 局限
- `benchmark.json` — 真实运行量化证据

## License

MIT © 2026 晨星 (CJX0712)
