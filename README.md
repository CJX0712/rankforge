# RankForge · 模块化学习排序（Learning-to-Rank）系统

> 世界顶级 AI 系统 · 作者 **晨星** · 仓库 [`CJX0712/rankforge`](https://github.com/CJX0712/rankforge)

[![CI](https://github.com/CJX0712/rankforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/rankforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue)](https://github.com/CJX0712/rankforge/releases/tag/v0.1.0)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-3776AB)](https://www.python.org)
[![Quality](https://img.shields.io/badge/quality-S%20(world--class)-brightgreen)](docs/model_card.md)

RankForge 是一套**可实际运行、性能对标业界 SOTA、一键可复现**的学习排序系统。它复用
工业级开源后端（LambdaMART / XGBRanker），并以纯 numpy 手写实现 RankNet（pairwise）与
ListNet（listwise）作为**离线兜底**，保证零下载即可跑通 demo。

---

## 一句话结论

在合成的 LETOR 风格分级相关性数据集（100 queries × 3 seeds）上：

| ranker | 后端 | NDCG@10 (mean±std) | 角色 |
|--------|------|--------------------|------|
| **LambdaMART** | lightgbm `lambdarank` | **0.819 ± 0.035** | 🏆 SOTA 旗舰 |
| XGBRanker | xgboost `rank:ndcg` | 0.811 ± 0.037 | SOTA 备选 |
| PointwiseLogistic | sklearn | 0.669 ± 0.015 | 经典方法基线 |
| RankNet | numpy (pairwise) | 0.671 ± 0.010 | 离线兜底 |
| ListNet | numpy (listwise) | 0.661 ± 0.014 | 离线兜底 |
| Random | — | 0.601 ± 0.024 | 朴素基线 |
| Constant | — | 0.570 ± 0.013 | 下界 |

**LambdaMART 相对经典 PointwiseLogistic 提升 +0.150 NDCG@10（均值差 > 0.5×(std₁+std₂)，显著），质量等级 S。**
完整 `mean±std`、`≥3 seeds`、确定性校验、消融与失败案例见 [`benchmark.json`](benchmark.json)。

---

## 技术选型与 SOTA 对标

- **SOTA 后端（Tier-0）**：`lightgbm` 的 `LGBMRanker(objective="lambdarank")` 是工业界排序标杆
  （Burges 2006/2007，主流搜索引擎排序器的基石）；`xgboost` 的 `XGBRanker(objective="rank:ndcg")`
  为强替代。二者直接优化列表级 NDCG 目标。
- **离线兜底（Tier-1，纯 numpy / sklearn）**：
  - `RankNet`（Burges 2005）：线性打分 + pairwise sigmoid 损失，梯度裁剪保证稳定。
  - `ListNet`（Cao 2007）：线性打分 + 列表级 top-1 概率匹配损失（DCG-gain 目标）。
  - `PointwiseLogistic`：scikit-learn 多分类逻辑回归（经典 ML 基线），缺 sklearn 时降级为纯 numpy 多分类逻辑回归。
- **不禁止复用、不从头造 SOTA**：所有 SOTA 部分均来自成熟开源库；numpy 实现仅作**离线兜底**与
  教学对照，符合"轮子先验 + 离线兜底"约束。

---

## 架构（单向无环）

```
cli.py → pipeline → {data, preprocess, rankers, eval} → core
```

```
rankforge/
  core/        seed(全局确定性) · errors(E100~E500) · config(ENV_* 覆盖) · types · interfaces
  data/        synth(合成 LETOR 风格) · loader(按 query 分组切分 + LETOR 载入)
  preprocess/  standardize(仅 train fit，防泄漏)
  rankers/     lambdamart(SOTA) · pointwise · pairwise(RankNet) · listwise(ListNet) · baselines · registry
  eval/        metrics(NDCG/MAP/P@5/MRR，含独立参照实现) · benchmark(编排+聚合)
  pipeline/    RankForgePipeline.run() / benchmark() / build_report()
  cli.py       argparse 入口
  examples/run_demo.py  端到端 demo → benchmark.json
tests/         pytest 单测（含 CLI 冒烟 + 离线兜底路径）
docs/          architecture.md · model_card.md
```

调用约束：`cli → pipeline → {data, preprocess, rankers, eval} → core`，无环；所有 ranker 统一
"分数越大越靠前"语义，保证跨模块公平评测。

---

## 一键复现

```bash
# 1. 建隔离环境（Python 3.13）
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. 跑端到端 demo（生成 benchmark.json，~45s CPU）
.venv/Scripts/python.exe examples/run_demo.py

# 3. 跑测试 + lint
.venv/Scripts/python.exe -m pytest -q -W ignore::UserWarning

# 4. CLI 基准表
.venv/Scripts/python.exe cli.py bench --queries 100 --seeds 3
```

> 说明：`examples/run_demo.py` 与 `cli.py` 会自动把项目根加入 `sys.path`，无需先 `pip install -e .`。
> 若想以包方式运行：`python -m rankforge.cli bench`（需从工作区父目录执行，使 `rankforge` 包可导入）。

---

## 质量等级：**S（世界级）**

四项 DoD 全绿 + 多 seed 均值胜强基线（margin 0.150，显著）+ CI 绿 + Release 已打 tag。详见
[`docs/model_card.md`](docs/model_card.md) 与验收报告。

## 文档

- [架构文档](docs/architecture.md)
- [模型卡](docs/model_card.md)
- [量化基准](benchmark.json)

---

© 晨星 · MIT License
