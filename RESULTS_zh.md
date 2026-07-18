# OptSQL 全量实验结果

## 实验设置

- 数据集：BIRD dev，全量 1,534 条。
- 模型：DashScope `qwen3-coder-plus` API。
- Embedding：本地 `Qwen/Qwen3-Embedding-0.6B`。
- 后处理：Meta-Controller 与 Optimization Phase。

## 核心指标

| 指标 | Generation | Controller Final | 变化 |
| --- | ---: | ---: | ---: |
| EX（严格离线复评） | 72.19% | 72.19% | 0.00 pp |
| VES（本机 5 次重复计时） | 76.607 | 76.614 | +0.007 |
| CR（本地 SQLite 估计） | 1.00578 | 1.00581 | +0.00004 |
| AR@0.8 | 69.91% | 69.91% | 0.00 pp |
| AR@1 | 69.39% | 69.39% | 0.00 pp |
| AR@1 / EX | 96.11% | 96.11% | 0.00 pp |

首次评测输出的 EX 为 72.28%。严格离线复评时有 2 条 gold SQL
执行超时，因此按 1,532 条有效 gold 计算为 1,106/1,532 = 72.19%。

## Controller 统计

| 状态 | 数量 | 占全量比例 |
| --- | ---: | ---: |
| `planning_only` | 341 | 22.23% |
| 进入 `planning_plus_optimization` | 1,193 | 77.77% |
| `optimization_skipped` | 967 | 63.04% |
| `optimization_rejected` | 93 | 6.06% |
| `optimization_validation_reflection_failed` | 91 | 5.93% |
| `optimization_skipped_by_explain` | 24 | 1.56% |
| `optimization_converged` | 18 | 1.17% |

Controller 共触发 1,193 条，最终接受 18 条改写。18 条改写没有改变
EX；本机重复计时中有 9 条加速、1 条变慢，其余变化不明显。

## Token 用量

- Prompt tokens：140,554,036
- Completion tokens：31,105,581
- Total tokens：171,659,617

## 说明

- EX 使用本地 SQLite 执行结果集合比较。
- VES 受 CPU、缓存、并发和重复次数影响，适合本地前后对照。
- CR/AR 使用 SQLite `EXPLAIN QUERY PLAN` 与表行数估计。
- BIRD test 没有公开 gold SQL，正式 test 指标需由 BIRD 官方评测。
