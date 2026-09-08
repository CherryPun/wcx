# v2_data —— V2 一代工作区（自包含）

> 本目录是 v1 → v2 迭代的一代快照。可独立运行本代实验，不改动 v1 正式产物。

## 内容

| 文件 | 说明 |
|---|---|
| `stage_e2e.py` | 自 `02_main_recommendation/stage_e2e.py` 拷贝的主版端到端；路径已适配：数据读仓库根 `05_shared_data/`，产物写本代 `_rerun/` |
| `_rerun/` | 本代运行产物（最终 Top3 CSV、指标 JSON），不入库 |

## 运行

```bash
python stage_e2e.py
```

- 依赖数据（需先在仓库根 `05_shared_data/` 就绪，未就绪见根 README“数据来源与重建”）：
  `nodes_attr_filled.csv`、`outcomes_7d_named.csv`、`business_capacity_summary.csv`、`bw_daily_7d.csv`
- 依赖：pandas/numpy/scipy/scikit-learn/xgboost

## 数据截止登记

| 数据源 | 截止日 |
|---|---|
| outcomes 账（业务起账日） | 2026-08-28 |
| 近 7 天带宽 bw_daily_7d | 2026-09-02 |
| 结算宽表 node_day_ops_wide_full | 2026-09-06 |
| 控制面 node_analysis_data | 2026-09-07 |

> 若上游成本归因（节点-业务冲销/分摊）结构调整完成并回流宽表，请在本代启用前确认标签金额口径是否需同步（见 `../README_V2迭代方案.md` 挂起项 D1）。

## 本代改动点（相对 v1 主版）

- 拷贝并适配路径常量（`ROOT = HERE.parents[2]` 指向仓库根、`DATA = ROOT/05_shared_data`、产物写 `HERE/_rerun`）；
- 模型/候选逻辑与 v1 主版一致，暂未修改（V2 路线各阶段按上级目录方案推进）。
