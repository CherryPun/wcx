# 账本 Schema v1（新）

**目的**：把"账本（唯一真相）"的粒度、字段、语义与不变量一次定死，避免后续返工（schema 晚改就要迁移）。
**适配器**：`wcx2实验代码（新）/账本适配（新）.py`（legacy 列名 → schema v1）。
**校验工具**：`wcx2实验代码（新）/账本契约检查（新）.py`；**固定样本**：`fixtures/ledger_sample_v1.csv`；**测试**：`test_ledger_contract.py`（6/6）。

## 1. 表定义：`ledger_node_business_day`（节点×业务×日）

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_day` | date | 自然日（业务发生日） |
| `node_id` | str | 节点 ID（与现网扫描/宽表口径一致） |
| `business` | int64 | **canonical** 业务 ID；七牛统一为 `10000280` |
| `build_bandwidth_mbps` | float > 0 | 建设带宽（分母唯一来源，单位 Mbps） |
| `cost_amount` | float | 该节点-该业务的成本归因额（矿主收益口径） |
| `revenue_amount` | float | 对应收入额 |
| `sample_weight` | float > 0 | 连续运行段权重 × 时效权重 |
| `active_business_count` | int ≥ 1 | 该节点当日活跃主流业务数 |
| `is_primary` | bool | 单业务节点日为 True；多业务节点日全为 False |
| `effective_support` | float ≥ 0 | 可选：有效支持度（分位/审计用） |

**派生量（不落库，避免双写）**：
```
miner_unit_income   = cost_amount / build_bandwidth_mbps
platform_unit_profit = (revenue_amount - cost_amount) / build_bandwidth_mbps
```

## 2. 聚合视图：`ledger_room_business_day`（机房×业务×日）

- 机房键 = `省 × 运营商 × 机房`（平台利润的**唯一**允许预测粒度，见 `成本归因与标签诊断（新）.md`）；
- 由节点级按机房键 + 业务 + 日聚合得到；**聚合必须守恒**（节点级求和 == 机房级）。

## 3. 不变量（契约检查强制）

| 编号 | 不变量 |
|---|---|
| I1 | 键唯一：`(sample_day, node_id, business)` |
| I2 | 必需列**无空值**（禁止依赖 pandas 静默跳过 NaN） |
| I3 | `build_bandwidth_mbps > 0`、`sample_weight > 0` |
| I4 | `business` 属于 allowlist（若提供） |
| I5 | `sample_day` 落在声明窗口内 |
| I6 | 单业务节点日必须 `is_primary=True`；每节点日至多一个 primary |
| I7 | 节点级求和 == 机房级聚合（守恒；由上游校验） |
| I8 | 总账：归因额合计不改变大盘（成本重分配必须和为 0，见 `成本归因与标签诊断（新）.md`） |

## 4. 多业务语义（必须明确，二选一）

| 模式 | 语义 | 用途 |
|---|---|---|
| **V5 兼容（一天一个干净业务）** | `active_business_count > 1` 的节点日**整日剔除**，不进入账本 | 当前 V5/V6 训练与推荐 |
| **份额模式（未来）** | 保留多业务行，按份额参与分配；`is_primary` 全为 False | C3 分配器改造后再启用 |

> 现状：V3 审计显示 **340 个节点日**因多业务被整日剔除（占 0.09%），口径已锁定为"剔除"，不是"均摊"。

## 5. 口径与单位约定

- 金额单位：元；带宽单位：**Mbps**（`build_bandwidth_mbps`）；时间：自然日（业务发生日，非结算日）；
- 评估口径（默认）：`--min-build-bandwidth 500 --eval-winsorize 1`（见 `指标口径对账（新）.md`）；
- 标签版本：`v3.1_daily_weighted_virtual_bindings`；换口径必须升版本号，不得原地覆盖。

## 6. 与现有实现的关系（不重写，只收敛入口）

```
ingest   -> build_v3_daily_business_training.py      （产 ledger = v1_training_pairs...）
quote    -> v5_hybrid_recommendation.py build        （池报价 + 区间）
allocate -> v6_capacity_aware_allocation.py build    （容量约束分配）
report   -> 门槛看板 / 业务级可信度 / RJ 模板报告
```
入口：`wcx2流水线（新）.py` + `pipeline_config.json`（路径与窗口；凭证仅走环境变量）。

## 7. 真实数据校验结果（2026-09-14）

**源头产出（推荐路径）**：`build_v3_daily_business_training.py` 的 ingest 阶段现在**直接产出账本 v1**
（`build_ledger_v1` + `--ledger-output`，默认 `<output-dir>/ledger_node_business_day_v1.csv`），
并新增 `bandwidth_below_floor` 标记；配 `test_ledger_builder.py`（3/3）。

**兼容路径**：旧产物可用 `账本适配（新）.py`（legacy 列名 → schema v1）后置适配。

**校验结果**：真实数据 **281,461 行**（12,131 节点 / 31 天）→ `账本契约检查（新）.py` **ALL PASS**；
两种产出路径的键完全一致。

**发现（分母尾部）**：建设带宽**最小 1 Mbps、P1=150 Mbps**，
**14.61%（41,124 行）低于 500 Mbps** —— 这就是单位收益极端值与 R² 塌陷的源头。
账本不删除这些行，只打 `bandwidth_below_floor=True`；**下游必须显式过滤**（评估默认 `--min-build-bandwidth 500`）。
建议上游明确：这类节点是真实低价值节点（应参与决策）还是口径异常（应修正归因）。
