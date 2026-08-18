---
name: superset-node-day-ops-wide-query
description: 在 Superset SQL Lab 中查询 mysql 数据源 yzh-starrocks 的 test.node_day_ops_wide_full 节点日宽表，支持按日期、客户、节点、地域、业务、状态筛选，查看带宽/成本/收入/利润指标，并导出 CSV 或 JSON。用户提到节点日宽表、node_day_ops_wide_full、yzh-starrocks、test schema 或需要按字段分析该表时使用。
---

# 节点日宽表查询

## 固定目标

- Superset 数据库：`yzh-starrocks`
- 数据库类型：`mysql`
- Superset `database_id`：`19`（执行前可通过数据库列表按名称复核）
- Schema：`test`
- 表：`node_day_ops_wide_full`
- 完整字段定义：[references/node_day_ops_wide_full_schema.json](references/node_day_ops_wide_full_schema.json)

## 查询流程

1. 先明确日期、节点、客户、业务、地域或状态筛选条件。
2. 读取字段参考，确认字段名、类型和中文含义，不根据字段名猜测不存在的指标口径。
3. 优先使用 `day` 条件缩小范围；`day` 类型为 `date`，格式为 `YYYY-MM-DD`。
4. 首次查询只选必要字段并使用 `LIMIT 5/10` 验证口径，再执行正式查询或导出。
5. 默认只执行 `SELECT`；禁止执行 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER`、`TRUNCATE` 等写入或 DDL 操作。
6. 结果较大时写入 CSV/JSON，并在回复中说明完整结果文件路径，不主动截断正式导出数据。

## 执行入口

复用已注册的 Superset SQL 执行器，并固定数据库和 Schema：

```bash
python3 /Users/nany/.codex/skills/superset-sql-query/common/run_superset_sql.py \
  --database-id 19 \
  --schema test \
  --sql "SELECT customerId, day, nodeId, customerName, state FROM node_day_ops_wide_full WHERE day = '2026-08-05' LIMIT 10" \
  --preview 10
```

该执行器从 `/Users/nany/.codex/.superset_env` 读取 Superset 凭据，支持 `--output-json` 和 `--output-csv`。查询失败时保留 HTTP 状态码和 Superset/Trino 响应体，不用空结果替代错误。

## 常用字段分组

- 主键与日期：`customerId`、`day`、`nodeId`
- 业务与资源：`customerName`、`deliveryType`、`resourceType`、`virtualCustomers`、`stage`
- 位置与网络：`province`、`city`、`isp`、`realISP`、`natType`、`tcpNatType`、`udpNatType`
- 带宽：`buildBandwidth`、`peak95`、`analyzePeak95`、`eveningAvg`、`eveningPeak95`、`unEveningAvg`
- 成本：`cost_original`、`cost_bonus`、`cost_slaDeduction`、`cost_settlement`、`cost_finalAmount`
- 收入：`revenue_amount`、`revenue_finalAmount`、`revenue_estimatedFinalAmount`
- 利润：`profit_profitAmount`、`profit_profitRate`、`profit_estimatedProfitAmount`、`profit_estimatedProfitRate`
- 状态与时间：`state`、`snapshotTime`、`updatedTime`、`peak95Time`

## 查询模板

按日期抽样：

```sql
SELECT customerId, day, nodeId, customerName, province, isp, state,
       peak95, cost_finalAmount, revenue_finalAmount,
       profit_profitAmount, profit_profitRate
FROM node_day_ops_wide_full
WHERE day = '2026-08-05'
LIMIT 10
```

按客户和日期汇总：

```sql
SELECT day, customerId, customerName,
       COUNT(DISTINCT nodeId) AS node_count,
       SUM(peak95) AS peak95_sum,
       SUM(cost_finalAmount) AS cost_final_amount,
       SUM(revenue_finalAmount) AS revenue_final_amount,
       SUM(profit_profitAmount) AS profit_amount
FROM node_day_ops_wide_full
WHERE day BETWEEN '2026-08-01' AND '2026-08-05'
  AND customerId = 123
GROUP BY day, customerId, customerName
ORDER BY day, customerId
```

按节点查看日变化：

```sql
SELECT day, nodeId, customerName, state, province, isp,
       peak95, peak95Ratio, peak95Time,
       cost_finalAmount, revenue_finalAmount, profit_profitAmount
FROM node_day_ops_wide_full
WHERE nodeId = 'node-id'
  AND day BETWEEN '2026-08-01' AND '2026-08-05'
ORDER BY day
```

## 口径约束

- `peak95` 的字段说明为日 95 带宽，单位 `bps`；`buildBandwidth` 和 `idcBandwidth` 的字段说明为 `Mbps`，不得混用单位后直接比较。
- `peak95Ratio`、`peakMaxRatio` 的字段类型是 `int`，字段说明只有“占比”，未提供百分比还是小数的明确口径；展示前必须保留原始值并标注待确认。
- JSON 字段（例如 `nodeTags`、`stairs`、`virtualCustomers`）只有在明确结构后再展开，不能假设数组或对象路径。
- 数值聚合前检查空值和重复粒度；该表是日宽表，结果是否可按节点直接求和必须依据用户指标和字段语义判断。
