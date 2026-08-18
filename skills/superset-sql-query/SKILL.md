---
name: superset-sql-query
description: "在 Superset SQL Lab 上执行临时 SQL 查询，适用于“帮我查下这段 SQL”“看看表有哪些字段”“导出这批节点明细”“按条件筛节点并输出 CSV/JSON”等场景。优先用于 `jarvis.node_analysis_data`、`jarvis.node_join` 等 Superset 可查表，支持直接传 SQL、SQL 文件、结果预览和导出。"
---

# Superset SQL 查询

## 适用场景

使用这个 skill 可以：

- 在 Superset SQL Lab 上执行临时 SQL
- 查询 `jarvis.node_analysis_data`、`jarvis.node_join` 等表
- 查看字段结构、抽样数据、业务切换、节点筛选结果
- 输出 preview、JSON、CSV
- 复用本机的 Superset 账号配置

适合处理这类请求：

- “帮我跑一下这段 SQL”
- “看看这个表有哪些字段”
- “按条件筛出节点并导出 CSV”
- “查一批 nodeid 的当前业务/状态”
- “把查询结果落成 json/csv”
- “我给你需求，你帮我写 SQL 并查结果”

当用户直接给业务需求而不是 SQL 时：

- 先按 [references/requirement-to-sql.md](./references/requirement-to-sql.md) 的流程拆需求
- 再按 [references/schema-dictionary.md](./references/schema-dictionary.md) 选表和字段
- 然后生成 SQL、执行查询、返回结果

## 前置要求

优先使用本机共享账号配置：

```bash
cat ~/.codex/.superset_env
```

默认会读取：

```bash
export SUPERSET_USERNAME='your-username'
export SUPERSET_PASSWORD='your-password'
```

如果需要临时覆盖，也可以在命令里传：

```bash
python3 common/run_superset_sql.py --sql "..." --username 'x' --password 'y'
```

## 关键约束

- 只支持 `SELECT` 查询
- 查询时尽量带分区条件，尤其是 `node_analysis_data.day`
- `node_analysis_data.day/hour` 是 `varchar`
- `node_join.day/hour` 是 `integer`
- 关联键通常是：

```sql
node_analysis_data.nodeid = node_join._id
```

如果要按天关联 `node_join`，注意类型转换：

```sql
CAST(node_join.day AS varchar) = node_analysis_data.day
```

## 快速开始

直接执行一段 SQL：

```bash
python3 common/run_superset_sql.py --sql "SELECT nodeid, day FROM node_analysis_data WHERE day = '20260417' LIMIT 10"
```

执行一个 SQL 文件：

```bash
python3 common/run_superset_sql.py --sql-file ./query.sql
```

把结果写成 JSON 和 CSV：

```bash
python3 common/run_superset_sql.py \
  --sql-file ./query.sql \
  --output-json ./result.json \
  --output-csv ./result.csv
```

从标准输入读取 SQL：

```bash
python3 common/run_superset_sql.py --sql-file - <<'SQL'
SELECT nodeid, customersummary
FROM node_analysis_data
WHERE day = '20260417'
  AND nodetype = 'node'
LIMIT 20
SQL
```

## 推荐工作流

1. 先确认表和字段

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'jarvis'
  AND table_name = 'node_analysis_data'
ORDER BY ordinal_position
```

2. 再做小范围抽样

```sql
SELECT *
FROM node_analysis_data
WHERE day = '20260417'
  AND nodetype = 'node'
LIMIT 5
```

3. 再跑正式筛选或导出

- 结果较多时优先写 CSV / JSON
- 需要解释结果时，先给总数、再给重点行

如果用户只给业务问题：

1. 先把问题翻成查询目标
2. 明确时间范围
3. 选 `node_analysis_data` / `node_join`
4. 先跑小样本验证
5. 再跑正式 SQL
6. 输出结果和重点结论

## 常见查询模板

按 nodeid 查最近一天业务：

```sql
SELECT nodeid, customersummary, state, purchasername, purchaserid
FROM node_analysis_data
WHERE day = '20260417'
  AND nodeid IN ('id1', 'id2')
```

查 `node_join` 静态画像：

```sql
SELECT
  _id,
  day,
  nodestaticinfo.stage,
  nodestaticinfo.nominalinfo.deliverytype,
  nodeinfo.status,
  nodeinfo.sevendayavg95ratio
FROM node_join
WHERE day = 20260417
  AND _id = 'some-node-id'
```

关联两张表：

```sql
SELECT
  nd.nodeid,
  nd.day,
  nd.customersummary,
  nd.state,
  nj.nodeinfo.status,
  nj.nodeinfo.needredeployreason
FROM node_analysis_data nd
JOIN node_join nj
  ON nd.nodeid = nj._id
 AND CAST(nj.day AS varchar) = nd.day
WHERE nd.day = '20260417'
  AND nd.nodetype = 'node'
```

## 说明

- 脚本入口：`common/run_superset_sql.py`
- 需求转 SQL 说明：`references/requirement-to-sql.md`
- 常用字段字典：`references/schema-dictionary.md`
- 全量字段清单：`references/full-schema.md`
- 如果用户只想看“结果”，优先返回总数、重点字段和输出文件路径
- 如果用户想“进一步分析原因”，可以继续衔接本仓库里的掉量归因或流失看板脚本
