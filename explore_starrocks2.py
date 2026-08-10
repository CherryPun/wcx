#!/usr/bin/env python3
"""二次探查：确认粒度、列出 test 库所有表、检查是否存在 business 维度。"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient

BASE = "http://superset.yzh-logverse.k8s.qiniu.io"
client = SupersetSQLClient(base_url=BASE, database_id=19)

def run(sql, label=""):
    print(f"\n=== {label} ===\n{sql.strip()}")
    try:
        rows = client.execute_sql(sql=sql, schema="test")
    except Exception as e:
        print("  ERROR:", str(e)[:500])
        return []
    print(f"  rows: {len(rows)}")
    return rows

# A) 粒度
run("""
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT nodeId) AS distinct_nodes,
  COUNT(DISTINCT day) AS distinct_days,
  COUNT(DISTINCT customerId) AS distinct_customers,
  MIN(day) AS min_day,
  MAX(day) AS max_day
FROM test.node_day_ops_wide_full
""", "grain_overview")

# B) 列出 test 库所有表
rows = run("SELECT table_name FROM information_schema.tables WHERE table_schema='test' ORDER BY table_name", "tables_in_test")
if rows:
    print("  TABLES:", [r["table_name"] for r in rows])

# C) 单节点跨天行数（判断粒度是否为 nodeId x day）
rows = run("""
SELECT nodeId, COUNT(*) AS rows_per_node, COUNT(DISTINCT day) AS days, COUNT(DISTINCT customerId) AS custs
FROM test.node_day_ops_wide_full
GROUP BY nodeId
ORDER BY rows_per_node DESC
LIMIT 5
""", "node_grain")

# D) 是否存在 revenue/cost 非空的节点（成本字段何时填充）
rows = run("""
SELECT
  SUM(CASE WHEN cost_finalAmount IS NULL OR cost_finalAmount='None' THEN 0 ELSE 1 END) AS cost_nonnull,
  SUM(CASE WHEN revenue_finalAmount IS NULL OR revenue_finalAmount='None' THEN 0 ELSE 1 END) AS rev_nonnull,
  SUM(CASE WHEN cost_finalAmount IS NULL OR cost_finalAmount='None' THEN 1 ELSE 0 END) AS cost_null,
  SUM(CASE WHEN revenue_finalAmount IS NULL OR revenue_finalAmount='None' THEN 1 ELSE 0 END) AS rev_null
FROM test.node_day_ops_wide_full
""", "nullness")

# E) node_day_ops_wide（非 full）列对比，看是否有 business
rows = run("SELECT * FROM test.node_day_ops_wide LIMIT 1", "wide_notfull_sample")
if rows:
    cols = list(rows[0].keys())
    print("  cols:", cols)
    print("  has business col?:", any('business' in c.lower() for c in cols))
    print("  has nodeId?:", 'nodeId' in cols)
