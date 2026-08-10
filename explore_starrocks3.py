#!/usr/bin/env python3
"""三次探查：打印真实聚合值；核查 customerId 是否即业务维度；检查 daily_pnl / join_mv1 是否含 business。"""
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
        print("  ERROR:", str(e)[:600])
        return []
    print(f"  rows: {len(rows)}")
    for r in rows[:8]:
        print("   ", json.dumps(r, ensure_ascii=False))
    return rows

# 1) 真实聚合值
run("""
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT nodeId) AS distinct_nodes,
  COUNT(DISTINCT day) AS distinct_days,
  COUNT(DISTINCT customerId) AS distinct_customers,
  MIN(day) AS min_day,
  MAX(day) AS max_day
FROM test.node_day_ops_wide_full
""", "grain_real")

# 2) 单节点粒度（判断是否为 nodeId x customerId x day）
run("""
SELECT nodeId, COUNT(*) AS rows_per_node, COUNT(DISTINCT day) AS days, COUNT(DISTINCT customerId) AS custs
FROM test.node_day_ops_wide_full
GROUP BY nodeId
ORDER BY rows_per_node DESC
LIMIT 5
""", "node_grain_real")

# 3) customerId 取值样例（是否即业务）
run("""
SELECT customerId, customerName, purchaserName, COUNT(*) AS cnt
FROM test.node_day_ops_wide_full
GROUP BY customerId, customerName, purchaserName
ORDER BY cnt DESC
LIMIT 20
""", "customers")

# 4) daily_pnl 结构
run("SELECT * FROM test.daily_pnl LIMIT 1", "daily_pnl_sample")
run("""
SELECT COUNT(*) AS n, COUNT(DISTINCT nodeId) AS nodes, COUNT(DISTINCT day) AS days
FROM test.daily_pnl
""", "daily_pnl_grain")

# 5) join_mv1 结构
run("SELECT * FROM test.join_mv1 LIMIT 1", "join_mv1_sample")
