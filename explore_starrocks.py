#!/usr/bin/env python3
"""探查 yzh-starrocks (DB 19) 真实结算表结构，确认粒度/主键/成本收入口径。"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient

BASE = "http://superset.yzh-logverse.k8s.qiniu.io"
client = SupersetSQLClient(base_url=BASE, database_id=19)

def run(sql):
    print("\n=== SQL ===\n", sql.strip())
    rows = client.execute_sql(sql=sql, schema="test")
    print(f"rows returned: {len(rows)}")
    return rows

# 1) 全字段样例
rows = run("SELECT * FROM test.node_day_ops_wide_full LIMIT 1")
if rows:
    print("--- columns & sample (LIMIT 1) ---")
    for k, v in rows[0].items():
        sval = str(v)
        if len(sval) > 80:
            sval = sval[:80] + "..."
        print(f"  {k}: {sval!r}")

# 2) 主键/粒度确认：distinct 计数 + 是否有 day 分区
rows = run("""
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT node_id) AS distinct_nodes,
  COUNT(DISTINCT business) AS distinct_businesses,
  COUNT(DISTINCT day) AS distinct_days,
  MIN(day) AS min_day,
  MAX(day) AS max_day
FROM test.node_day_ops_wide_full
""")
print("--- grain overview ---")
print(json.dumps(rows[0], ensure_ascii=False, indent=2))

# 3) 看 business 取值集合
rows = run("""
SELECT business, COUNT(*) AS cnt
FROM test.node_day_ops_wide_full
GROUP BY business
ORDER BY cnt DESC
LIMIT 30
""")
print("--- business values ---")
for r in rows:
    print(f"  {r['business']}: {r['cnt']}")

# 4) 看 node_id 格式 + 一个节点的多业务多天样例
rows = run("""
SELECT node_id, business, day,
       cost_finalAmount, cost_settlement, cost_price,
       revenue_finalAmount, revenue_amount, profit_profitAmount,
       settlePeriodType, priceNumber
FROM test.node_day_ops_wide_full
WHERE day = 20260805
LIMIT 10
""")
print("--- sample node/business/day rows ---")
for r in rows:
    print("  ", json.dumps(r, ensure_ascii=False))
