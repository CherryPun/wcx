#!/usr/bin/env python3
"""五次探查：上线节点按天分布（决定训练窗口）；验证日期窗口语法。"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient

BASE = "http://superset.yzh-logverse.k8s.qiniu.io"
sr = SupersetSQLClient(base_url=BASE, database_id=19)

def run(sql, label=""):
    print(f"\n=== [SR] {label} ===\n{sql.strip()}")
    try:
        rows = sr.execute_sql(sql=sql, schema="test")
    except Exception as e:
        print("  ERROR:", str(e)[:800]); return []
    print(f"  rows: {len(rows)}")
    for r in rows[:60]:
        print("   ", json.dumps(r, ensure_ascii=False))
    return rows

# 0) 语法验证
run("SELECT CAST('2026-05-01' AS DATE) AS d, DATE_ADD(CAST('2026-05-01' AS DATE), INTERVAL 7 DAY) AS d7", "syntax_test")

# 1) 上线节点按天分布（MIN(day) 作为上线日）
run("""
SELECT online_day, COUNT(*) AS nodes
FROM (SELECT nodeId, MIN(day) AS online_day FROM test.node_day_ops_wide_full GROUP BY nodeId) t
GROUP BY online_day
ORDER BY online_day
""", "online_by_day")
