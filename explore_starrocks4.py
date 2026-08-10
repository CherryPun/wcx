#!/usr/bin/env python3
"""四次探查：customerId->业务名 映射；node_join 属性列路径；starrocks nodeId 是否能在 node_join 命中。"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient

BASE = "http://superset.yzh-logverse.k8s.qiniu.io"
sr = SupersetSQLClient(base_url=BASE, database_id=19)
jv = SupersetSQLClient(base_url=BASE, database_id=2)

def run_sr(sql, label=""):
    print(f"\n=== [SR] {label} ===\n{sql.strip()}")
    try:
        rows = sr.execute_sql(sql=sql, schema="test")
    except Exception as e:
        print("  ERROR:", str(e)[:600]); return []
    print(f"  rows: {len(rows)}")
    for r in rows[:6]:
        print("   ", json.dumps(r, ensure_ascii=False))
    return rows

def run_jv(sql, label=""):
    print(f"\n=== [JV] {label} ===\n{sql.strip()}")
    try:
        rows = jv.execute_sql(sql=sql, schema="jarvis")
    except Exception as e:
        print("  ERROR:", str(e)[:600]); return []
    print(f"  rows: {len(rows)}")
    for r in rows[:6]:
        print("   ", json.dumps(r, ensure_ascii=False))
    return rows

# 1) customer 映射表
for t in ["customer", "customer2", "customer3"]:
    run_sr(f"SELECT * FROM test.{t} LIMIT 3", f"{t}_sample")

# 2) 取一个 starrocks 节点，看它在 node_join 是否命中（nodeId 格式是否一致）
run_sr("SELECT DISTINCT nodeId FROM test.node_day_ops_wide_full WHERE day='2026-08-04' LIMIT 3", "sr_sample_nodes")

# 3) node_join 属性列路径（验证能取到 vendorid/deliverytype 等）
run_jv("""
SELECT
  _id,
  nodestaticinfo.vendorid AS vendorid,
  nodestaticinfo.nominalinfo.deliverytype AS deliverytype,
  nodestaticinfo.nominalinfo.resourcetype AS resourcetype,
  nodestaticinfo.nominalinfo.dialtype AS dialtype,
  nodestaticinfo.nominalinfo.nattype AS nattype,
  nodestaticinfo.nominalinfo.scheduleisps AS scheduleisps,
  nodestaticinfo.regsource AS regsource,
  nodestaticinfo.customermode AS customermode,
  province, isp, city,
  nodestaticinfo.nominalinfo.devicetype AS device_type,
  nodestaticinfo.nominalinfo.archtype AS arch_type,
  nodestaticinfo.nominalinfo.isvm AS isvm
FROM jarvis.node_join
WHERE day = 20260805
LIMIT 3
""", "node_join_attrs")

# 4) 验证某一 starrocks 节点能在 node_join 命中
run_jv("""
SELECT _id, nodestaticinfo.vendorid, nodestaticinfo.nominalinfo.deliverytype
FROM jarvis.node_join
WHERE day = 20260805 AND _id = 'ant242495571d37b3e28bf67e5d7e615'
LIMIT 1
""", "join_hit_check")
