#!/usr/bin/env python3
"""验证 node_join 的 _id IN 过滤是否生效，以及单节点是否多行。"""
import os, sys, json
sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient

BASE = "http://superset.yzh-logverse.k8s.qiniu.io"
jv = SupersetSQLClient(base_url=BASE, database_id=2)

nid = "ant2124d4ec302243c918244715a30b6"

# 1) 单节点 COUNT
sql1 = f"SELECT _id, COUNT(*) AS n FROM jarvis.node_join WHERE day=20260805 AND _id='{nid}' GROUP BY _id"
r1 = jv.execute_sql(sql=sql1, schema="jarvis")
print("single-node count:", json.dumps(r1, ensure_ascii=False))

# 2) IN 单节点，看返回行数（是否=1 还是被忽略返回很多）
sql2 = f"SELECT _id FROM jarvis.node_join WHERE day=20260805 AND _id IN ('{nid}')"
r2 = jv.execute_sql(sql=sql2, schema="jarvis")
print("IN single-node rows:", len(r2), " ids:", sorted({x['_id'] for x in r2})[:5])

# 3) 验证 IN 多节点是否真的只返回这些节点（取3个已知ID）
ids3 = "','".join([nid, "ant242495571d37b3e28bf67e5d7e615", "ant02305fddca66f1c5b21ffc29122c2"])
sql3 = f"SELECT _id FROM jarvis.node_join WHERE day=20260805 AND _id IN ('{ids3}')"
r3 = jv.execute_sql(sql=sql3, schema="jarvis")
got = sorted({x['_id'] for x in r3})
print("IN 3-node rows:", len(r3), " distinct ids:", got)
