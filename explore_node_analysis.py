"""探查 node_analysis_data：分批拉静态字段（避免大 IN 超时）"""
import os, sys
sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
c = SupersetSQLClient(database_id=2)
def run(sql): return pd.DataFrame(c.execute_sql(sql=sql))

DAY = "20260805"
out = pd.read_csv(os.path.join(HERE, "outcomes_raw.csv"))
sample_nodes = out["node_id"].astype(str).unique().tolist()
print(f"样本节点数: {len(sample_nodes)}")

BATCH = 1500
parts = []
for i in range(0, len(sample_nodes), BATCH):
    chunk = sample_nodes[i:i+BATCH]
    nids = "','".join(chunk)
    df = run(f"""
    SELECT nodeid,
           ANY_VALUE(bandwidth)      AS bw_rated,
           ANY_VALUE(corenum)        AS corenum,
           ANY_VALUE(memtotal)       AS memtotal,
           ANY_VALUE(totaldisksize)  AS totaldisksize,
           ANY_VALUE(hdddisksize)    AS hdddisksize,
           ANY_VALUE(ssddisksize)    AS ssddisksize,
           ANY_VALUE(systemdisksize) AS systemdisksize,
           ANY_VALUE(hardwaretype)   AS hardwaretype,
           ANY_VALUE(os)             AS os,
           ANY_VALUE(arch)           AS arch
    FROM jarvis.node_analysis_data
    WHERE day = '{DAY}' AND nodeid IN ('{nids}')
    GROUP BY nodeid
    """)
    parts.append(df)
    print(f"  batch {i//BATCH+1}: +{len(df)} nodes")

res = pd.concat(parts, ignore_index=True)
print(f"\n覆盖节点数: {len(res)} / {len(sample_nodes)}")
print("\n== 样本折叠行（前8）==")
with pd.option_context('display.max_columns', None, 'display.width', 200):
    print(res.head(8).to_string())
print("\n== 静态字段缺失率 ==")
for col in ["bw_rated","corenum","memtotal","totaldisksize","hardwaretype","os","arch"]:
    miss = res[col].isna().mean()
    print(f"  {col}: 缺失率={miss:.3f}")
res.to_csv(os.path.join(HERE, "node_analysis_static_sample.csv"), index=False)
print("\n已保存 node_analysis_static_sample.csv")
