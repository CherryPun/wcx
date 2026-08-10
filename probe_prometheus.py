"""Prometheus 静态/在盘库存特征可行性探查（ZFS/SMART/盘型）。

仅抽 80 个 node_analysis 覆盖到的节点，验证 node_id 映射与特征可用性。
指标（非运行产生的：盘型/SMART健康/库存）：
  node_disk_ata_rotation_rate_rpm  -> 0=SSD, >0=HDD（盘型，静态）
  jarvis_disk_available_spare     -> SSD 可用备件(%)，健康态
  jarvis_disk_bad_block_count     -> 坏块数，健康态
  jarvis_disk_case_temperature    -> 盘体温度(℃)，健康态
"""
import os, sys, json, urllib.parse, urllib.request
sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
import pandas as pd

BASE = "https://vm-select.mvm.qiniu.io/select/293:0/prometheus/api/v1"

def query(expr, timeout=30):
    data = urllib.parse.urlencode({"query": expr}).encode()
    req = urllib.request.Request(BASE + "/query", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

HERE = os.path.dirname(os.path.abspath(__file__))
en = pd.read_csv(os.path.join(HERE, "nodes_enriched.csv"))
sample = en[en["os"].notna()]["node_id"].astype(str).head(80).tolist()
regex = "|".join(sample)
print(f"样本节点: {len(sample)}")

METRICS = {
    "rotation_rate": "node_disk_ata_rotation_rate_rpm",
    "available_spare": "jarvis_disk_available_spare",
    "bad_block": "jarvis_disk_bad_block_count",
    "case_temp": "jarvis_disk_case_temperature",
}
for key, m in METRICS.items():
    try:
        payload = query(f'{m}{{node_id=~"{regex}"}}')
        res = payload.get("data", {}).get("result", [])
        # 每个 node_id 取一个值（多盘时取均值）
        from collections import defaultdict
        agg = defaultdict(list)
        for item in res:
            nid = item["metric"].get("node_id")
            try: v = float(item["value"][1])
            except (TypeError, ValueError): continue
            agg[nid].append(v)
        cov = len(agg)
        print(f"  {key:16s} 命中节点 {cov:3d}/{len(sample)}  示例值: " +
              (f"min/mean/max={min(min(v) for v in agg.values()):.2f}/"
               f"{sum(sum(v) for v in agg.values())/sum(len(v) for v in agg.values()):.2f}/"
               f"{max(max(v) for v in agg.values()):.2f}" if cov else "无数据"))
    except Exception as e:
        print(f"  {key:16s} 查询失败: {e}")
