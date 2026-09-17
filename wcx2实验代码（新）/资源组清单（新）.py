# -*- coding: utf-8 -*-
"""[wcx2 新增] 资源组清单与节点映射（新）：用于与 RJ 的 2,757 组做键级对照"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
V3 = R / "recent_month_large_mainstream_v3_daily_weighted（新）"
PAIRS = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"
DAYS = ["2026-09-09", "2026-09-08"]

p = pd.read_csv(PAIRS, low_memory=False, usecols=lambda c: c in [
    "node_id", "sample_day", "province", "isp", "scheduleisps", "nattype", "construction_bandwidth_mbps"])
p = p[p.sample_day.isin(DAYS)].copy()
aud = pd.read_csv(V3 / "node_day_sampling_audit_v3.csv", low_memory=False,
                  usecols=lambda c: c in ["node_id", "sample_day", "daily_city", "daily_transprovrate"])
p = p.merge(aud, on=["node_id", "sample_day"], how="left")
bn = pd.read_csv(R / "snapshots" / "bench_20260909.csv", low_memory=False,
                 usecols=["node_id", "overall_packet_loss_benchmark_satisfaction_pct"])
v = pd.to_numeric(bn["overall_packet_loss_benchmark_satisfaction_pct"], errors="coerce")
bn["q"] = ["未知" if pd.isna(x) else ("优" if x >= 95 else ("中" if x >= 80 else "差")) for x in v]
p = p.merge(bn[["node_id", "q"]].drop_duplicates("node_id"), on="node_id", how="left")

p["city"] = p["daily_city"].fillna("缺失").astype(str)
p["transprov"] = pd.to_numeric(p["daily_transprovrate"], errors="coerce").map(
    lambda x: "缺失" if pd.isna(x) else ("出省" if x >= 100 else "本省"))
p["q"] = p["q"].fillna("未知")
p["ipv6"] = "缺失(数仓无节点级源)"
p["bw"] = pd.to_numeric(p["construction_bandwidth_mbps"], errors="coerce")
p = p[p.bw >= 500]
DIMS = ["province", "city", "isp", "scheduleisps", "transprov", "nattype", "ipv6", "q"]
for c in DIMS:
    p[c] = p[c].fillna("缺失").astype(str)
p["grp"] = p[DIMS].agg("|".join, axis=1)

grp = p.groupby("grp").agg(nodes=("node_id", "nunique"), rows=("node_id", "size"),
                           bw_total_mbps=("bw", "sum"), days=("sample_day", "nunique")).reset_index()
grp = grp.join(p[DIMS + ["grp"]].drop_duplicates("grp").set_index("grp"), on="grp")
grp["证据"] = np.where(grp.nodes >= 3, "可用", "证据不足")
grp = grp[["grp", *DIMS, "nodes", "rows", "bw_total_mbps", "days", "证据"]]
grp.to_csv(R / "资源组清单（新）.csv", index=False, encoding="utf-8-sig")
p[["node_id", "sample_day", *DIMS, "grp", "bw"]].to_csv(R / "节点资源组映射（新）.csv", index=False, encoding="utf-8-sig")

print(f"资源组数 {grp.shape[0]} | 节点-日行 {len(p)} | 节点 {p.node_id.nunique()}")
print("缺失情况：")
for c in DIMS:
    print(f"  {c:<16} 缺失/未知占比 {p[c].isin(['缺失', '未知']).mean():.1%}")
print("规模 Top5：")
print(grp.nlargest(5, "nodes")[["grp", "nodes", "bw_total_mbps", "证据"]].to_string(index=False, max_colwidth=46))
print("\n[out] 资源组清单（新）.csv / 节点资源组映射（新）.csv")
