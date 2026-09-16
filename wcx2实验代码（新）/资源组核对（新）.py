# -*- coding: utf-8 -*-
"""[wcx2 新增] 资源组核对（新）：解释"710 组 vs RJ 2,757 组"的差距来源。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent
V3 = R / "recent_month_large_mainstream_v3_daily_weighted（新）"
DAY = "2026-09-09"
p = pd.read_csv(V3 / "v1_training_pairs_large_mainstream_recent_1m.csv", low_memory=False,
                usecols=lambda c: c in ["node_id", "sample_day", "province", "isp", "scheduleisps",
                                        "network_schedule_type", "nattype", "construction_bandwidth_mbps"])
p = p[(p.sample_day == DAY) & (pd.to_numeric(p.construction_bandwidth_mbps, errors="coerce") >= 500)]
print(f"样本节点-业务行 {len(p)} | 节点 {p.node_id.nunique()}")

aud = pd.read_csv(V3 / "node_day_sampling_audit_v3.csv", low_memory=False,
                  usecols=lambda c: c in ["node_id", "sample_day", "daily_city", "capacity_peak95_bps", "daily_transprovrate"])
aud = aud[aud.sample_day == DAY].drop_duplicates("node_id")
p = p.merge(aud, on="node_id", how="left")
p["transprov"] = pd.to_numeric(p.get("daily_transprovrate"), errors="coerce").map(
    lambda v: "未知" if pd.isna(v) else ("出省" if v >= 100 else "本省"))
prof = pd.read_csv(R / "recent_month_large_1d（新）" / "multibusiness_nodes_large_recent_1m.csv", low_memory=False)
ic_cands = [c for c in ["analysis_issupportipv6", "join_isipv6schedule", "dial_ipv6_enable"] if c in prof.columns]
ic = ic_cands[0] if ic_cands else None
pv = prof[["node_id"] + ic_cands].drop_duplicates("node_id")
pv["ipv6_any"] = pv[ic_cands].astype(str).apply(lambda r: next((x for x in r if x not in ("nan", "None", "")), "未知"), axis=1)
p = p.merge(pv[["node_id", "ipv6_any"]], on="node_id", how="left")

bench = pd.read_csv(R / "snapshots" / f"bench_{DAY.replace('-', '')}.csv", low_memory=False,
                    usecols=["node_id", "overall_packet_loss_benchmark_satisfaction_pct"])
bench["q"] = pd.to_numeric(bench["overall_packet_loss_benchmark_satisfaction_pct"], errors="coerce")
bench["q"] = bench["q"].map(lambda v: "未知" if pd.isna(v) else ("优" if v >= 95 else ("中" if v >= 80 else "差")))
p = p.merge(bench[["node_id", "q"]].drop_duplicates("node_id"), on="node_id", how="left")

p["city"] = p["daily_city"].fillna("未知").astype(str)
p["q"] = p["q"].fillna("未知")
for c in ["province", "isp", "scheduleisps", "network_schedule_type", "nattype"]:
    if c in p.columns:
        p[c] = p[c].fillna("未知").astype(str)

dims = ["province", "city", "isp", "scheduleisps", "transprov", "nattype", "ipv6", "q"]
p["ipv6"] = p["ipv6_any"].fillna("未知").astype(str)
if "network_schedule_type" not in p.columns:
    p["network_schedule_type"] = "未知"
dims = [c for c in dims if c in p.columns]
print("\n各维度取值数（含'未知'）：")
for c in dims:
    vc = p[c].value_counts()
    print(f"  {c:<24} {vc.size:>4} 档 | 未知占比 {p[c].eq('未知').mean():.1%} | Top3 {list(vc.index[:3])}")

print("\n逐维累加后的组数（观察到的组合）：")
cum = p[[]].copy()
cols = []
for c in dims:
    cols.append(c)
    n = p[cols].astype(str).agg("|".join, axis=1).nunique()
    print(f"  +{c:<24} -> {n} 组")

prod = 1
for c in dims:
    prod *= p[c].nunique()
print(f"\n理论上限（各维取值数相乘）= {prod:,} 组 | 实际观测 {p[dims].astype(str).agg('|'.join, axis=1).nunique()} 组")
print(f"节点数 {p.node_id.nunique()}（RJ 口径资源组 2,757，需更多节点与压测覆盖）")
