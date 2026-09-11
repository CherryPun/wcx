# -*- coding: utf-8 -*-
"""P2 基线度量（新）：V5 测试集 hit@1/@3 + 容量稀释因子。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"
V6 = HERE / "recent_month_large_mainstream_v6_capacity_full（新）"

p = pd.read_csv(V5 / "v5_temporal_validation_predictions.csv", dtype={"node_id": "string", "business": "string"}, low_memory=False)
p["business"] = p["business"].astype(str)
for c in ["sample_weight", "actual_miner_unit_income", "actual_platform_unit_profit",
          "predicted_miner_unit_income", "predicted_platform_unit_profit"]:
    p[c] = pd.to_numeric(p.get(c), errors="coerce")
p["pred_combined"] = p["predicted_miner_unit_income"].fillna(0) + p["predicted_platform_unit_profit"].fillna(0)
p["actual_combined"] = p["actual_miner_unit_income"].fillna(0) + p["actual_platform_unit_profit"].fillna(0)

# 观测业务：组内 actual 非空的业务
obs = p[p["actual_miner_unit_income"].notna() | p["actual_platform_unit_profit"].notna()]
has_obs = obs.groupby(["node_id", "sample_day"]).size()
o = io.StringIO()
o.write("== V5 测试预测文件 ==\n")
o.write("  行数=%d；组(node-day)=%d；其中含观测业务组=%d\n\n" % (len(p), p.groupby(["node_id", "sample_day"]).ngroups, len(has_obs)))

# hit@1/@3
groups = list(p.groupby(["node_id", "sample_day"], sort=False))
h1 = h3 = wsum = 0.0
covered = 0
for (nid, day), g in groups:
    og = g[g["actual_miner_unit_income"].notna() | g["actual_platform_unit_profit"].notna()]
    if og.empty:
        continue
    trueb = og.sort_values("sample_weight", ascending=False)["business"].iloc[0]
    top = g.sort_values("pred_combined", ascending=False)["business"].head(3).tolist()
    w = float(og["sample_weight"].fillna(1.0).sum())
    h1 += float(top[0] == trueb) * w
    h3 += float(trueb in top) * w
    wsum += w
    covered += 1
o.write("  有观测业务节点日 hit@1=%.2f%% hit@3=%.2f%% (n=%d)\n\n" % (
    h1 / max(wsum, 1) * 100, h3 / max(wsum, 1) * 100, covered))

# 稀释因子（业务层：load/ceiling）
a = pd.read_csv(V6 / "v6_capacity_allocation_summary.csv", low_memory=False)
biz = a[a["pool_level"].eq("business")].copy()
biz["dilution"] = np.minimum(1.0, biz["raw_capacity_ceiling_mbps"] / biz["unconstrained_top1_load_mbps"].replace(0, np.nan))
biz["will_dilute"] = biz["unconstrained_top1_overflow_mbps"].fillna(0) > 0
o.write("== 容量稀释（业务层） ==\n")
o.write("  全切会稀释(overflow>0)业务 n=%d / %d\n" % (int(biz["will_dilute"].sum()), len(biz)))
for _, r in biz[biz["will_dilute"]].sort_values("unconstrained_top1_overflow_mbps", ascending=False).iterrows():
    o.write("    %s %s: load=%.0f cap=%.0f 稀释因子≈%.2f 需扩=%.0f Gbps\n" % (
        r["business"], r["business_name"], r["unconstrained_top1_load_mbps"], r["raw_capacity_ceiling_mbps"],
        r["dilution"], max(r["unconstrained_top1_load_mbps"] - r["raw_capacity_ceiling_mbps"], 0) / 1000))
txt = o.getvalue()
(HERE / "P2基线度量（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
