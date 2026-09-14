# -*- coding: utf-8 -*-
"""P5 凹边际试点（新）：用历史供给分桶的边际流量 ΔT/ΔS（等渗递减）对候选分按业务折减，
输出 V6 可用推荐文件；随后外部跑 V6 对照。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "recent_month_large_mainstream_v5_cap20_curfull（新）"
V6 = ROOT / "recent_month_large_mainstream_v6_capacity_cap20_curfull（新）"
N_BINS = 5

d = pd.read_csv(ROOT / "业务日供给流量（新）.csv", dtype={"customerId": str})
d["S"] = pd.to_numeric(d["supply_mbps"], errors="coerce")
d["T"] = pd.to_numeric(d["traffic_bps"], errors="coerce") / 1e6
d = d.dropna(subset=["S", "T"]); d = d[(d["S"] > 0) & (d["T"] > 0)]

# 业务当前负荷（用于定位所处段）
a = pd.read_csv(V6 / "v6_capacity_allocation_summary.csv", low_memory=False)
b = a[a.pool_level == "business"].copy(); b["business"] = b["business"].astype(str)
cur = dict(zip(b.business, pd.to_numeric(b.current_active_build_bandwidth_mbps, errors="coerce")))
base_slope = {}
factor = {}
for biz, g in d.groupby("customerId"):
    if g["S"].nunique() < N_BINS or len(g) < 20:
        continue
    g = g.sort_values("S")
    edges = np.unique(np.quantile(g["S"], np.linspace(0, 1, N_BINS + 1)))
    if len(edges) < 3:
        continue
    slopes = []
    for i in range(len(edges) - 1):
        w = g[(g["S"] >= edges[i]) & (g["S"] < edges[i + 1])]
        if len(w) >= 3 and w["S"].std() > 0:
            slope = float(np.polyfit(w["S"], w["T"], 1)[0])
            slopes.append(max(slope, 0.0))
        else:
            slopes.append(np.nan)
    slopes = np.array(slopes, dtype=float)
    # 等渗递减（反向累积最小值）
    for i in range(len(slopes) - 2, -1, -1):
        if np.isfinite(slopes[i + 1]):
            slopes[i] = min(slopes[i], slopes[i + 1])
    s0 = slopes[0] if np.isfinite(slopes[0]) and slopes[0] > 0 else np.nan
    if not np.isfinite(s0):
        continue
    base_slope[biz] = s0
    load = cur.get(biz, np.nan)
    pos = int(np.searchsorted(edges, load)) - 1 if pd.notna(load) else 0
    pos = max(0, min(pos, len(slopes) - 1))
    sl = slopes[pos] if np.isfinite(slopes[pos]) else s0
    factor[biz] = float(max(0.0, min(1.0, sl / s0)))

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "str"}, low_memory=False)
for c in ["combined_score_top1", "combined_score_top2", "combined_score_top3"]:
    rec[c] = pd.to_numeric(rec.get(c), errors="coerce")
for c in ["business_top1", "business_top2", "business_top3"]:
    rec[c] = rec[c].astype(str).str.replace(r"\.0$", "", regex=True)

out = rec.copy()
for i, r in rec.iterrows():
    cs = []
    for k in (1, 2, 3):
        biz = str(r[f"business_top{k}"]); s = r[f"combined_score_top{k}"]
        if biz and biz != "nan" and pd.notna(s):
            cs.append((biz, float(s) * factor.get(biz, 1.0)))
    if not cs:
        continue
    cs.sort(key=lambda x: -x[1])
    for k, (biz, s) in enumerate(cs, 1):
        out.at[i, f"business_top{k}"] = biz
        out.at[i, f"combined_score_top{k}"] = s

out.to_csv(ROOT / "v5_node_recommendations_p5（新）.csv", index=False, encoding="utf-8-sig")
o = io.StringIO()
o.write("== P5 经验边际因子（业务层，按当前负荷所在段） ==\n")
o.write("可算业务 %d；因子<1 的业务: %s\n" % (len(factor), {k: round(v, 3) for k, v in factor.items() if v < 0.999}))
o.write("输出: v5_node_recommendations_p5（新）.csv\n")
txt = o.getvalue(); (ROOT / "P5经验边际（新）.txt").write_text(txt, encoding="utf-8"); print(txt)
