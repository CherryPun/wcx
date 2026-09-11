# -*- coding: utf-8 -*-
"""规则对比（新）：同一数据下比较不同排序规则的切换/持有/集中度。
规则：value(边际增益密度)、urgency(当前利润低优先)、history(节点上线早优先)、random(多seed分布)。"""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "recent_month_large_mainstream_v5_cap20_curfull（新）"
V6 = ROOT / "recent_month_large_mainstream_v6_capacity_cap20_curfull（新）"
V3 = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）"
MIN_GAIN, N_SEED = 0.03, 20

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "str"}, low_memory=False)
nr = pd.read_csv(V6 / "v6_capacity_node_report.csv", dtype={"node_id": "str"}, low_memory=False)
bw = {str(k): float(v) for k, v in zip(nr["node_id"], pd.to_numeric(nr["build_bandwidth_mbps"], errors="coerce").fillna(1000.0))}
pool = pd.read_csv(V6 / "v6_capacity_pool_summary.csv", low_memory=False)
pool["business"] = pool["business"].astype(str)
biz = pool[pool["pool_level"] == "business"]
remain0 = {str(b): float(h) for b, h in zip(biz["business"], pd.to_numeric(biz["allocatable_headroom_mbps"], errors="coerce").fillna(0.0))}
facts = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv", dtype={"node_id": "str"}, low_memory=False)
facts["sample_day"] = pd.to_datetime(facts["sample_day"], errors="coerce")
online = facts.groupby("node_id")["sample_day"].min().dt.strftime("%Y-%m-%d").to_dict()

rec["cur"] = rec["current_business"].astype(str).str.replace(r"\.0$", "", regex=True)
rec["curp"] = pd.to_numeric(rec.get("current_actual_platform_profit_1d"), errors="coerce")
rec = rec[rec["node_id"].astype(str).isin(bw)].copy()

items = []
for _, r in rec.iterrows():
    cands = []
    for k in (1, 2, 3):
        b = str(r.get(f"business_top{k}", "")).replace(".0", "")
        s = pd.to_numeric(r.get(f"combined_score_top{k}"), errors="coerce")
        pf = pd.to_numeric(r.get(f"platform_unit_profit_top{k}"), errors="coerce")
        if b and b != "nan" and pd.notna(s):
            cands.append((b, float(s), pf))
    if not cands:
        continue
    cs = pd.to_numeric(r.get("current_predicted_combined_score"), errors="coerce")
    gain = cands[0][1] - (float(cs) if pd.notna(cs) else 0.0)
    items.append({"node_id": r["node_id"], "cur": r["cur"], "bw": bw[r["node_id"]], "cands": cands,
                  "gain": gain, "density": gain / max(bw[r["node_id"]], 1.0),
                  "urgent": r["curp"] if pd.notna(r["curp"]) else 1e9,
                  "online": online.get(r["node_id"], "9999-12-31")})

def simulate(order):
    remain = dict(remain0)
    sw = Counter(); nsw = 0
    for it in order:
        chosen = None
        for b, s, pf in it["cands"]:
            if it["gain"] < MIN_GAIN:
                break
            if pf is not None and pd.notna(pf) and pf <= 0:
                continue
            if remain.get(b, 0.0) >= it["bw"]:
                chosen = b
                break
        if chosen and chosen != it["cur"]:
            remain[chosen] = remain.get(chosen, 0.0) - it["bw"]
            if it["cur"] and it["cur"] != "nan" and it["cur"] in remain:
                remain[it["cur"]] = remain.get(it["cur"], 0.0) + it["bw"]
            sw[chosen] += 1; nsw += 1
    cnt = sum(sw.values())
    mx = max(sw.values()) / cnt if cnt else 0
    hhi = sum((v / cnt) ** 2 for v in sw.values()) if cnt else 0
    return nsw, mx, hhi, sw.most_common(3)

o = io.StringIO()
o.write("== 规则对比（切换目标口径；N=%d） ==\n" % len(items))
o.write("%-10s %8s %10s %8s %s\n" % ("规则", "切换", "max占比", "HHI", "Top3目标"))
orders = {
    "value": sorted(items, key=lambda x: -x["density"]),
    "urgency": sorted(items, key=lambda x: (x["urgent"], -x["density"])),
    "history": sorted(items, key=lambda x: (x["online"], -x["density"])),
}
for name, order in orders.items():
    n, mx, hhi, top = simulate(order)
    o.write("%-10s %8d %10.4f %8.4f %s\n" % (name, n, mx, hhi, top))
# random 多 seed 分布
ns, mxs, hhis = [], [], []
for s in range(N_SEED):
    rng = np.random.RandomState(s)
    order = sorted(items, key=lambda x: rng.rand())
    n, mx, hhi, _ = simulate(order)
    ns.append(n); mxs.append(mx); hhis.append(hhi)
o.write("%-10s %8.0f %10.4f %8.4f (n=%d seeds; 范围 切换 %d~%d, max %.3f~%.3f)\n"
        % ("random", np.mean(ns), np.mean(mxs), np.mean(hhis), N_SEED, min(ns), max(ns), min(mxs), max(mxs)))
txt = o.getvalue()
(ROOT / "规则对比结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
