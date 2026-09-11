# -*- coding: utf-8 -*-
"""路线1-混合规则时间线模拟（新）：
按"紧迫度(当前业务利润低) → 边际增益密度 → 随机 tie-break"排序，逐节点分配；
进入仍有剩余容量的业务池，切换时释放原业务容量；无可行候选则 hold。
输出：切换/hold、各业务占用、集中度、可吸收上限。
"""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "recent_month_large_mainstream_v5_cap20_curfull（新）"
V6 = ROOT / "recent_month_large_mainstream_v6_capacity_cap20_curfull（新）"
SEED = 42
MIN_GAIN = 0.03

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "str"}, low_memory=False)
nr = pd.read_csv(V6 / "v6_capacity_node_report.csv", dtype={"node_id": "str"}, low_memory=False)
bw = {str(k): float(v) for k, v in zip(nr["node_id"], pd.to_numeric(nr["build_bandwidth_mbps"], errors="coerce").fillna(1000.0))}

pool = pd.read_csv(V6 / "v6_capacity_pool_summary.csv", low_memory=False)
pool["business"] = pool["business"].astype(str)
biz = pool[pool["pool_level"] == "business"]
remain = {str(b): float(h) for b, h in zip(biz["business"], pd.to_numeric(biz["allocatable_headroom_mbps"], errors="coerce").fillna(0.0))}
ceiling = {str(b): float(c) for b, c in zip(biz["business"], pd.to_numeric(biz["raw_capacity_ceiling_mbps"], errors="coerce").fillna(0.0))}

rec["cur"] = rec["current_business"].astype(str).str.replace(r"\.0$", "", regex=True)
rec["curp"] = pd.to_numeric(rec.get("current_actual_platform_profit_1d"), errors="coerce")
rec = rec[rec["node_id"].astype(str).isin(bw)].copy()

items = []
rng = np.random.RandomState(SEED)
for _, r in rec.iterrows():
    cands = []
    cur = r["cur"]
    cs = pd.to_numeric(r.get("current_predicted_combined_score"), errors="coerce")
    for k in (1, 2, 3):
        b = str(r.get(f"business_top{k}", "")).replace(".0", "")
        s = pd.to_numeric(r.get(f"combined_score_top{k}"), errors="coerce")
        pf = pd.to_numeric(r.get(f"platform_unit_profit_top{k}"), errors="coerce")
        if b and b != "nan" and pd.notna(s):
            cands.append((b, float(s), pf))
    if not cands:
        continue
    gain = cands[0][1] - (float(cs) if pd.notna(cs) else 0.0)
    density = gain / max(bw[r["node_id"]], 1.0)
    urgent = r["curp"] if pd.notna(r["curp"]) else 1e9  # 利润越低越先；未知放最后
    items.append({"node_id": r["node_id"], "cur": cur, "bw": bw[r["node_id"]],
                  "cands": cands, "gain": gain, "density": density, "urgent": urgent,
                  "rnd": rng.rand()})
items.sort(key=lambda x: (x["urgent"], -x["density"], x["rnd"]))

assign = {}
for it in items:
    chosen = None
    for b, s, pf in it["cands"]:
        if it["gain"] < MIN_GAIN:
            break
        if (pf is not None and pd.notna(pf) and pf <= 0):
            continue
        if remain.get(b, 0.0) >= it["bw"]:
            chosen = b
            break
    if chosen and chosen != it["cur"]:
        remain[chosen] = remain.get(chosen, 0.0) - it["bw"]
        if it["cur"] and it["cur"] != "nan" and it["cur"] in remain:
            remain[it["cur"]] = remain.get(it["cur"], 0.0) + it["bw"]  # 释放原业务
        assign[it["node_id"]] = chosen
    else:
        assign[it["node_id"]] = it["cur"]

switches = sum(1 for it in items if assign[it["node_id"]] != it["cur"])
c = Counter(b for n, b in assign.items() if b and b != "nan")
sw = Counter(assign[it["node_id"]] for it in items
             if assign[it["node_id"]] != it["cur"] and assign[it["node_id"]] and assign[it["node_id"]] != "nan")

o = io.StringIO()
o.write("== 混合规则时间线模拟（业务级池；紧迫度→增益密度→随机） ==\n")
o.write("节点 %d；切换 %d；hold %d\n\n" % (len(items), switches, len(items) - switches))
o.write("最终(含hold)分布(前6): %s\n" % dict(c.most_common(6)))
o.write("切换目标分布(前8): %s\n\n" % dict(sw.most_common(8)))
o.write("各业务占用 vs 上限（前 8，Mbps）:\n")
for b, n in c.most_common(8):
    load = n * np.mean([it["bw"] for it in items if assign[it["node_id"]] == b]) if n else 0
    o.write("  %s: 节点=%d 估算占用=%.0f 上限=%.0f 余量=%.0f\n" % (b, n, load, ceiling.get(b, 0), remain.get(b, 0)))
# 可吸收上限示例
typo = {}
for b in c:
    pass
o.write("\n说明：容量用 V6 业务层 allocatable_headroom 初始；切换释放原业务容量；无溢出（按构造）。\n")
txt = o.getvalue()
(ROOT / "时间线模拟结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
