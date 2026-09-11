# -*- coding: utf-8 -*-
"""路线1（新）：迭代式拥堵定价原型（离线，不改 solver）。
思路：用池利用率对候选分做折减，分配→重算负载→再折减，迭代到稳定；
对比"无定价（全切 Top1）"的溢出、以及一次折减的溢出。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "recent_month_large_mainstream_v5_cap20_curfull（新）"
V6 = ROOT / "recent_month_large_mainstream_v6_capacity_cap20_curfull（新）"
T = 5

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "str"}, low_memory=False)
nr = pd.read_csv(V6 / "v6_capacity_node_report.csv", dtype={"node_id": "str"}, low_memory=False)
cap = pd.read_csv(V6 / "v6_business_capacity_summary.csv", low_memory=False)
cap["business"] = cap["business"].astype(str)
ceil = dict(zip(cap["business"], pd.to_numeric(cap["raw_capacity_ceiling_mbps"], errors="coerce")))

bw = dict(zip(nr["node_id"].astype(str), pd.to_numeric(nr["build_bandwidth_mbps"], errors="coerce").fillna(1000.0)))
rec = rec[rec["node_id"].astype(str).isin(bw)].copy()

def cands(r):
    out = []
    for k in (1, 2, 3):
        b = str(r.get(f"business_top{k}", ""))
        s = pd.to_numeric(r.get(f"combined_score_top{k}"), errors="coerce")
        if b and b != "nan" and pd.notna(s):
            out.append((b, float(s)))
    return out

nodes = []
for _, r in rec.iterrows():
    c = cands(r)
    if not c:
        continue
    cur = str(r.get("current_business", ""))
    cs = pd.to_numeric(r.get("current_predicted_combined_score"), errors="coerce")
    if cur and cur != "nan" and pd.notna(cs):
        c.append((cur, float(cs)))
    nodes.append((str(r["node_id"]), c))

bwv = [bw[nid] for nid, _ in nodes]

def assign(scores, loads):
    pick = []
    for c in scores:
        best, bs = None, -1e18
        for b, s in c:
            l = loads.get(b, 0.0)
            cl = ceil.get(b)
            pen = 1.0 if (cl is None or cl <= 0) else min(1.0, cl / max(l, 1.0))
            v = s * pen
            if v > bs:
                bs, best = v, b
        pick.append(best)
    return pick

def loads_of(pick):
    lo = {}
    for (nid, _), b in zip(nodes, pick):
        if b:
            lo[b] = lo.get(b, 0.0) + bw[nid]
    return lo

def overflow(lo):
    tot = 0.0
    bizs = 0
    for b, l in lo.items():
        cl = ceil.get(b)
        if cl and l > cl:
            tot += l - cl
            bizs += 1
    return tot / 1000.0, bizs

scores = [c for _, c in nodes]
trace = []
pick0 = assign(scores, {})
trace.append(("无定价",) + overflow(loads_of(pick0)))
# 迭代定价（阻尼：负载 EMA，缓解振荡）
ALPHA = 0.3
lo = {}
for t in range(30):
    pick = assign(scores, lo)
    lo_new = loads_of(pick)
    keys = set(lo) | set(lo_new)
    lo = {k: (1 - ALPHA) * lo.get(k, 0.0) + ALPHA * lo_new.get(k, 0.0) for k in keys}
    if t < 10 or t % 5 == 0:
        o2, nb2 = overflow(lo_new)
        trace.append((f"迭代{t+1}", o2, nb2))

o = io.StringIO()
o.write("== 迭代拥堵定价（业务层池，T=%d） ==\n" % T)
o.write("%-8s %12s %8s\n" % ("轮次", "溢出Gbps", "溢出业务"))
for name, ov, nb in trace:
    o.write("%-8s %12.0f %8d\n" % (name, ov, nb))
# 最终分布
final_pick = assign(scores, lo)
from collections import Counter
c = Counter(b for b in final_pick if b)
o.write("\n最终 Top1 分布(前8): %s\n" % dict(c.most_common(8)))
txt = o.getvalue()
(ROOT / "迭代拥堵定价结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
