# -*- coding: utf-8 -*-
"""P2 平台利润改进试点（新）：综合分降权（矿主0.5/0.8/1.0）+ 平台硬闸门（90%下界>0）。
评估：Top1 集中度、Top1 是否容量可行、观察口径命中（v3 真值）。"""
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

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "str"}, low_memory=False)
pool = pd.read_csv(V6 / "v6_capacity_pool_summary.csv", low_memory=False)
pool["business"] = pool["business"].astype(str)
b = pool[pool.pool_level == "business"]
head = {x.business: float(x.allocatable_headroom_mbps) for x in b.itertuples()}

facts = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv", dtype={"node_id": "str", "business": "str"}, low_memory=False)
facts["business"] = facts["business"].astype(str).str.replace(r"\.0$", "", regex=True)
for c in ["cum_cost_7d", "buildBandwidth", "sample_weight"]:
    facts[c] = pd.to_numeric(facts[c], errors="coerce")
facts["mu"] = facts["cum_cost_7d"] / facts["buildBandwidth"].where(facts["buildBandwidth"] > 0)
gb = facts.dropna(subset=["mu"]).groupby(["node_id", "business"], as_index=False).apply(
    lambda g: pd.Series({"mu": np.average(g["mu"], weights=g["sample_weight"].fillna(1.0))}))
best = gb.sort_values("mu").groupby("node_id").tail(1).set_index("node_id")["business"].to_dict()

rows = []
def metrics(choose):
    cnt = Counter(); h1 = h3 = n = feas = 0
    for _, r in rec.iterrows():
        t = choose(r)
        if t is None:
            continue
        n += 1
        cnt[t[0]] += 1
        if head.get(t[0], 0) > 0:
            feas += 1
        bb = best.get(r["node_id"])
        if bb:
            h1 += int(t[0] == bb)
            h3 += int(bb in t)
    tot = sum(cnt.values()) or 1
    if not cnt:
        return n, 0.0, 0.0, 0, 0.0, 0.0, []
    mx = max(cnt.values()) / tot
    hhi = sum((v / tot) ** 2 for v in cnt.values())
    return n, mx, hhi, feas, h1 / max(n, 1), h3 / max(n, 1), cnt.most_common(4)

def cands(r):
    out = []
    for k in (1, 2, 3):
        biz = str(r.get(f"business_top{k}", "")).replace(".0", "")
        mn = pd.to_numeric(r.get(f"miner_unit_income_top{k}"), errors="coerce")
        pf = pd.to_numeric(r.get(f"platform_unit_profit_top{k}"), errors="coerce")
        lo = pd.to_numeric(r.get(f"platform_low90_top{k}"), errors="coerce")
        if biz and biz != "nan":
            out.append((biz, mn, pf, lo))
    return out

# 各节点候选的 min-max 归一（含当前业务）
def norm_triple(r):
    c = cands(r)
    cur = (str(r.get("current_business", "")).replace(".0", ""),
           pd.to_numeric(r.get("current_predicted_miner_unit_income"), errors="coerce"),
           pd.to_numeric(r.get("current_predicted_platform_unit_profit"), errors="coerce"), np.nan)
    allc = c + [cur]
    mn = [x[1] for x in allc if pd.notna(x[1])]; pn = [x[2] for x in allc if pd.notna(x[2])]
    def nz(v, arr):
        if not arr or pd.isna(v):
            return 0.0
        lo, hi = min(arr), max(arr)
        return (v - lo) / (hi - lo) if hi > lo else 0.5
    return [x for x in c], nz, (mn, pn)

def make_chooser(w, gate):
    def choose(r):
        c, nz, (mn, pn) = norm_triple(r)
        scored = []
        for (biz, m, p, lo) in c:
            if gate and not (pd.notna(lo) and lo > 0):
                continue
            s = w * nz(m, mn) + (1 - w) * nz(p, pn)
            scored.append((biz, s))
        if not scored:
            return None
        scored.sort(key=lambda x: -x[1])
        return [x[0] for x in scored]
    return choose

o = io.StringIO()
o.write("== P2 平台降权/闸门试点（矿主权重 w；平台 1-w） ==\n")
o.write("%-16s %6s %8s %8s %8s %8s %8s\n" % ("配置", "n", "max占比", "HHI", "可行", "hit@1", "hit@3"))
for w in (0.5, 0.8, 1.0):
    n, mx, hhi, feas, h1, h3, top = metrics(make_chooser(w, False))
    o.write("%-16s %6d %8.4f %8.4f %8d %8.4f %8.4f\n" % (f"w={w:.1f}", n, mx, hhi, feas, h1, h3))
n, mx, hhi, feas, h1, h3, top = metrics(make_chooser(0.8, True))
o.write("%-16s %6d %8.4f %8.4f %8d %8.4f %8.4f\n" % ("w=0.8+闸门", n, mx, hhi, feas, h1, h3))
o.write("\n（命中=Top1/Top3 命中节点实际单位收益最优业务；观察口径）\n")
txt = o.getvalue()
(ROOT / "P2平台试点（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
