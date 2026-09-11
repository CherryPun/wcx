# -*- coding: utf-8 -*-
"""排序错因分桶（新）：
- 两种真值口径：A=观察性单位收益最优；B=模型目标(0.5矿主+0.5平台 单位收益)最优；
- 统计 Top1/Top3 命中、未命中错因（被谁挤掉/是否高带宽/是否超容业务）；
- 剔除超容业务后再算 Top1/Top3（量化容量约束对命中的影响）。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"
OVER = {"1382680008", "10000224", "10000096", "10000041"}


def norm(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


f = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv", dtype={"node_id": "string", "business": "string"}, low_memory=False)
f["business"] = norm(f["business"])
for c in ["cum_cost_7d", "cum_revenue_7d", "buildBandwidth", "sample_weight"]:
    f[c] = pd.to_numeric(f.get(c), errors="coerce")
f["bw"] = f["buildBandwidth"].where(f["buildBandwidth"] > 0)
f["miner_u"] = f["cum_cost_7d"] / f["bw"]
f["plat_u"] = (f["cum_revenue_7d"] - f["cum_cost_7d"]) / f["bw"]
f = f.dropna(subset=["miner_u"])

# node×business 加权单位收益
def agg(g):
    w = g["sample_weight"].fillna(1.0)
    return pd.Series({
        "miner_u": np.average(g["miner_u"], weights=w),
        "plat_u": np.average(g["plat_u"].fillna(0), weights=w),
        "bw": g["bw"].median(),
        "w": w.sum(),
    })

nb = f.groupby(["node_id", "business"], as_index=False).apply(agg)
obs = {nid: g.set_index("business") for nid, g in nb.groupby("node_id")}

# 模型真值口径 B：节点内 min-max 归一后 0.5/0.5
def best_A_B(g):
    gi = g.set_index("business")
    a = gi["miner_u"].idxmax()
    mn, mx = gi["miner_u"].min(), gi["miner_u"].max()
    pmn, pmx = gi["plat_u"].min(), gi["plat_u"].max()
    mscore = (gi["miner_u"] - mn) / (mx - mn) if mx > mn else pd.Series(0.0, index=gi.index)
    pscore = (gi["plat_u"] - pmn) / (pmx - pmn) if pmx > pmn else pd.Series(0.0, index=gi.index)
    comb = 0.5 * mscore + 0.5 * pscore
    b = comb.idxmax()
    return pd.Series({"best_A": a, "best_B": b})

bb = nb.groupby("node_id").apply(best_A_B)
bestA = bb["best_A"].to_dict()
bestB = bb["best_B"].to_dict()

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string"}, low_memory=False)
for c in ["business_top1", "business_top2", "business_top3"]:
    rec[c] = norm(rec[c])

stats = Counter()
miss_displacer = Counter()
miss_truebest = Counter()
miss_bigbw = 0
miss_over = 0
n = 0
h1A = h3A = h1B = h3B = 0
h1A_c = h3A_c = 0
for _, r in rec.iterrows():
    g = obs.get(r["node_id"])
    if g is None or g.empty:
        continue
    n += 1
    t = [r["business_top1"], r["business_top2"], r["business_top3"]]
    a = bestA.get(r["node_id"]); b = bestB.get(r["node_id"])
    if a:
        h1A += int(t[0] == a); h3A += int(a in t)
    if b:
        h1B += int(t[0] == b); h3B += int(b in t)
    # 剔除超容业务后的 Top1/Top3（A 口径）
    tc = [x for x in t if x not in OVER]
    t1c = tc[0] if tc else ""
    if a:
        h1A_c += int(t1c == a); h3A_c += int(a in tc)
    if a and a not in t:
        stats["missA"] += 1
        miss_displacer[t[0]] += 1
        miss_truebest[a] += 1
        if a in g.index and t[0] in g.index:
            if g.loc[t[0], "bw"] > g.loc[a, "bw"]:
                miss_bigbw += 1
        if t[0] in OVER:
            miss_over += 1

o = []
o.append("== 排序错因分桶（n=%d） ==" % n)
o.append("真值A(观察单位收益) : Top1=%.2f%% Top3=%.2f%%" % (h1A / n * 100, h3A / n * 100))
o.append("真值B(模型0.5/0.5)  : Top1=%.2f%% Top3=%.2f%%" % (h1B / n * 100, h3B / n * 100))
o.append("剔除超容业务后(A)   : Top1=%.2f%% Top3=%.2f%%" % (h1A_c / n * 100, h3A_c / n * 100))
o.append("")
o.append("未命中A n=%d；被挤掉Top1分布(前8): %s" % (stats["missA"], dict(miss_displacer.most_common(8))))
o.append("未命中A的真优分布(前8): %s" % dict(miss_truebest.most_common(8)))
o.append("未命中中 挤掉者带宽更大占比: %.1f%%" % (miss_bigbw / max(stats["missA"], 1) * 100))
o.append("未命中中 挤掉者为超容业务占比: %.1f%%" % (miss_over / max(stats["missA"], 1) * 100))
txt = "\n".join(o)
(HERE / "P2错因分桶（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
