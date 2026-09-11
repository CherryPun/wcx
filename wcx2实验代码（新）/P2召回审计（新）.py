# -*- coding: utf-8 -*-
"""召回审计（新）：真优业务是否在"候选集/推荐Top3"内。
真优=节点窗口内实际跑过业务中单位收益最高者（观察性）。"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"


def norm(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


f = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv", dtype={"node_id": "string", "business": "string"}, low_memory=False)
f["business"] = norm(f["business"])
for c in ["cum_cost_7d", "buildBandwidth", "sample_weight"]:
    f[c] = pd.to_numeric(f.get(c), errors="coerce")
f["unit"] = f["cum_cost_7d"] / f["buildBandwidth"].where(f["buildBandwidth"] > 0)
g = f.dropna(subset=["unit"]).groupby(["node_id", "business"], as_index=False).apply(
    lambda x: pd.Series({"unit": np.average(x["unit"], weights=x["sample_weight"].fillna(1.0))}))
best = g.sort_values("unit").groupby("node_id").tail(1).set_index("node_id")["business"].to_dict()

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string"}, low_memory=False)
for c in ["business_top1", "business_top2", "business_top3"]:
    rec[c] = norm(rec[c])

# 候选集：取自模型元数据（若无则用主流 allowlist）
cand = set()
mp = V5 / "v5_hybrid_model.json"
if mp.exists():
    m = json.loads(mp.read_text(encoding="utf-8"))
    meta = m.get("metadata", m)
    cand = set(str(x) for x in meta.get("candidate_businesses", []))
allow = HERE / "mainstream_business_allowlist.csv"
if not cand and allow.exists():
    cand = set(norm(pd.read_csv(allow, dtype=str)["business"]))

n = in_cand = in_top1 = in_top3 = 0
miss = {}
for _, r in rec.iterrows():
    b = best.get(r["node_id"])
    if b is None:
        continue
    n += 1
    if b in cand:
        in_cand += 1
    t1, t2, t3 = r["business_top1"], r["business_top2"], r["business_top3"]
    if b == t1:
        in_top1 += 1
    if b in {t1, t2, t3}:
        in_top3 += 1
    if b not in cand:
        miss[b] = miss.get(b, 0) + 1

o = io.StringIO()
o.write("== 召回审计（node n=%d, 候选集大小=%d） ==\n" % (n, len(cand)))
o.write("真优在候选集内: %.2f%%\n" % (in_cand / max(n, 1) * 100))
o.write("真优=Top1: %.2f%%；真优在Top3: %.2f%%\n" % (in_top1 / max(n, 1) * 100, in_top3 / max(n, 1) * 100))
o.write("\n真优不在候选集的分布(前10): %s\n" % dict(sorted(miss.items(), key=lambda kv: -kv[1])[:10]))
txt = o.getvalue()
(HERE / "P2召回审计（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
