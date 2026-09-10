# -*- coding: utf-8 -*-
"""P2 节点级 observed regret（新）：
以"节点在整个窗口内实际跑过的业务集合"为参考，评估 V5/hold/M1M2 的
Top1/Top3 覆盖与 regret（相对该节点观测到的最优业务）。观察性度量，非因果。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"


def norm(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


f = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv",
                dtype={"node_id": "string", "business": "string"}, low_memory=False)
f["business"] = norm(f["business"])
for c in ["cum_cost_7d", "cum_revenue_7d", "buildBandwidth", "sample_weight"]:
    f[c] = pd.to_numeric(f.get(c), errors="coerce")
f["bw"] = f["buildBandwidth"].where(f["buildBandwidth"] > 0)
f["unit_miner"] = f["cum_cost_7d"] / f["bw"]
# 节点×业务 观测单位收益（加权）
grp = f.dropna(subset=["unit_miner"]).groupby(["node_id", "business"], as_index=False).apply(
    lambda g: pd.Series({
        "unit": np.average(g["unit_miner"], weights=g["sample_weight"].fillna(1.0)),
        "w": g["sample_weight"].fillna(1.0).sum(),
    })
)
node_obs = {nid: g.set_index("business")[["unit", "w"]] for nid, g in grp.groupby("node_id")}

recs = {
    "V5基线": V5 / "v5_node_recommendations.csv",
    "hold": HERE / "v5_node_recommendations_hold（新）.csv",
    "M1M2": HERE / "v5_node_recommendations_m1m2（新）.csv",
}
o = io.StringIO()
o.write("%-10s %8s %8s %8s %10s %10s\n" % ("口径", "hit@1", "hit@3", "覆盖", "regret", "n节点"))
for tag, p in recs.items():
    d = pd.read_csv(p, dtype={"node_id": "string"}, low_memory=False)
    for c in ["business_top1", "business_top2", "business_top3"]:
        d[c] = norm(d.get(c, ""))
    h1 = h3 = cov = 0.0
    reg, regw = [], []
    n = 0
    for _, r in d.iterrows():
        obs = node_obs.get(r["node_id"])
        if obs is None or obs.empty:
            continue
        n += 1
        w = float(obs["w"].sum())
        t1, t2, t3 = r["business_top1"], r["business_top2"], r["business_top3"]
        best = float(obs["unit"].max())
        h1 += float(t1 in obs.index) * w
        h3 += float((t1 in obs.index) or (t2 in obs.index) or (t3 in obs.index)) * w
        if t1 in obs.index:
            cov += w
            sel = float(obs.loc[t1, "unit"])
            reg.append(max(0.0, (best - sel) / max(abs(best), 1e-9)))
            regw.append(w)
    tot = max(sum(float(node_obs[r.node_id]["w"].sum()) for r in d.itertuples() if r.node_id in node_obs), 1e-9)
    o.write("%-10s %7.2f%% %7.2f%% %7.2f%% %9.3f %10d\n" % (
        tag, h1 / tot * 100, h3 / tot * 100, cov / tot * 100,
        float(np.average(reg, weights=regw)) if reg else float("nan"), n))
o.write("\n说明：参考集=该节点窗口内实际跑过业务的加权单位收益；regret=Top1落在参考集时的相对损失；\n")
o.write("      覆盖=Top1命中参考集的比例（未覆盖不计 regret）。观察性度量，非因果。\n")
txt = o.getvalue()
(HERE / "P2节点级regret（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
