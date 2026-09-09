# -*- coding: utf-8 -*-
"""V2.1-1(unit 可行性·修正)：以 per(node,customer) buildBandwidth 为分母，
对比绝对结算最优与单位收益最优(cost/band)是否一致；覆盖按 pair 计算。"""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "05_shared_data"
OUT = HERE / "_rerun"

frames = []
for p in sorted(glob.glob(str(OUT / "cb_*.csv"))):
    frames.append(pd.read_csv(p, dtype={"nodeId": str, "customerId": str}))
cap = pd.concat(frames, ignore_index=True)
cap["buildBandwidth"] = pd.to_numeric(cap["buildBandwidth"], errors="coerce")
cap = cap.dropna(subset=["buildBandwidth"])
cap = cap.rename(columns={"customerId": "business"})
cap["business"] = cap["business"].astype(str).str.replace(r"\.0$", "", regex=True)
pair = (cap.groupby(["nodeId", "business"])["buildBandwidth"].median().reset_index()
        .rename(columns={"nodeId": "node_id", "buildBandwidth": "cap_bw"}))
print("per(node,business) cap 对:", len(pair), " 覆盖 node:", pair.node_id.nunique())

o = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
o["business"] = o["business"].astype(str).str.replace(r"\.0$", "", regex=True)
o = o[o["outcome_days"] >= 7]
m = o.merge(pair, on=["node_id", "business"], how="inner")
print("有 cap 的账行:", len(m), "/", len(o), " 行覆盖率=", round(len(m)/len(o)*100, 1), "%")
print("有 cap 的节点:", m.node_id.nunique(), " 其中 ant", int(m.node_id.str.startswith("ant").sum()))
m["unit_cost"] = m["cum_cost_7d"] / m["cap_bw"]
m["is_ant"] = m.node_id.str.startswith("ant")

rows = []
for (nid, grp) in m.groupby("node_id"):
    ab = grp.sort_values("cum_cost_7d").iloc[-1]
    ub = grp.sort_values("unit_cost").iloc[-1]
    rows.append({"node_id": nid, "is_ant": grp.is_ant.iloc[0], "n_biz": len(grp),
                 "abs_best": ab.business, "unit_best": ub.business, "same": ab.business == ub.business,
                 "abs_cost_abs": float(ab.cum_cost_7d), "abs_cost_unit": float(ub.cum_cost_7d),
                 "unit_abs": float(ab.unit_cost), "unit_unit": float(ub.unit_cost)})
r = pd.DataFrame(rows)
r.to_csv(OUT / "unit_feas_nodes2.csv", index=False, encoding="utf-8-sig")
print("== 两口径最优是否一致(有 cap 节点) ==")
for name, g in [("全部", r), ("ant", r[r.is_ant]), ("IDC/非ant", r[~r.is_ant])]:
    if not len(g):
        continue
    diff = g[~g.same]
    print(f"{name}: 节点={len(g)} 相同率={round(g.same.mean()*100,1)}% 不同={len(diff)}")
    if len(diff):
        up = (diff.unit_unit / diff.unit_abs.replace(0, float("nan")) - 1)
        dec = (diff.abs_cost_unit / diff.abs_cost_abs.replace(0, float("nan")) - 1)
        print(f"   不同时：单位收益提升中位={round(up.median()*100,1)}%  绝对结算下降中位={round(dec.median()*100,1)}%")
