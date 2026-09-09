# -*- coding: utf-8 -*-
"""V2.1-1(unit 口径可行性)：在 buildBandwidth 当日覆盖节点子集上，
对比“绝对结算最优”与“单位收益最优(cost/buildBandwidth)”两个 oracle 是否一致，
判断单位口径是否值得全量建模。输出本代 _rerun/，不改主版。"""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "05_shared_data"
OUT = HERE / "_rerun"

# node 级建设带宽（当日覆盖子集）
frames = []
for p in sorted(glob.glob(str(OUT / "nc_*.csv"))):
    frames.append(pd.read_csv(p, dtype={"nodeId": str}))
cap = pd.concat(frames, ignore_index=True)
cap["buildBandwidth"] = pd.to_numeric(cap["buildBandwidth"], errors="coerce")
cap = cap.dropna(subset=["buildBandwidth"])
cap = (cap.groupby("nodeId")["buildBandwidth"].max().reset_index()
       .rename(columns={"buildBandwidth": "cap_mbps"}))
print("buildBandwidth 覆盖节点:", len(cap))

o = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
o["business"] = o["business"].astype(str).str.replace(r"\.0$", "", regex=True)
o = o[o["outcome_days"] >= 7]
o = o.merge(cap, left_on="node_id", right_on="nodeId", how="inner")
o["unit_cost"] = o["cum_cost_7d"] / o["cap_mbps"]
o["is_ant"] = o.node_id.str.startswith("ant")
print("可参与单位口径账行:", len(o), " 节点:", o.node_id.nunique())

rows = []
for (nid, grp) in o.groupby("node_id"):
    ab = grp.sort_values("cum_cost_7d").iloc[-1]
    ub = grp.sort_values("unit_cost").iloc[-1]
    rows.append({"node_id": nid, "is_ant": grp.is_ant.iloc[0], "n_biz": len(grp),
                 "abs_best": ab.business, "unit_best": ub.business,
                 "same": ab.business == ub.business,
                 "abs_cost_abs_best": float(ab.cum_cost_7d), "abs_cost_unit_best": float(ub.cum_cost_7d),
                 "unit_abs_best": float(ab.unit_cost), "unit_unit_best": float(ub.unit_cost)})
r = pd.DataFrame(rows)
r.to_csv(OUT / "unit_feas_nodes.csv", index=False, encoding="utf-8-sig")
print("== 两口径最优是否一致(节点级) ==")
for name, g in [("全部", r), ("ant", r[r.is_ant]), ("IDC/非ant", r[~r.is_ant])]:
    diff = g[~g.same]
    if len(g):
        print(f"{name}: 节点={len(g)} 两口径最优相同率={round(g.same.mean()*100,1)}%  不同={len(diff)}")
        if len(diff):
            up = (diff.unit_unit_best / diff.unit_abs_best.replace(0, float("nan")) - 1)
            print(f"   不同节点中：换单位最优相对绝对最优的单位收益提升中位={round(up.median()*100,1)}%"
                  f"   绝对结算降低中位={round(((diff.abs_cost_unit_best/diff.abs_cost_abs_best.replace(0,float('nan')))-1).median()*100,1)}%")
