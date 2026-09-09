# -*- coding: utf-8 -*-
"""V2.1-1：smallBox(ant)/IDC 分层评估（时间外 80/20）。
独立复刻 stage_e2e 的切分与“真实最优”标签，按设备类型分层统计 hit1/hit3。
不修改主版；输出写本代 _rerun/。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "05_shared_data"
OUT = HERE / "_rerun"

attrs = pd.read_csv(DATA / "nodes_attr_filled.csv", dtype={"node_id": str}, low_memory=False).drop_duplicates("node_id")
o = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
o["business"] = o["business"].astype(str).str.replace(r"\.0$", "", regex=True)
o = o[o["outcome_days"] >= 7]
o["cum_profit_7d"] = o["cum_revenue_7d"] - o["cum_cost_7d"]
o["online_day_dt"] = pd.to_datetime(o["online_day"], errors="coerce")
frame = o.merge(attrs[["node_id"]], on="node_id", how="inner")

node_first = (frame.groupby("node_id")["online_day_dt"].min().reset_index()
              .sort_values("online_day_dt").reset_index(drop=True))
cut = int(len(node_first) * 0.8)
test_ids = set(node_first.iloc[cut:]["node_id"].astype(str))
pos_te = frame[frame.node_id.isin(test_ids)].copy()
# 真实最优 = 测试节点账中 cum_cost_7d 最大业务（与 stage_e2e 口径一致）
true_best = (pos_te.sort_values("cum_cost_7d")
             .groupby("node_id", as_index=False).tail(1)[["node_id", "business"]]
             .rename(columns={"business": "true_best"}))

rec = pd.read_csv(OUT / "final_top3_e2e.csv", dtype={"node_id": str})
rec = rec[rec.node_id.isin(test_ids)].copy()
for c in ["top1_business", "top2_business", "top3_business"]:
    rec[c] = rec[c].astype(str).str.replace(r"\.0$", "", regex=True)
rec = rec.merge(true_best, on="node_id", how="left")
rec["is_ant"] = rec.node_id.str.startswith("ant")
rec["hit1"] = rec["top1_business"] == rec["true_best"]
rec["hit3"] = rec[["top1_business", "top2_business", "top3_business"]].eq(rec["true_best"], axis=0).any(axis=1)

rows = []
for name, g in [("全部", rec), ("ant/smallBox", rec[rec.is_ant]), ("IDC/非ant", rec[~rec.is_ant])]:
    rows.append({"组": name, "测试节点": len(g), "占比%": round(len(g) / len(rec) * 100, 1),
                 "hit1%": round(g.hit1.mean() * 100, 2), "hit3%": round(g.hit3.mean() * 100, 2)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(OUT / "v21_layer_eval.csv", index=False, encoding="utf-8-sig")
print("wrote", OUT / "v21_layer_eval.csv")
