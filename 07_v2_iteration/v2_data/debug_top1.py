# -*- coding: utf-8 -*-
"""debug：在同一 80/20 测试集上逐节点对比 default 与 cut06-15 的 Top1 推荐与真实最优，定位命中翻盘来源。"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "05_shared_data"
OUT = HERE / "_rerun"

o = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
o["business"] = o["business"].astype(str).str.replace(r"\.0$", "", regex=True)
o = o[o["outcome_days"] >= 7]
o["online_day_dt"] = pd.to_datetime(o["online_day"], errors="coerce")
o["is_ant"] = o.node_id.str.startswith("ant")
nf = (o.groupby("node_id")["online_day_dt"].min().reset_index().sort_values("online_day_dt").reset_index(drop=True))
cut = int(len(nf) * 0.8)
test_ids = set(nf.iloc[cut:]["node_id"].astype(str))
pos_te = o[o.node_id.isin(test_ids)].copy()
tb = (pos_te.sort_values("cum_cost_7d").groupby("node_id", as_index=False).tail(1)
      [["node_id", "business"]].rename(columns={"business": "true_best"}))

def load(fn):
    d = pd.read_csv(OUT / fn, dtype={"node_id": str})
    d = d[d.node_id.isin(test_ids)].copy()
    for c in ["top1_business", "top2_business", "top3_business"]:
        d[c] = d[c].astype(str).str.replace(r"\.0$", "", regex=True)
    return d

D = load("final_default.csv").rename(columns={"top1_business": "d_top1", "top2_business": "d_top2", "top3_business": "d_top3"})
C = load("final_cut.csv").rename(columns={"top1_business": "c_top1", "top2_business": "c_top2", "top3_business": "c_top3"})
m = D.merge(C[["node_id", "c_top1", "c_top2", "c_top3"]], on="node_id", how="inner").merge(tb, on="node_id")
m["is_ant"] = m.node_id.str.startswith("ant")
m["d_hit1"] = m.d_top1 == m.true_best
m["c_hit1"] = m.c_top1 == m.true_best
m["d_hit3"] = m[["d_top1", "d_top2", "d_top3"]].eq(m.true_best, axis=0).any(axis=1)
m["c_hit3"] = m[["c_top1", "c_top2", "c_top3"]].eq(m.true_best, axis=0).any(axis=1)
m["same_top1"] = m.d_top1 == m.c_top1

print("共同测试节点:", len(m))
print("default hit1=", round(m.d_hit1.mean(), 4), " cut hit1=", round(m.c_hit1.mean(), 4))
print("default hit3=", round(m.d_hit3.mean(), 4), " cut hit3=", round(m.c_hit3.mean(), 4))
print("top1 与 default 不同节点数:", int((~m.same_top1).sum()), "/", len(m))
comb = pd.crosstab(m.d_hit1, m.c_hit1)
print("hit1 交叉(default 行 x cut 列, False/True):")
print(comb)
cols = ["node_id", "is_ant", "true_best", "d_top1", "d_hit1", "c_top1", "c_hit1", "same_top1"]
m[cols].to_csv(OUT / "debug_top1_diff.csv", index=False, encoding="utf-8-sig")
# 仅 default 对而 cut 错 的节点，看 true_best 类型与是否 ant
only_d = m[m.d_hit1 & ~m.c_hit1]
only_c = m[~m.d_hit1 & m.c_hit1]
print("仅default对:", len(only_d), " 其中ant", int(only_d.is_ant.sum()), " 仅cut对:", len(only_c), " 其中ant", int(only_c.is_ant.sum()))
print("wrote", OUT / "debug_top1_diff.csv")
