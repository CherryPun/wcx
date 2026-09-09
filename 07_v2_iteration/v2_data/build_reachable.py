# -*- coding: utf-8 -*-
"""V2 阶段D：可达量特征集初探（只统计，不预测）。
从近 7 天日95带宽(bw_daily_7d)构每节点可达/利用率候选特征，统计覆盖与分层，供 v3 建模参考。
不改主版文件，输出写本代 _rerun/。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "05_shared_data"
OUT = HERE / "_rerun"

bw = pd.read_csv(DATA / "bw_daily_7d.csv", dtype={"node_id": str, "business": str}, low_memory=False)
bw["business"] = bw["business"].astype(str).str.replace(r"\.0$", "", regex=True)
bw["peak95_mbps"] = pd.to_numeric(bw["peak95"], errors="coerce") / 1e6
feat = bw.groupby("node_id").agg(
    bw_days=("day", "nunique"),
    biz_count=("business", "nunique"),
    peak95_max_mbps=("peak95_mbps", "max"),
    peak95_p50_mbps=("peak95_mbps", "median"),
).reset_index()
feat["days_ratio_7"] = feat["bw_days"] / 7.0
attrs = pd.read_csv(DATA / "nodes_attr_filled.csv", dtype={"node_id": str}, low_memory=False)
attrs["bw"] = pd.to_numeric(attrs.get("bw"), errors="coerce")
feat = feat.merge(attrs[["node_id", "bw"]], on="node_id", how="left")
feat["util_p50"] = feat["peak95_p50_mbps"] / feat["bw"] * 100.0
feat["util_max"] = feat["peak95_max_mbps"] / feat["bw"] * 100.0
feat = feat.replace([float("inf"), float("-inf")], float("nan"))
feat.to_csv(OUT / "reachable_feats.csv", index=False, encoding="utf-8-sig")
print("可达特征节点覆盖:", len(feat), "/", len(attrs.drop_duplicates("node_id")))
ok = feat.dropna(subset=["util_p50", "bw_days"])
print("有bw且可算利用率节点:", len(ok))
print(ok["util_p50"].describe().round(2).to_string())
print("util_p50<5%节点:", int((ok["util_p50"] < 5).sum()),
      "  <20%节点:", int((ok["util_p50"] < 20).sum()))
