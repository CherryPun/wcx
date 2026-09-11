# -*- coding: utf-8 -*-
"""模型迭代②（新）：为训练对加入"业务量趋势"特征（严格过去式，无泄漏）。
trend = 该业务前7天量 / 前7~14天量（量=capacity_peak95_mbps 求和），离散为桶。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
p = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"

d = pd.read_csv(p, dtype={"node_id": "string", "business": "string"}, low_memory=False)
d["sample_day"] = pd.to_datetime(d["sample_day"], errors="coerce")
d["business"] = d["business"].astype(str)
d["vol"] = pd.to_numeric(d.get("capacity_peak95_mbps"), errors="coerce").fillna(0.0)

bvol = d.groupby(["business", "sample_day"], as_index=False)["vol"].sum()

def bucket(ratio):
    if pd.isna(ratio):
        return "trend_unknown"
    if ratio <= 0.7:
        return "down"
    if ratio <= 1.0:
        return "flat_down"
    if ratio <= 1.3:
        return "flat_up"
    return "up"

trend = {}
for biz, g in bvol.groupby("business"):
    s = g.set_index("sample_day")["vol"].sort_index()
    for day in s.index:
        recent = s[(s.index >= day - pd.Timedelta(days=7)) & (s.index <= day - pd.Timedelta(days=1))].sum()
        prior = s[(s.index >= day - pd.Timedelta(days=14)) & (s.index <= day - pd.Timedelta(days=8))].sum()
        ratio = (recent / prior) if prior > 0 else np.nan
        trend[(biz, day)] = bucket(ratio)

d["business_trend"] = [trend.get((b, t), "trend_unknown") for b, t in zip(d["business"], d["sample_day"])]
out = HERE / "v1_training_pairs_trend（新）.csv"
d.drop(columns=["vol"]).to_csv(out, index=False, encoding="utf-8-sig")
print("rows", len(d), "->", out)
print(d["business_trend"].value_counts().to_dict())
print(d[d.business == "10000244"][["sample_day", "business_trend"]].tail(6).to_string(index=False))
