# -*- coding: utf-8 -*-
"""平台重做①（新）：从 V3 日账提取**业务级**定价/结算特征（只用测试期之前的窗口，避免泄漏），
合并进训练对，供 V5 平台模型使用（env V5_PLATFORM_FEATURES=1）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）"
p = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"

d = pd.read_csv(p, dtype={"node_id": "string", "business": "string"}, low_memory=False)
d["sample_day"] = pd.to_datetime(d["sample_day"], errors="coerce")
d["business"] = d["business"].astype(str)
cut = d["sample_day"].max() - pd.Timedelta(days=10)   # 留出测试7天+校准3天
hist = d[d["sample_day"] <= cut]

for c in ["customer_unit_price", "miner_unit_price", "customer_measure", "buildBandwidth"]:
    d[c] = pd.to_numeric(d.get(c), errors="coerce")
    hist[c] = pd.to_numeric(hist.get(c), errors="coerce")

price = hist.groupby("business")["customer_unit_price"].median()
ptype = hist.groupby("business")["customer_price_item_name"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
meas = hist.groupby("business")["customer_measure"].median()
supp = hist.groupby("business")["buildBandwidth"].median()


def bucket(x, qs=(0.25, 0.5, 0.75)):
    if pd.isna(x):
        return "unknown"
    return "q3" if x > qs[2] else ("q2" if x > qs[1] else ("q1" if x > qs[0] else "q0"))


biz_price_b = {b: bucket(v) for b, v in price.items()}
biz_meas_b = {b: bucket(v) for b, v in meas.items()}
biz_supp_b = {b: bucket(v) for b, v in supp.items()}

d["biz_price_bucket"] = d["business"].map(biz_price_b).fillna("unknown")
d["biz_price_type"] = d["business"].map(ptype).fillna("unknown").astype(str)
d["biz_measure_bucket"] = d["business"].map(biz_meas_b).fillna("unknown")
d["biz_supply_bucket"] = d["business"].map(biz_supp_b).fillna("unknown")

out = ROOT / "v1_training_pairs_plat（新）.csv"
d.to_csv(out, index=False, encoding="utf-8-sig")
print("wrote", out.name, len(d), "rows; feats:", [c for c in d.columns if c.startswith("biz_")])
print("price_type 分布:", d["biz_price_type"].value_counts().head(5).to_dict())
