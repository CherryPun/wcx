# -*- coding: utf-8 -*-
"""生成 current business（同点/窗口内最近有效日；V3 清洗口径）：
每个大节点取窗口内**最近一个有 V3 干净归属业务**的节点日作为当前业务，并附新鲜度。
输出供 V5 使用的 current_business 文件（列名对齐 v4.load_current_business）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）"
OUT = ROOT / "current_business_同点全量（新）.csv"

d = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv",
                dtype={"node_id": "str", "business": "str"}, low_memory=False)
d["sample_day"] = pd.to_datetime(d["sample_day"], errors="coerce")
for c in ["cum_cost_7d", "cum_revenue_7d"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["sample_day", "business"]).sort_values(["node_id", "sample_day"])
end = d["sample_day"].max()

latest = d.groupby("node_id", as_index=False).tail(1).copy()
latest["days_since"] = (end - latest["sample_day"]).dt.days
# 近 7 天有效天数（该节点在窗口内最后 7 天出现的次数）
d7 = d[d["sample_day"] > end - pd.Timedelta(days=7)]
valid7 = d7.groupby("node_id")["sample_day"].nunique().rename("valid_days_7")
latest = latest.merge(valid7, on="node_id", how="left")
latest["valid_days_7"] = latest["valid_days_7"].fillna(0).astype(int)

out = pd.DataFrame({
    "node_id": latest["node_id"].astype(str),
    "current_business": latest["business"].astype(str),
    "current_business_name": latest["business_name"].fillna("").astype(str),
    "current_business_day": latest["sample_day"].dt.strftime("%Y-%m-%d"),
    "current_business_source": "v3_clean_latest",
    "current_business_confidence": 1,
    "current_business_cost_sum": latest["cum_cost_7d"],
    "current_business_profit_sum": latest["cum_revenue_7d"] - latest["cum_cost_7d"],
    "days_since": latest["days_since"].astype(int),
    "valid_days_7": latest["valid_days_7"].astype(int),
})
out.to_csv(OUT, index=False, encoding="utf-8-sig")

nodes = pd.read_csv(V3 / "current_online_inservice_non_idc_large_nodes_v3.csv", dtype={"node_id": "str"})
cov = nodes["node_id"].isin(set(out["node_id"]))
print("大节点", len(nodes), "覆盖", int(cov.sum()), "未覆盖", int((~cov).sum()))
print("days_since 分布:", out["days_since"].value_counts().sort_index().to_dict())
print("wrote", OUT.name)
