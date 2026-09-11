# -*- coding: utf-8 -*-
"""时效 hold 重排（新）：若 V5 Top1 是显著掉量业务（cost_chg<=-20% 且基数>=1000），
则维持当前业务（不切换）；否则保持原 Top1。输出 V6 可直接使用的推荐文件。"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"
THRESHOLD, MIN_BASE_COST = -0.20, 1000

d = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv",
                dtype={"node_id": "string", "business": "string"}, low_memory=False)
d["sample_day"] = pd.to_datetime(d["sample_day"], errors="coerce")
d["business"] = d["business"].astype(str)
d["cum_cost_7d"] = pd.to_numeric(d["cum_cost_7d"], errors="coerce")
end = d["sample_day"].max()
c_last = d[d["sample_day"] > end - pd.Timedelta(days=7)].groupby("business")["cum_cost_7d"].sum()
c_prev = d[(d["sample_day"] > end - pd.Timedelta(days=14)) & (d["sample_day"] <= end - pd.Timedelta(days=7))].groupby("business")["cum_cost_7d"].sum()
chg = c_last / c_prev.reindex(c_last.index) - 1


def declining(biz: str) -> bool:
    if c_last.get(biz, 0.0) < MIN_BASE_COST or biz not in chg.index:
        return False
    c = chg.get(biz)
    return (not pd.isna(c)) and c <= THRESHOLD


rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string"}, low_memory=False)
for c in ["business_top1", "business_top2", "business_top3", "current_business"]:
    rec[c] = rec[c].astype(str)
for c in ["combined_score_top1", "combined_score_top2", "combined_score_top3", "current_predicted_combined_score"]:
    if c in rec.columns:
        rec[c] = pd.to_numeric(rec[c], errors="coerce")

out = rec.copy()
rows, held = [], 0
new_counts = {}
for i, r in rec.iterrows():
    t1 = r["business_top1"]
    cur = r["current_business"]
    if declining(t1) and isinstance(cur, str) and cur not in ("", "nan"):
        out.at[i, "business_top1"] = cur
        if "business_name_top1" in out.columns:
            out.at[i, "business_name_top1"] = r.get("current_business_name", "")
        if "combined_score_top1" in out.columns and pd.notna(r.get("current_predicted_combined_score")):
            out.at[i, "combined_score_top1"] = r["current_predicted_combined_score"]
        held += 1
        new = cur
    else:
        new = t1
    new_counts[new] = new_counts.get(new, 0) + 1
    rows.append({"node_id": r["node_id"], "old_top1": t1, "new_top1": new,
                 "held": int(new != t1 and declining(t1))})

path = HERE / "v5_node_recommendations_hold（新）.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(HERE / "时效hold明细（新）.csv", index=False, encoding="utf-8-sig")
new_series = pd.Series(new_counts)

o = io.StringIO()
o.write("== 时效 hold（掉量 Top1 -> 维持当前业务） ==\n")
o.write("阈值 cost_chg<=%.0f%%, 基数>=%d；hold 节点 %d / %d (%.1f%%)\n"
        % (THRESHOLD * 100, MIN_BASE_COST, held, len(rows), held / max(len(rows), 1) * 100))
o.write("原 Top1 分布(前8): %s\n" % rec["business_top1"].value_counts().head(8).to_dict())
o.write("hold 后 Top1 分布(前8): %s\n" % new_series.sort_values(ascending=False).head(8).to_dict())
txt = o.getvalue()
(HERE / "时效hold结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
