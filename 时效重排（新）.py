# -*- coding: utf-8 -*-
"""排序层时效重排（新，温和版）：对显著掉量业务折减候选综合分并重排 Top1~3，
输出可供 V6 直接使用的推荐文件（business_topX / combined_score_topX 重排）。"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"

FLOOR = 0.85           # 折减下限（温和）
THRESHOLD = -0.20      # 仅对 cost_chg <= -20% 的业务折减
MIN_BASE_COST = 1000

d = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv",
                dtype={"node_id": "string", "business": "string"}, low_memory=False)
d["sample_day"] = pd.to_datetime(d["sample_day"], errors="coerce")
d["business"] = d["business"].astype(str)
d["cum_cost_7d"] = pd.to_numeric(d["cum_cost_7d"], errors="coerce")
end = d["sample_day"].max()
c_last = d[d["sample_day"] > end - pd.Timedelta(days=7)].groupby("business")["cum_cost_7d"].sum()
c_prev = d[(d["sample_day"] > end - pd.Timedelta(days=14)) & (d["sample_day"] <= end - pd.Timedelta(days=7))].groupby("business")["cum_cost_7d"].sum()
chg = c_last / c_prev.reindex(c_last.index) - 1


def factor_of(biz: str) -> float:
    base = c_last.get(biz, 0.0)
    if base < MIN_BASE_COST or biz not in chg.index:
        return 1.0
    c = chg.get(biz)
    if pd.isna(c) or c > THRESHOLD:
        return 1.0
    return max(FLOOR, 1.0 + float(c))


rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string"}, low_memory=False)
for c in ["combined_score_top1", "combined_score_top2", "combined_score_top3"]:
    rec[c] = pd.to_numeric(rec[c], errors="coerce")
for c in ["business_top1", "business_top2", "business_top3"]:
    rec[c] = rec[c].astype(str)
for c in ["business_name_top1", "business_name_top2", "business_name_top3"]:
    if c in rec.columns:
        rec[c] = rec[c].fillna("").astype(str)

out = rec.copy()
rows = []
changes = 0
new_counts = {}
for i, r in rec.iterrows():
    cand = []
    for k in (1, 2, 3):
        b = r[f"business_top{k}"]
        s = r[f"combined_score_top{k}"]
        nm = r.get(f"business_name_top{k}", "")
        if isinstance(b, str) and b not in ("", "nan") and pd.notna(s):
            cand.append([b, s, nm])
    if not cand:
        continue
    old = cand[0][0]
    cand.sort(key=lambda x: x[1] * factor_of(x[0]), reverse=True)
    for k, (b, s, nm) in enumerate(cand, 1):
        out.at[i, f"business_top{k}"] = b
        out.at[i, f"combined_score_top{k}"] = s
        if f"business_name_top{k}" in out.columns:
            out.at[i, f"business_name_top{k}"] = nm
    if cand[0][0] != old:
        changes += 1
    new_counts[cand[0][0]] = new_counts.get(cand[0][0], 0) + 1
    rows.append({"node_id": r["node_id"], "old_top1": old, "new_top1": cand[0][0],
                 "old_factor": round(factor_of(old), 3), "new_factor": round(factor_of(cand[0][0]), 3)})

path = HERE / "v5_node_recommendations_rerank（新）.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(HERE / "时效重排明细（新）.csv", index=False, encoding="utf-8-sig")
new_series = pd.Series(new_counts)

o = io.StringIO()
o.write("== 时效重排（温和版） ==\n")
o.write("参数：FLOOR=%.2f, 阈值 cost_chg<=%.0f%%, min_base_cost=%d\n" % (FLOOR, THRESHOLD * 100, MIN_BASE_COST))
o.write("节点数 %d；Top1 改变 %d (%.1f%%)\n" % (len(rows), changes, changes / max(len(rows), 1) * 100))
o.write("重排前 Top1 分布(前8): %s\n" % rec["business_top1"].value_counts().head(8).to_dict())
o.write("重排后 Top1 分布(前8): %s\n" % new_series.sort_values(ascending=False).head(8).to_dict())
o.write("输出推荐文件: %s\n" % path.name)
txt = o.getvalue()
(HERE / "时效重排结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
