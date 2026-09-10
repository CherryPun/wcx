# -*- coding: utf-8 -*-
"""排序层时效重排（新）：对持续掉量业务按其近端量比折减候选综合分，重排 Top1~3。
不改模型训练；仅对 V5 输出的 Top1~3 候选做再排序，便于对比效果。"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"

FLOOR = 0.5          # 折减下限
MIN_BASE_COST = 1000  # 只对近7天成本基数 >= 该值的业务折减（避免小业务噪声）

# --- 掉量因子 ---
d = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv",
                dtype={"node_id": "string", "business": "string"}, low_memory=False)
d["sample_day"] = pd.to_datetime(d["sample_day"], errors="coerce")
d["business"] = d["business"].astype(str)
d["cum_cost_7d"] = pd.to_numeric(d["cum_cost_7d"], errors="coerce")
end = d["sample_day"].max()
last7 = d[d["sample_day"] > end - pd.Timedelta(days=7)]
prev7 = d[(d["sample_day"] > end - pd.Timedelta(days=14)) & (d["sample_day"] <= end - pd.Timedelta(days=7))]
c_last = last7.groupby("business")["cum_cost_7d"].sum()
c_prev = prev7.groupby("business")["cum_cost_7d"].sum()
chg = (c_last / c_prev.reindex(c_last.index) - 1)


def factor_of(biz: str) -> float:
    base = c_last.get(biz, 0.0)
    if base < MIN_BASE_COST or biz not in chg.index:
        return 1.0
    c = chg.get(biz)
    if pd.isna(c) or c >= 0:
        return 1.0
    return max(FLOOR, 1.0 + float(c))


# --- 重排 ---
rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string"}, low_memory=False)
for c in ["combined_score_top1", "combined_score_top2", "combined_score_top3"]:
    rec[c] = pd.to_numeric(rec[c], errors="coerce")
for c in ["business_top1", "business_top2", "business_top3"]:
    rec[c] = rec[c].astype(str)

rows, changes = [], 0
new_counts = {}
for _, r in rec.iterrows():
    cand = [(r["business_top1"], r["combined_score_top1"]),
            (r["business_top2"], r["combined_score_top2"]),
            (r["business_top3"], r["combined_score_top3"])]
    cand = [(b, s) for b, s in cand if isinstance(b, str) and b not in ("", "nan") and pd.notna(s)]
    if not cand:
        continue
    old = cand[0][0]
    best = max(cand, key=lambda x: x[1] * factor_of(x[0]))
    newb = best[0]
    if newb != old:
        changes += 1
    new_counts[newb] = new_counts.get(newb, 0) + 1
    rows.append({"node_id": r["node_id"], "old_top1": old, "new_top1": newb,
                 "old_score": r["combined_score_top1"], "new_score": best[1],
                 "old_factor": round(factor_of(old), 3), "new_factor": round(factor_of(newb), 3)})

res = pd.DataFrame(rows)
res.to_csv(HERE / "时效重排明细（新）.csv", index=False, encoding="utf-8-sig")
new_series = pd.Series(new_counts)

o = io.StringIO()
o.write("== 时效重排（Top1~3 再排序；对掉量业务折减 combined_score） ==\n")
o.write("参数：floor=%.2f, min_base_cost=%.0f\n" % (FLOOR, MIN_BASE_COST))
o.write("节点数 %d；Top1 改变 %d (%.1f%%)\n\n" % (len(res), changes, changes / max(len(res), 1) * 100))
o.write("重排前 Top1 分布(前8): %s\n" % rec["business_top1"].value_counts().head(8).to_dict())
o.write("重排后 Top1 分布(前8): %s\n\n" % new_series.sort_values(ascending=False).head(8).to_dict())
o.write("重点业务 Top1 变化：\n")
for b in ["10000096", "10000224", "10000244", "10000183"]:
    before = int((rec["business_top1"] == b).sum())
    after = int(new_series.get(b, 0))
    o.write("  %s: %d -> %d\n" % (b, before, after))
txt = o.getvalue()
(HERE / "时效重排结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
