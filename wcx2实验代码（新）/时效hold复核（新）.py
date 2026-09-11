# -*- coding: utf-8 -*-
"""时效 hold 复核（新）：在 V5 时间外测试窗口比较 基线 vs hold 的加权 hit@1/@3 与切换量。
ID 归一：去掉 CSV 浮点残留的 ".0"。"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"


def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


facts = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv",
                    dtype={"node_id": "string", "business": "string"}, low_memory=False)
facts["sample_day"] = pd.to_datetime(facts["sample_day"], errors="coerce")
facts["business"] = norm(facts["business"])
facts["sample_weight"] = pd.to_numeric(facts.get("sample_weight"), errors="coerce").fillna(1.0)
end = facts["sample_day"].max()
test = facts[facts["sample_day"] > end - pd.Timedelta(days=7)].copy()

keep = ["node_id", "business_top1", "business_top2", "business_top3", "current_business"]
base = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string"}, low_memory=False)
hold = pd.read_csv(HERE / "v5_node_recommendations_hold（新）.csv", dtype={"node_id": "string"}, low_memory=False)
base = base[[c for c in keep if c in base.columns]].copy()
hold = hold[[c for c in keep if c in hold.columns]].copy()
for df in (base, hold):
    for c in ["business_top1", "business_top2", "business_top3", "current_business"]:
        if c in df.columns:
            df[c] = norm(df[c])

m = test.merge(base.add_prefix("b_").rename(columns={"b_node_id": "node_id"}), on="node_id", how="left")
m = m.merge(hold.add_prefix("h_").rename(columns={"h_node_id": "node_id"}), on="node_id", how="left")


def hits(prefix):
    t1 = norm(m[prefix + "business_top1"])
    t2 = norm(m.get(prefix + "business_top2", pd.Series([""] * len(m))))
    t3 = norm(m.get(prefix + "business_top3", pd.Series([""] * len(m))))
    obs = norm(m["business"])
    w = m["sample_weight"]
    h1 = (t1 == obs)
    h3 = (t1 == obs) | (t2 == obs) | (t3 == obs)
    return float((h1 * w).sum() / w.sum()), float((h3 * w).sum() / w.sum()), int(h1.sum()), int(h3.sum())


def switches(prefix):
    t1 = norm(m[prefix + "business_top1"])
    return int((t1 != norm(m[prefix + "current_business"])).sum())


b1, b3, bn1, bn3 = hits("b_")
h1, h3, hn1, hn3 = hits("h_")
o = io.StringIO()
o.write("== hold 复核（测试窗口 %s~%s；node-day=%d）==\n" % (
    (end - pd.Timedelta(days=6)).date(), end.date(), len(m)))
o.write("%-8s %10s %10s %10s %10s %10s\n" % ("口径", "hit@1", "hit@3", "命中@1数", "命中@3数", "切换数"))
o.write("%-8s %9.2f%% %9.2f%% %10d %10d %10d\n" % ("基线", b1 * 100, b3 * 100, bn1, bn3, switches("b_")))
o.write("%-8s %9.2f%% %9.2f%% %10d %10d %10d\n" % ("hold", h1 * 100, h3 * 100, hn1, hn3, switches("h_")))
# 只统计参与 hold 的节点（baseline Top1 掉量）
o.write("\n说明：hit=推荐 Top1/Top3 是否等于当日实际跑的干净业务（加权 sample_weight）。\n")
txt = o.getvalue()
(HERE / "时效hold复核（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
