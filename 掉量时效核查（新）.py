# -*- coding: utf-8 -*-
"""排序层掉量时效核查（新）：按业务计算近 7 天 vs 前 7 天的量与成本变化，
并对照 V5 Top1 推荐暴露度。不改模型，只做诊断。"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"

d = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv",
                dtype={"node_id": "string", "business": "string"}, low_memory=False)
d["sample_day"] = pd.to_datetime(d["sample_day"], errors="coerce")
d["business"] = d["business"].astype(str)
for c in ["cum_cost_7d", "cum_revenue_7d", "capacity_peak95_mbps", "buildBandwidth"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

end = d["sample_day"].max()
last7 = d[d["sample_day"] > end - pd.Timedelta(days=7)]
prev7 = d[(d["sample_day"] > end - pd.Timedelta(days=14)) & (d["sample_day"] <= end - pd.Timedelta(days=7))]
peak_day = d.groupby("sample_day")["cum_cost_7d"].sum().idxmax()
peak_biz = d[d["sample_day"] == peak_day].groupby("business")["cum_cost_7d"].sum()

agg = lambda x: pd.Series({
    "cost_last7": x["cum_cost_7d"].sum(),
    "peak_last7": x["capacity_peak95_mbps"].sum(),
    "nodes_last7": x["node_id"].nunique(),
})
a = last7.groupby("business").apply(agg)
a.columns = ["cost_last7", "peak_last7", "nodes_last7"]
b = prev7.groupby("business").apply(agg)
b.columns = ["cost_prev7", "peak_prev7", "nodes_prev7"]
t = a.join(b, how="left")
t["cost_delta"] = t["cost_last7"] - t["cost_prev7"]
t["cost_chg"] = t["cost_last7"] / t["cost_prev7"] - 1
t["peak_chg"] = t["peak_last7"] / t["peak_prev7"] - 1
t["peak_vs_peakday"] = t["peak_last7"] / peak_biz.reindex(t.index)

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string", "business_top1": "string"}, low_memory=False)
top1 = rec["business_top1"].astype(str).value_counts().rename("v5_top1_nodes")
t = t.join(top1, how="left")
t["v5_top1_nodes"] = t["v5_top1_nodes"].fillna(0).astype(int)

names = d.dropna(subset=["business"]).drop_duplicates("business").set_index("business")["business_name"].to_dict()
t["name"] = t.index.map(lambda x: names.get(str(x), ""))

o = io.StringIO()
o.write("窗口 %s ~ %s\n" % (d.sample_day.min().date(), end.date()))
o.write("近7天 = %s~%s；前7天 = %s~%s\n\n"
        % ((end - pd.Timedelta(days=6)).date(), end.date(),
           (end - pd.Timedelta(days=13)).date(), (end - pd.Timedelta(days=7)).date()))
o.write("== 掉量最大的业务（按近7天成本基数排序，cost_chg<0） ==\n")
dec = t[t["cost_chg"] < 0].sort_values("cost_last7", ascending=False).head(15)
o.write("%-12s %-22s %10s %8s %8s %8s %8s\n" % ("business", "name", "cost_last7", "cost_chg", "peak_chg", "vs峰值日", "V5Top1"))
for biz, r in dec.iterrows():
    o.write("%-12s %-22s %10.0f %+7.1f%% %+7.1f%% %7.1f%% %8d\n"
            % (biz, str(r["name"])[:20], r["cost_last7"], r["cost_chg"] * 100,
               r["peak_chg"] * 100, r["peak_vs_peakday"] * 100, r["v5_top1_nodes"]))

o.write("\n== 10000244 专项 ==\n")
if "10000244" in t.index:
    r = t.loc["10000244"]
    o.write("  cost_last7=%.0f cost_chg=%+.1f%% peak_chg=%+.1f%% vs峰值日=%.1f%% V5Top1节点=%d\n"
            % (r["cost_last7"], r["cost_chg"] * 100, r["peak_chg"] * 100, r["peak_vs_peakday"] * 100, r["v5_top1_nodes"]))

o.write("\n== V5 Top1 落在掉量业务的暴露（cost_chg<0） ==\n")
tot = int(t["v5_top1_nodes"].sum())
dec_nodes = int(t.loc[t["cost_chg"] < 0, "v5_top1_nodes"].sum())
o.write("  Top1 总节点 %d；其中落在掉量业务 %d (%.1f%%)\n" % (tot, dec_nodes, dec_nodes / max(tot, 1) * 100))

t.reset_index().to_csv(HERE / "掉量时效核查明细（新）.csv", index=False, encoding="utf-8-sig")
txt = o.getvalue()
(HERE / "掉量时效核查（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
