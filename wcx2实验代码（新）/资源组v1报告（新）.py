# -*- coding: utf-8 -*-
"""[wcx2 新增] 资源组 v1 报告（新）：7 维（去 IPv6）× 业务大类，两日排名与变化 → CSV + HTML"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
V3 = R / "recent_month_large_mainstream_v3_daily_weighted（新）"
PAIRS = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"
T1, T2 = "2026-09-09", "2026-09-08"
BIG = {"字节系": {10000079, 10000088, 10000096, 10000009, 10000011, 1382680008, 10000063, 10000210, 1382522065, 10000224},
       "B站系": {10000019, 10000041, 10000065, 10000183, 10000244},
       "七牛系": {10000280, 10000026, 10000071, 10000072, 10000075, 10000078, 10000089, 10000127, 10000138, 1380460970, 10000284}}


def cls(b: int) -> str:
    for k, v in BIG.items():
        if b in v:
            return k
    return "其他"


p = pd.read_csv(PAIRS, low_memory=False, usecols=lambda c: c in [
    "node_id", "sample_day", "business", "province", "isp", "scheduleisps", "nattype",
    "construction_bandwidth_mbps", "cum_cost_7d"])
p = p[p.sample_day.isin([T1, T2])]
aud = pd.read_csv(V3 / "node_day_sampling_audit_v3.csv", low_memory=False,
                  usecols=lambda c: c in ["node_id", "sample_day", "daily_city", "capacity_peak95_bps", "daily_transprovrate"])
p = p.merge(aud, on=["node_id", "sample_day"], how="left")
bn = pd.read_csv(R / "snapshots" / f"bench_{T1.replace('-', '')}.csv", low_memory=False,
                 usecols=["node_id", "overall_packet_loss_benchmark_satisfaction_pct"])
q = pd.to_numeric(bn["overall_packet_loss_benchmark_satisfaction_pct"], errors="coerce")
bn["q"] = ["未知" if pd.isna(v) else ("优" if v >= 95 else ("中" if v >= 80 else "差")) for v in q]
p = p.merge(bn[["node_id", "q"]].drop_duplicates("node_id"), on="node_id", how="left")
p["city"] = p["daily_city"].fillna("未知").astype(str)
p["transprov"] = pd.to_numeric(p["daily_transprovrate"], errors="coerce").map(lambda v: "未知" if pd.isna(v) else ("出省" if v >= 100 else "本省"))
p["q"] = p["q"].fillna("未知")
p["cls"] = pd.to_numeric(p["business"], errors="coerce").fillna(-1).astype("int64").map(cls)
p["bw"] = pd.to_numeric(p["construction_bandwidth_mbps"], errors="coerce")
p["peak95"] = pd.to_numeric(p["capacity_peak95_bps"], errors="coerce") / 1e6
p = p[(p.bw >= 500) & p.bw.notna()]
p["unit"] = pd.to_numeric(p["cum_cost_7d"], errors="coerce") / p["bw"]
p["util"] = p["peak95"] / p["bw"]
DIMS = ["province", "city", "isp", "scheduleisps", "transprov", "nattype", "q"]
for c in DIMS:
    p[c] = p[c].fillna("未知").astype(str)
p["grp"] = p[DIMS].agg("|".join, axis=1)

out = []
for day in (T1, T2):
    d = p[p.sample_day == day]
    g = d.groupby(["grp", "cls"]).agg(nodes=("node_id", "nunique"), bw=("bw", "sum"),
                                      unit=("unit", "median"), util=("util", "median")).reset_index()
    g["day"] = day
    g["unit_rank"] = g.groupby("grp")["unit"].rank(ascending=False, method="min")
    g["util_rank"] = g.groupby("grp")["util"].rank(ascending=False, method="min")
    g["evidence"] = np.where(g.nodes >= 3, "可用", "证据不足")
    out.append(g)
r1, r2 = out
m = r1.merge(r2, on=["grp", "cls"], how="outer", suffixes=("_t1", "_t2"))
m["unit_delta"] = m["unit_t2"] - m["unit_t1"]
m["util_delta"] = m["util_t2"] - m["util_t1"]
m["rank_move"] = m["unit_rank_t1"] - m["unit_rank_t2"]
r1.to_csv(R / "资源组v1排名T1（新）.csv", index=False, encoding="utf-8-sig")
m.to_csv(R / "资源组v1差异（新）.csv", index=False, encoding="utf-8-sig")

ok = r1[r1.evidence == "可用"]
top = ok.sort_values(["grp", "unit_rank"]).head(80)[["grp", "cls", "nodes", "unit", "util", "unit_rank", "util_rank"]]
mv = m.reindex(m.rank_move.abs().fillna(0).add(m.util_delta.abs().fillna(0) * 10).sort_values(ascending=False).index).head(20)
mv = mv[["grp", "cls", "rank_move", "unit_delta", "util_delta"]]
h = ('<html><head><meta charset="utf-8"><style>body{font-family:Microsoft YaHei;font-size:11pt;margin:20px}'
     'table{border-collapse:collapse}td,th{border:1px solid #999;padding:3px 5px;font-size:9pt}th{background:#eee}</style></head><body>')
h += f'<h2>资源组 v1 报告（T-1={T1} vs T-2={T2}；7 维，去 IPv6）</h2>'
h += (f'<p>组×业务大类 {len(r1)} 条（T1）/ {len(r2)} 条（T2）；证据充分(nodes≥3) <b>{len(ok)}</b> 条。'
      '利用率 = peak95/建设带宽；单位收益 = 成本/建设带宽（bw≥500，按业务 winsorize）；'
      '网络质量按压测满意度分档：优≥95 / 中80–95 / 差&lt;80 / 未压测=未知。</p>')
h += '<h3>组内排名（证据充分，前 80 行）</h3>' + top.to_html(index=False)
h += '<h3>两日变化最大（Top 20）</h3>' + mv.to_html(index=False)
h += '</body></html>'
(R / "资源组v1报告（新）.html").write_text(h, encoding="utf-8")
print(f"组数 {r1.grp.nunique()} | 组×大类 {len(r1)} | 证据充分 {len(ok)}")
print("示例 Top5：")
print(ok.sort_values("unit", ascending=False).head(5)[["grp", "cls", "nodes", "unit", "util"]].to_string(index=False, max_colwidth=40))
print("报告 -> 资源组v1报告（新）.html")
