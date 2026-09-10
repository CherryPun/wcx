# -*- coding: utf-8 -*-
"""M1+M2（新）：稀释感知期望收益 + 降档/hold 规则。
对 V5 候选 Top1~3 按"业务/业务+运营商/业务+运营商+调度"池余量折减分数；
若目标池放不下本节点（余量<=0）则该候选不可行；都不行则维持当前（hold）。
输出 V6 可直接使用的推荐文件。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
V6 = HERE / "recent_month_large_mainstream_v6_capacity_full（新）"

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string"}, low_memory=False)
nodes = pd.read_csv(V3 / "current_online_inservice_non_idc_large_nodes_v3.csv", dtype={"node_id": "string"}, low_memory=False)
pool = pd.read_csv(V6 / "v6_capacity_pool_summary.csv", low_memory=False)

rec["province"] = rec.get("province", "").astype(str)
rec["isp"] = rec.get("isp", "").astype(str)
rec["network_schedule_type"] = rec.get("network_schedule_type", "").astype(str)
rec["scheduleisps"] = rec.get("scheduleisps", "").astype(str)
bw = pd.to_numeric(nodes.get("bw", nodes.get("bandwidth")), errors="coerce")
nodes["node_bw"] = bw
rec = rec.merge(nodes[["node_id", "node_bw"]], on="node_id", how="left")
rec["node_bw"] = pd.to_numeric(rec["node_bw"], errors="coerce").fillna(1000.0)


def target_isp(row):
    s = str(row.get("scheduleisps", "") or "").strip()
    if s and s.lower() not in ("nan", "none"):
        for sep in [",", "，", "|", "/", " "]:
            if sep in s:
                return s.split(sep)[0].strip()
        return s
    return str(row.get("isp", ""))


rec["schedule_target_isp"] = rec.apply(target_isp, axis=1)


def headroom_for(business, isp, sched, tisp, province):
    sub = pool[pool["business"].astype(str) == str(business)]
    if sub.empty:
        return np.nan
    b = sub[sub["pool_level"] == "business"]
    if not b.empty:
        hr = float(b.iloc[0]["allocatable_headroom_mbps"])
    else:
        hr = np.nan
    bi = sub[(sub["pool_level"] == "business_isp") & (sub["daily_isp"].astype(str) == str(isp))]
    if not bi.empty and pd.notna(bi.iloc[0]["allocatable_headroom_mbps"]):
        hr = min(hr, float(bi.iloc[0]["allocatable_headroom_mbps"])) if pd.notna(hr) else float(bi.iloc[0]["allocatable_headroom_mbps"])
    bs = sub[(sub["pool_level"] == "business_isp_schedule") & (sub["daily_isp"].astype(str) == str(isp))
             & (sub["network_schedule_type"].astype(str) == str(sched)) & (sub["schedule_target_isp"].astype(str) == str(tisp))]
    if not bs.empty and pd.notna(bs.iloc[0]["allocatable_headroom_mbps"]):
        v = float(bs.iloc[0]["allocatable_headroom_mbps"])
        hr = min(hr, v) if pd.notna(hr) else v
    return hr


out = rec.copy()
rows = []
n_hold = n_infeas = 0
new_counts = {}
for i, r in rec.iterrows():
    nbid = r["node_bw"]
    cands = []
    for k in (1, 2, 3):
        b = str(r.get(f"business_top{k}", ""))
        s = pd.to_numeric(r.get(f"combined_score_top{k}"), errors="coerce")
        if not b or b == "nan" or pd.isna(s):
            continue
        hr = headroom_for(b, r["isp"], r["network_schedule_type"], r["schedule_target_isp"], r["province"])
        dil = 0.0 if (pd.isna(hr) or hr <= 0) else min(1.0, float(hr) / max(float(nbid), 1.0))
        cands.append((b, float(s) * dil, dil))
    cur = str(r.get("current_business", ""))
    cur_score = pd.to_numeric(r.get("current_predicted_combined_score"), errors="coerce")
    if cur and cur != "nan" and pd.notna(cur_score):
        cands.append((cur, float(cur_score), 1.0))
    if not cands:
        continue
    cands.sort(key=lambda x: x[1], reverse=True)
    best_b, best_s, best_d = cands[0]
    if best_b == cur:
        n_hold += 1
        if best_d == 1.0 and all(c[2] == 0.0 for c in cands if c[0] != cur):
            n_infeas += 1
    # 写回 Top1~3：把最优业务放第一，其余按原顺序补
    seq = [best_b] + [c[0] for c in cands if c[0] != best_b]
    seen, seq2 = set(), []
    for b in seq:
        if b not in seen:
            seen.add(b)
            seq2.append(b)
    for k in range(1, 4):
        b = seq2[k - 1] if k - 1 < len(seq2) else ""
        out.at[i, f"business_top{k}"] = b
        if k == 1:
            out.at[i, "combined_score_top1"] = best_s
    new_counts[best_b] = new_counts.get(best_b, 0) + 1
    rows.append({"node_id": r["node_id"], "old_top1": str(r.get("business_top1")), "new_top1": best_b,
                 "dilution": round(best_d, 3), "held": int(best_b == cur)})

path = HERE / "v5_node_recommendations_m1m2（新）.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(HERE / "M1M2明细（新）.csv", index=False, encoding="utf-8-sig")
new_series = pd.Series(new_counts)

o = io.StringIO()
o.write("== M1+M2 稀释感知重排（新） ==\n")
o.write("节点数 %d；维持当前(hold) %d (%.1f%%)；其中目标池全不可行 %d\n"
        % (len(rows), n_hold, n_hold / max(len(rows), 1) * 100, n_infeas))
o.write("原 Top1 分布(前8): %s\n" % rec["business_top1"].astype(str).value_counts().head(8).to_dict())
o.write("M1M2 Top1 分布(前8): %s\n" % new_series.sort_values(ascending=False).head(8).to_dict())
o.write("输出: %s\n" % path.name)
txt = o.getvalue()
(HERE / "M1M2结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
