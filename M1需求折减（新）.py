# -*- coding: utf-8 -*-
"""M1（V5 打分侧，新）：按业务级"需求/供给"稀释因子折减 V5 候选综合分。
稀释因子 = min(1, 容量上限 / 全切Top1负载)（来自 V6 基线业务层容量）。
业务级因子对所有节点一致，避免"逐节点重复使用余量"的偏松问题。
输出 V6 可直接使用的推荐文件。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V5 = HERE / "recent_month_large_mainstream_v5_hybrid（新）"
V6 = HERE / "recent_month_large_mainstream_v6_capacity_full（新）"

cap = pd.read_csv(V6 / "v6_capacity_allocation_summary.csv", low_memory=False)
biz = cap[cap["pool_level"].eq("business")].copy()
dil = {}
for _, r in biz.iterrows():
    load = pd.to_numeric(r.get("unconstrained_top1_load_mbps"), errors="coerce")
    ceil = pd.to_numeric(r.get("raw_capacity_ceiling_mbps"), errors="coerce")
    b = str(r["business"])
    if pd.notna(load) and load > 0 and pd.notna(ceil):
        dil[b] = float(min(1.0, ceil / load))
    else:
        dil[b] = 1.0


def factor(b):
    return dil.get(str(b), 1.0)


rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "string"}, low_memory=False)
for c in ["combined_score_top1", "combined_score_top2", "combined_score_top3"]:
    rec[c] = pd.to_numeric(rec.get(c), errors="coerce")
for c in ["business_top1", "business_top2", "business_top3", "current_business"]:
    rec[c] = rec[c].astype(str)

out = rec.copy()
rows, changed = [], 0
new_counts = {}
for i, r in rec.iterrows():
    cands = []
    for k in (1, 2, 3):
        b = str(r.get(f"business_top{k}", ""))
        s = r.get(f"combined_score_top{k}")
        if b and b != "nan" and pd.notna(s):
            cands.append((b, float(s) * factor(b)))
    if not cands:
        continue
    old = str(r.get("business_top1"))
    cands.sort(key=lambda x: x[1], reverse=True)
    if cands[0][0] != old:
        changed += 1
    new_counts[cands[0][0]] = new_counts.get(cands[0][0], 0) + 1
    for k, (b, s) in enumerate(cands, 1):
        out.at[i, f"business_top{k}"] = b
        out.at[i, f"combined_score_top{k}"] = s
    rows.append({"node_id": r["node_id"], "old_top1": old, "new_top1": cands[0][0],
                 "factor_old": round(factor(old), 3), "factor_new": round(factor(cands[0][0]), 3)})

path = HERE / "v5_node_recommendations_demand（新）.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
new_series = pd.Series(new_counts)
o = io.StringIO()
o.write("== M1 打分侧需求折减（业务级稀释因子） ==\n")
o.write("非 1 因子业务: %s\n\n" % {k: round(v, 3) for k, v in dil.items() if v < 1})
o.write("Top1 改变 %d / %d\n" % (changed, len(rows)))
o.write("原 Top1 分布(前8): %s\n" % rec["business_top1"].value_counts().head(8).to_dict())
o.write("折减后 Top1 分布(前8): %s\n" % new_series.sort_values(ascending=False).head(8).to_dict())
o.write("输出: %s\n" % path.name)
txt = o.getvalue()
(HERE / "M1需求折减结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
