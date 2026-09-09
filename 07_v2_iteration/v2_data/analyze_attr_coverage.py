# -*- coding: utf-8 -*-
"""V2 阶段A-D3a：训练节点关键属性缺失诊断与 as-of 回填候选盘点。
只读共享数据并写本代 _rerun/，不修改任何主版文件。
目标：定位“在训练样本合并后、关键画像字段缺失”的节点，产出待回填清单。
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "05_shared_data"
OUT = HERE / "_rerun"
OUT.mkdir(parents=True, exist_ok=True)

KEY = ["province", "isp", "bw", "corenum", "memtotal",
       "totaldisksize", "hdddisksize", "ssddisksize", "systemdisksize",
       "deliverytype", "device_type", "nattype"]

attrs = pd.read_csv(DATA / "nodes_attr_filled.csv", dtype={"node_id": str}, low_memory=False)
attrs = attrs.drop_duplicates("node_id").set_index("node_id")
outcomes = pd.read_csv(DATA / "outcomes_7d_named.csv",
                       dtype={"node_id": str, "business": str}, low_memory=False)
outcomes = outcomes[outcomes["outcome_days"] >= 7]
target = set(outcomes["node_id"])
present = attrs.index.intersection(target)
missing_node = target.difference(attrs.index)
sub = attrs.loc[list(present)].copy()

rows = []
for c in KEY:
    if c not in sub.columns:
        continue
    nonnull = sub[c].notna()
    rows.append({"field": c, "rows_non_null": int(nonnull.sum()),
                 "rows_total": len(sub), "coverage": round(float(nonnull.mean()), 4)})
cov = pd.DataFrame(rows)
print("目标节点(满7天账):", len(target), " 在画像表:", len(present), " 不在画像表:", len(missing_node))
print(cov.to_string(index=False))

# 每目标节点“关键字段缺失”情况 -> 待回填候选
cols = [c for c in KEY if c in sub.columns]
sub["_n_missing"] = sub[cols].isna().sum(axis=1)
sub["_any_missing"] = sub["_n_missing"] > 0
cand = sub[sub["_any_missing"]].copy()
cand_out = cand.reset_index()[["node_id"] + cols + ["_n_missing"]]
cand_out["missing_fields"] = cand_out[cols].isna().apply(
    lambda r: "|".join(cols[i] for i in range(len(cols)) if r.iloc[i]), axis=1)
cand_out = cand_out[["node_id", "_n_missing", "missing_fields"]]
cand_out.to_csv(OUT / "attr_missing_candidates.csv", index=False, encoding="utf-8-sig")
print("关键字段至少缺1个的节点数:", len(cand_out))
print("缺失分布(每节点缺几个字段):")
print(cand_out["_n_missing"].value_counts().sort_index().to_string())
print("wrote", OUT / "attr_missing_candidates.csv")
