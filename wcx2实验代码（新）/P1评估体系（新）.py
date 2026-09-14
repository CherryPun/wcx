# -*- coding: utf-8 -*-
"""P1 评估体系（新）：① 可行集(容量内)score-regret；② 稀释后期望收益；③ 观察性 OPE/IPS 基线。
均为离线度量；regret 基于模型分数（模型型），非因果。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "recent_month_large_mainstream_v5_cap20_curfull（新）"
V6 = ROOT / "recent_month_large_mainstream_v6_capacity_cap20_curfull（新）"

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "str"}, low_memory=False)
a = pd.read_csv(V6 / "v6_capacity_allocation_summary.csv", low_memory=False)
b = a[a.pool_level == "business"].copy(); b["business"] = b["business"].astype(str)
ceil = dict(zip(b.business, pd.to_numeric(b.raw_capacity_ceiling_mbps, errors="coerce")))
head = dict(zip(b.business, pd.to_numeric(b.allocatable_headroom_mbps, errors="coerce").fillna(0)))
load = dict(zip(b.business, pd.to_numeric(b.unconstrained_top1_load_mbps, errors="coerce")))

for k in ("1", "2", "3"):
    rec[f"b{k}"] = rec[f"business_top{k}"].astype(str).str.replace(r"\.0$", "", regex=True)
    rec[f"s{k}"] = pd.to_numeric(rec[f"combined_score_top{k}"], errors="coerce")
rec["cur"] = rec["current_business"].astype(str).str.replace(r"\.0$", "", regex=True)

regret, regn, feas_n, hold_n = [], 0, 0, 0
dil = []
for _, r in rec.iterrows():
    cand = [(getattr(r, f"b{k}"), getattr(r, f"s{k}")) for k in ("1", "2", "3")
            if pd.notna(getattr(r, f"s{k}")) and getattr(r, f"b{k}") not in ("", "nan")]
    if not cand:
        continue
    feas = [(bb, s) for bb, s in cand if head.get(bb, 0) > 0]
    if not feas:
        hold_n += 1
        continue
    feas_n += 1
    oracle = max(s for _, s in feas)
    sel = cand[0]
    sel = sel if head.get(sel[0], 0) > 0 else None
    if sel is None:
        hold_n += 1
        continue
    regn += 1
    regret.append(max(0.0, (oracle - sel[1]) / max(abs(oracle), 1e-9)))
    u = min(1.0, (ceil.get(sel[0], 0) / load[sel[0]])) if load.get(sel[0], 0) else 1.0
    dil.append(sel[1] * u)

# 分配级容量损失：无约束可得分 vs V6 planned 得分
nr = pd.read_csv(V6 / "v6_capacity_node_report.csv", dtype={"node_id": "str"}, low_memory=False)
planned_gain = pd.to_numeric(nr.get("planned_score_gain_vs_current"), errors="coerce").fillna(0).clip(lower=0).sum()
best_gain = pd.to_numeric(nr.get("best_capacity_candidate_score_gain"), errors="coerce").fillna(0).clip(lower=0).sum()
loss = 1 - planned_gain / best_gain if best_gain > 0 else float("nan")

o = io.StringIO()
o.write("== P1 评估基线 ==\n")
o.write("节点 %d；候选可行(容量内) %d；无可行动作(hold) %d\n" % (len(rec), feas_n, hold_n))
o.write("分配级容量损失 = 1 - planned增益/理想增益 = %.4f（理想 %.2f vs 实际 %.2f）\n"
        % (loss, best_gain, planned_gain))
o.write("稀释后期望分值 mean=%.4f vs 名义 top1 分 mean=%.4f（稀释来自业务全切负载/上限）\n"
        % (float(np.mean(dil)) if dil else float("nan"), float(np.nanmean(rec["s1"]))))

# OPE/IPS 基线（观察性）：用 V5 时序验证文件中的倾向权重
p = V5 / "v5_temporal_validation_predictions.csv"
if p.exists():
    t = pd.read_csv(p, dtype={"node_id": "str", "business": "str"}, low_memory=False)
    t["actual"] = pd.to_numeric(t.get("actual_miner_unit_income"), errors="coerce")
    t["prop"] = pd.to_numeric(t.get("propensity"), errors="coerce").clip(lower=0.01)
    # 观测策略价值 = 直接均值；理想重加权(以观测分布近似)作为尺度参考
    obs_val = float(t["actual"].mean())
    w = 1.0 / t["prop"]
    ips = float((t["actual"] * w).sum() / w.sum())
    o.write("\nOPE(观察性)：观测均值=%.4f；倾向加权均值=%.4f（propensity 截断 0.01）\n" % (obs_val, ips))
    o.write("说明：单节点日仅一个观测业务，IPS 仅作尺度参考，不能证明策略因果增益。\n")
txt = o.getvalue()
(ROOT / "P1评估基线（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
