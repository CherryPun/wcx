# -*- coding: utf-8 -*-
"""四层池时间线模拟（新）：按业务/业务+运营商/业务+运营商+调度/业务+省份+运营商+调度 四层容量约束，
逐节点分配（混合规则：紧迫度→增益密度→随机），消费/释放四层余量。"""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "recent_month_large_mainstream_v5_cap20_curfull（新）"
V6 = ROOT / "recent_month_large_mainstream_v6_capacity_cap20_curfull（新）"
SEED, MIN_GAIN = 42, 0.03

rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "str"}, low_memory=False)
nr = pd.read_csv(V6 / "v6_capacity_node_report.csv", dtype={"node_id": "str"}, low_memory=False)
bw = {str(k): float(v) for k, v in zip(nr["node_id"], pd.to_numeric(nr["build_bandwidth_mbps"], errors="coerce").fillna(1000.0))}

p = pd.read_csv(V6 / "v6_capacity_pool_summary.csv", low_memory=False)
p["business"] = p["business"].astype(str)
p = p[p["capacity_evidence_sufficient"].astype(str).str.lower().isin(["true", "1"])].copy()
for c in ["daily_province", "daily_isp", "network_schedule_type", "schedule_target_isp"]:
    p[c] = p[c].fillna("").astype(str)
hr_b = {r.business: float(r.allocatable_headroom_mbps) for r in p[p.pool_level == "business"].itertuples()}
hr_bi = {(r.business, r.daily_isp): float(r.allocatable_headroom_mbps) for r in p[p.pool_level == "business_isp"].itertuples()}
hr_bis = {(r.business, r.daily_isp, r.network_schedule_type, r.schedule_target_isp): float(r.allocatable_headroom_mbps)
          for r in p[p.pool_level == "business_isp_schedule"].itertuples()}
hr_bpis = {(r.business, r.daily_province, r.daily_isp, r.network_schedule_type, r.schedule_target_isp): float(r.allocatable_headroom_mbps)
           for r in p[p.pool_level == "business_province_isp_schedule"].itertuples()}

# 初始余量（可变副本）
R = {"b": dict(hr_b), "bi": dict(hr_bi), "bis": dict(hr_bis), "bpis": dict(hr_bpis)}

def keys(b, prov, isp, sched, tisp):
    return [("b", (b,)), ("bi", (b, isp)), ("bis", (b, isp, sched, tisp)), ("bpis", (b, prov, isp, sched, tisp))]

def headroom(b, prov, isp, sched, tisp):
    if b not in R["b"]:
        return -1.0
    vals = []
    for lvl, k in keys(b, prov, isp, sched, tisp):
        if k in R[lvl]:
            vals.append(R[lvl][k])
    return min(vals) if vals else -1.0

def consume(b, prov, isp, sched, tisp, delta):
    for lvl, k in keys(b, prov, isp, sched, tisp):
        if k in R[lvl]:
            R[lvl][k] += delta

rec["cur"] = rec["current_business"].astype(str).str.replace(r"\.0$", "", regex=True)
rec["curp"] = pd.to_numeric(rec.get("current_actual_platform_profit_1d"), errors="coerce")
rec["prov"] = rec.get("province", "").astype(str)
rec["isp"] = rec.get("isp", "").astype(str)
rec["sched"] = rec.get("network_schedule_type", "").astype(str)
rec["tisp"] = rec.get("scheduleisps", "").astype(str).replace({"nan": ""})
rec["tisp"] = [ (x.split(",")[0].strip() if x else i) for x, i in zip(rec["tisp"], rec["isp"]) ]
rec = rec[rec["node_id"].astype(str).isin(bw)].copy()

rng = np.random.RandomState(SEED)
items = []
for _, r in rec.iterrows():
    cands = []
    for k in (1, 2, 3):
        b = str(r.get(f"business_top{k}", "")).replace(".0", "")
        s = pd.to_numeric(r.get(f"combined_score_top{k}"), errors="coerce")
        pf = pd.to_numeric(r.get(f"platform_unit_profit_top{k}"), errors="coerce")
        if b and b != "nan" and pd.notna(s):
            cands.append((b, float(s), pf))
    if not cands:
        continue
    cs = pd.to_numeric(r.get("current_predicted_combined_score"), errors="coerce")
    gain = cands[0][1] - (float(cs) if pd.notna(cs) else 0.0)
    items.append({"node_id": r["node_id"], "cur": r["cur"], "bw": bw[r["node_id"]], "cands": cands,
                  "gain": gain, "density": gain / max(bw[r["node_id"]], 1.0),
                  "urgent": r["curp"] if pd.notna(r["curp"]) else 1e9,
                  "prov": r["prov"], "isp": r["isp"], "sched": r["sched"], "tisp": r["tisp"],
                  "rnd": rng.rand()})
items.sort(key=lambda x: (x["urgent"], -x["density"], x["rnd"]))

sw = Counter(); nsw = 0
for it in items:
    chosen = None
    for b, s, pf in it["cands"]:
        if it["gain"] < MIN_GAIN:
            break
        if pf is not None and pd.notna(pf) and pf <= 0:
            continue
        if headroom(b, it["prov"], it["isp"], it["sched"], it["tisp"]) >= it["bw"]:
            chosen = b
            break
    if chosen and chosen != it["cur"]:
        consume(chosen, it["prov"], it["isp"], it["sched"], it["tisp"], -it["bw"])
        consume(it["cur"], it["prov"], it["isp"], it["sched"], it["tisp"], +it["bw"])
        sw[chosen] += 1; nsw += 1

cnt = sum(sw.values())
mx = max(sw.values()) / cnt if cnt else 0
hhi = sum((v / cnt) ** 2 for v in sw.values()) if cnt else 0
o = io.StringIO()
o.write("== 四层池时间线模拟（业务/运营商/调度/省份） ==\n")
o.write("节点 %d；切换 %d；hold %d\n" % (len(items), nsw, len(items) - nsw))
o.write("切换目标分布(前8): %s\n" % dict(sw.most_common(8)))
o.write("max占比=%.4f HHI=%.4f\n" % (mx, hhi))
o.write("\n对照（业务层）：切换 1477 / hold 811\n")
txt = o.getvalue()
(ROOT / "四层池模拟结果（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
