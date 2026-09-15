# -*- coding: utf-8 -*-
"""[wcx2 新增] 敏感度与上界（新）：本地最后一项分析

A) 评估口径敏感度：建设带宽下限 × winsorize 分位 → 时间外 R²/RMSE
B) 分配器敏感度：凹段斜率 × 切点分位 → LP 相对 greedy 的目标值差
C) 上界：容量缺口导致的"不可得价值"上界 vs 分配质量可改进空间

用法：python "敏感度分析（新）.py"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "recent_month_large_mainstream_v5_sw_7c2.0（新）"
PAIRS = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）" / "v1_training_pairs_large_mainstream_recent_1m.csv"
SVI = ROOT / "recent_month_large_mainstream_v6_capacity_cap20_curfull（新）" / "v6_business_capacity_summary.csv"
REC = V5 / "v5_node_recommendations.csv"
FLOORS = [0, 100, 500, 1000, 2000]
TAILS = [(0.01, 0.99), (0.05, 0.95), (0.10, 0.90)]
SLOPES = [(1.0, 0.6, 0.3), (1.0, 0.9, 0.8)]
KNOTS = [(0.5, 0.9), (0.3, 0.8)]


def wr2(a, q, w):
    a = np.asarray(a, float); q = np.asarray(q, float); w = np.asarray(w, float)
    m = np.isfinite(a) & np.isfinite(q)
    a, q, w = a[m], q[m], w[m]
    if len(a) == 0:
        return np.nan, np.nan
    gm = np.average(a, weights=w)
    sse = np.average((a - q) ** 2, weights=w)
    sst = np.average((a - gm) ** 2, weights=w)
    return np.sqrt(sse), 1 - sse / sst


def wq(values, weights, q):
    o = np.argsort(values)
    v, w = np.asarray(values)[o], np.asarray(weights)[o]
    return float(np.interp(q, np.cumsum(w) / w.sum(), v))


def part_a() -> pd.DataFrame:
    pred = pd.read_csv(V5 / "v5_temporal_validation_predictions.csv", low_memory=False)
    pairs = pd.read_csv(PAIRS, low_memory=False, usecols=["node_id", "sample_day", "business", "construction_bandwidth_mbps"])
    pairs["business"] = pd.to_numeric(pairs["business"], errors="coerce").astype("Int64")
    pred["business"] = pd.to_numeric(pred["business"], errors="coerce").astype("Int64")
    m = pred.merge(pairs, on=["node_id", "sample_day", "business"], how="left")
    m["w"] = pd.to_numeric(m["sample_weight"], errors="coerce").fillna(1.0)
    rows = []
    base = m.dropna(subset=["construction_bandwidth_mbps"])
    for floor in FLOORS:
        s = base[base["construction_bandwidth_mbps"] >= floor] if floor else base
        for lo, hi in TAILS:
            # 向量化裁剪（保持行序，避免与预测错位）
            out = {}
            for col, tag in (("actual_miner_unit_income", "m"), ("actual_platform_unit_profit", "p")):
                v = pd.to_numeric(s[col], errors="coerce")
                wv = s["w"].to_numpy()
                lows = s.groupby("business")[col].transform(lambda x: wq(pd.to_numeric(x, errors="coerce").to_numpy(), s.loc[x.index, "w"].to_numpy(), lo))
                highs = s.groupby("business")[col].transform(lambda x: wq(pd.to_numeric(x, errors="coerce").to_numpy(), s.loc[x.index, "w"].to_numpy(), hi))
                out[tag] = np.clip(pd.to_numeric(v, errors="coerce").to_numpy(), lows.to_numpy(), highs.to_numpy())
            mr, mR = wr2(out["m"], s["predicted_miner_unit_income"], s["w"])
            pr, pR = wr2(out["p"], s["predicted_platform_unit_profit"], s["w"])
            rows.append({"带宽下限": floor, "winsorize": f"P{int(lo*100)}-P{int(hi*100)}", "n": len(s),
                         "矿主R2": round(mR, 3), "矿主RMSE": round(mr, 4),
                         "平台R2": round(pR, 3), "平台RMSE": round(pr, 4)})
    return pd.DataFrame(rows)


def part_b() -> pd.DataFrame:
    import importlib.util
    spec = importlib.util.spec_from_file_location("alloc", ROOT / "wcx2实验代码（新）" / "分配器v0（新）.py")
    alloc = importlib.util.module_from_spec(spec); spec.loader.exec_module(alloc)
    rec = pd.read_csv(REC, low_memory=False)
    cap = pd.read_csv(SVI, low_memory=False)
    items, pool_cap, biz_cap, demand, _ = alloc.build_items(rec, cap, "headroom")
    rows = []
    for slopes in SLOPES:
        for k1q, k2q in KNOTS:
            k1 = demand.reindex(biz_cap.index).fillna(0.0) * k1q
            k2 = demand.reindex(biz_cap.index).fillna(0.0) * k2q
            knots = pd.DataFrame({"g_mean": items.groupby("business")["g_pb"].mean()}).reindex(biz_cap.index).fillna(0.0)
            knots["k1"] = k1
            knots["k2"] = k2
            old = alloc.SEGMENT_SLOPES
            alloc.SEGMENT_SLOPES = slopes
            lp = alloc.solve_lp(items, pool_cap, biz_cap, knots)
            gr = alloc.greedy(items, pool_cap, biz_cap)
            lp_v, gr_v = alloc.concave_value(lp, knots), alloc.concave_value(gr, knots)
            alloc.SEGMENT_SLOPES = old
            rows.append({"段斜率": "/".join(f"{s:g}" for s in slopes),
                         "切点": f"P{int(k1q*100)}/P{int(k2q*100)}",
                         "LP目标": round(lp_v), "greedy目标": round(gr_v),
                         "LP相对": f"{(lp_v-gr_v)/max(gr_v,1e-9):.2%}",
                         "LP业务数": int(lp.groupby(level=1).ngroups), "greedy业务数": int(gr.groupby(level=1).ngroups)})
    return pd.DataFrame(rows)


def part_c() -> pd.DataFrame:
    """上界：容量缺口 vs 分配质量。"""
    cap = pd.read_csv(SVI, low_memory=False)
    cap = cap[cap["pool_level"].eq("business")].copy()
    cap["business"] = pd.to_numeric(cap["business"], errors="coerce")
    cap = cap.dropna(subset=["business"])
    ceiling = cap.set_index(cap["business"].astype("int64"))["operational_capacity_ceiling_mbps"].astype(float)
    rec = pd.read_csv(REC, low_memory=False)
    rec["bw"] = pd.to_numeric(rec["build_bandwidth_mbps"], errors="coerce").fillna(0.0)
    demand = None
    for k in (1, 2, 3):
        col = f"business_top{k}"
        if col in rec.columns:
            b = pd.to_numeric(rec[col], errors="coerce")
            d = rec.assign(b=b).dropna(subset=["b"]).groupby("b")["bw"].sum()
            demand = d if demand is None else demand.add(d, fill_value=0)
    dm = pd.Series(demand)
    gap = (dm.reindex(ceiling.index).fillna(0) - ceiling).clip(lower=0)
    total_demand = float(dm.reindex(ceiling.index).fillna(0).sum())
    return pd.DataFrame([
        {"项": "业务需求合计(Gbps)", "值": round(total_demand / 1000, 1)},
        {"项": "业务级上限合计(Gbps)", "值": round(float(ceiling.sum()) / 1000, 1)},
        {"项": "扩容缺口合计(Gbps)", "值": round(float(gap.sum()) / 1000, 1)},
        {"项": "缺口/需求", "值": f"{gap.sum()/max(total_demand,1):.1%}"},
        {"项": "分配质量可改进上界（LP vs greedy，建模值）", "值": "约 +9%（见 分配器v0（新）.md）"},
    ])


if __name__ == "__main__":
    print("=== A. 评估口径敏感度（带宽下限 × winsorize 分位）===")
    a = part_a()
    print(a.to_string(index=False))
    print("\n=== B. 分配器敏感度（凹段斜率 × 切点分位）===")
    print(part_b().to_string(index=False))
    print("\n=== C. 上界：容量缺口 vs 分配质量 ===")
    print(part_c().to_string(index=False))
