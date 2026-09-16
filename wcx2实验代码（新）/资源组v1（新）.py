# -*- coding: utf-8 -*-
"""[wcx2 新增] 资源组 v1（新）：组×业务大类 的收益/利用率排名 + T-1 vs T-2 差异

资源组 = 省+市+原运营商+调度运营商+出省配置+NAT+IPv6+网络质量(压测满意度分档)
细分方向 = 业务大类；利用率 = 计费带宽/建设带宽；阈值：优≥95 / 中80-95 / 差<80 / 未压测=未知
用法：python "资源组v1（新）.py" [--t1 2026-09-09] [--t2 2026-09-08]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）"
PAIRS = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"
PROFILE = ROOT / "recent_month_large_1d（新）" / "multibusiness_nodes_large_recent_1m.csv"
SNAP = ROOT / "snapshots"
BIG = {"字节系": {10000079, 10000088, 10000096, 10000009, 10000011, 1382680008, 10000063, 10000210, 1382522065, 10000224},
       "B站系": {10000019, 10000041, 10000065, 10000183, 10000244},
       "七牛系": {10000280, 10000026, 10000071, 10000072, 10000075, 10000078, 10000089, 10000127, 10000138, 1380460970, 10000284}}


def big_class(b: int) -> str:
    for k, v in BIG.items():
        if b in v:
            return k
    return "其他"


def quality(pct: float) -> str:
    if pd.isna(pct):
        return "未知"
    return "优" if pct >= 95 else ("中" if pct >= 80 else "差")


def load_bench(day: str) -> pd.DataFrame:
    f = SNAP / f"bench_{day.replace('-', '')}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["node_id", "q"])
    d = pd.read_csv(f, usecols=["node_id", "overall_packet_loss_benchmark_satisfaction_pct"], low_memory=False)
    d["q"] = pd.to_numeric(d["overall_packet_loss_benchmark_satisfaction_pct"], errors="coerce").map(quality)
    return d[["node_id", "q"]].drop_duplicates("node_id")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t1", default="2026-09-09")
    ap.add_argument("--t2", default="2026-09-08")
    a = ap.parse_args()
    cols = ["node_id", "sample_day", "business", "province", "isp", "nattype", "scheduleisps",
            "network_schedule_type", "construction_bandwidth_mbps", "effective_bandwidth_mbps", "cum_cost_7d"]
    p = pd.read_csv(PAIRS, low_memory=False, usecols=lambda c: c in cols)
    p = p[p["sample_day"].isin([a.t1, a.t2])].copy()
    prof = pd.read_csv(PROFILE, low_memory=False)
    ipv6col = next((c for c in ["dial_ipv6_enable", "join_isipv6schedule"] if c in prof.columns), None)
    ipv6 = prof[["node_id", ipv6col]].drop_duplicates("node_id") if ipv6col else pd.DataFrame(columns=["node_id", "ipv6"])

    rows = []
    for day in (a.t1, a.t2):
        d = p[p.sample_day == day].merge(ipv6, on="node_id", how="left").merge(load_bench(day), on="node_id", how="left")
        d["ipv6"] = d[ipv6col].fillna("未知").astype(str) if ipv6col else "未知"
        d["q"] = d["q"].fillna("未知")
        d["city"] = "未知"
        dims = [c for c in ["province", "city", "isp", "scheduleisps", "network_schedule_type", "nattype", "ipv6", "q"] if c in d.columns]
        for c in ["province", "isp", "scheduleisps", "network_schedule_type", "nattype"]:
            if c in d.columns:
                d[c] = d[c].fillna("未知").astype(str)
        d["grp"] = d[dims].astype(str).agg("|".join, axis=1)
        d["cls"] = pd.to_numeric(d["business"], errors="coerce").fillna(-1).astype("int64").map(big_class)
        d["bw"] = pd.to_numeric(d["construction_bandwidth_mbps"], errors="coerce")
        d["billed"] = pd.to_numeric(d["effective_bandwidth_mbps"], errors="coerce")
        d = d[(d.bw >= 500) & d.bw.notna()]
        d["unit"] = pd.to_numeric(d["cum_cost_7d"], errors="coerce") / d["bw"]
        aud = pd.read_csv(V3 / "node_day_sampling_audit_v3.csv", low_memory=False,
                          usecols=lambda c: c in ["node_id", "sample_day", "capacity_peak95_bps", "daily_city"])
        aud = aud[aud.sample_day == day].drop_duplicates(["node_id", "sample_day"])
        aud["peak95_mbps"] = pd.to_numeric(aud["capacity_peak95_bps"], errors="coerce") / 1e6
        d = d.merge(aud[["node_id", "peak95_mbps"] + (["daily_city"] if "daily_city" in aud.columns else [])],
                    on="node_id", how="left")
        if "daily_city" in d.columns:
            d["city"] = d["daily_city"].fillna("未知").astype(str)
        d["util"] = d["peak95_mbps"] / d["bw"]
        g = d.groupby(["grp", "cls"]).agg(nodes=("node_id", "nunique"), bw=("bw", "sum"),
                                          unit=("unit", "median"), util=("util", "median")).reset_index()
        g["day"] = day
        g["unit_rank"] = g.groupby("grp")["unit"].rank(ascending=False, method="min")
        g["util_rank"] = g.groupby("grp")["util"].rank(ascending=False, method="min")
        g["evidence"] = np.where(g.nodes >= 3, "可用", "证据不足")
        rows.append(g)
    r1, r2 = rows[0], rows[1]
    m = r1.merge(r2, on=["grp", "cls"], how="outer", suffixes=("_t1", "_t2"))
    m["unit_delta"] = m["unit_t2"] - m["unit_t1"]
    m["util_delta"] = m["util_t2"] - m["util_t1"]
    m["rank_move"] = m["unit_rank_t1"] - m["unit_rank_t2"]
    out = ROOT / "资源组v1差异（新）.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")
    r1.to_csv(ROOT / "资源组v1排名T1（新）.csv", index=False, encoding="utf-8-sig")
    print(f"组数 {m['grp'].nunique()} | 组×大类 {len(m)} | 可用证据 {int(m['evidence_t1'].eq('可用').sum()) if 'evidence_t1' in m else '-'}")
    print(f"T1={a.t1} 组×大类 {len(r1)} | T2={a.t2} 组×大类 {len(r2)}")
    print("\n示例（T1 组内单位收益 Top1，取前 5 组）：")
    ex = r1[r1.unit_rank == 1].nlargest(5, "unit")[["grp", "cls", "nodes", "unit", "util", "evidence"]]
    print(ex.to_string(index=False, max_colwidth=42))
    print("\n两日变化最大的 5 条（|rank_move| 或 |util_delta|）：")
    mv = m.assign(score=lambda d: d.rank_move.abs().fillna(0) + (d.util_delta.abs().fillna(0) * 10)).nlargest(5, "score")
    print(mv[["grp", "cls", "rank_move", "unit_delta", "util_delta"]].to_string(index=False, max_colwidth=42))
    print(f"\n[out] -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
