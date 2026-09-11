# -*- coding: utf-8 -*-
"""调用 RJ V6 的容量计算链路（load_inputs -> prepare_capacity_facts -> aggregate_capacity_pools
-> build_capacity_pool_summary），只输出业务容量摘要，不依赖 V5 推荐。
聚焦 10000244 B站专线-盒子V2 的容量上限与超限。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import v6_capacity_aware_allocation as v6

HERE = Path(__file__).resolve().parent


def main() -> int:
    src = HERE / "recent_month_large_1d（新）"
    out = HERE / "recent_month_large_mainstream_v6_capacity（新）"
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        output_dir=out,
        source_dir=src,
        allowlist=HERE / "mainstream_business_allowlist.csv",
        business_map=HERE / "v1_business_name_map_enriched.csv",
        virtual_bindings=None,
        recommendations=None,
        candidate_json=src / "large_candidate_node_ids_recent_1m.json",
        raw_input=None,
        start_day="2026-08-10",
        end_day="2026-09-09",
        target_utilization=0.70,
        forecast_days=14,
        minimum_valid_days=7,
        minimum_bandwidth_coverage=0.80,
        minimum_score_gain=0.03,
        chunk_size=120,
        workers=6,
        refresh=True,
    )
    raw, allowlist, names, bindings, raw_path = v6.load_inputs(args)
    facts, audit = v6.prepare_capacity_facts(raw, allowlist, names, bindings)
    pools = v6.aggregate_capacity_pools(facts)
    summary, corr = v6.build_capacity_pool_summary(
        pools, args.target_utilization, args.forecast_days,
        args.minimum_valid_days, args.minimum_bandwidth_coverage,
    )
    summary.to_csv(out / "v6_capacity_pool_summary（新）.csv", index=False, encoding="utf-8-sig")
    biz = summary[summary["pool_level"].eq("business")].copy()
    biz.to_csv(out / "v6_business_capacity_summary（新）.csv", index=False, encoding="utf-8-sig")
    target = biz[biz["business"].astype(str).eq("10000244")]
    cols = ["business", "business_name", "as_of_day", "valid_capacity_days", "recent_bandwidth_coverage",
            "capacity_evidence_sufficient", "capacity_state", "forecast_peak95_mbps_p75",
            "raw_capacity_ceiling_mbps", "current_active_build_bandwidth_mbps",
            "allocatable_headroom_mbps", "oversupplied_bandwidth_mbps"]
    lines = ["== 10000244 业务整体容量（RJ V6 口径） =="]
    if target.empty:
        lines.append("  未找到 10000244（可能不在 allowlist/无容量证据）")
    else:
        r = target.iloc[0]
        for c in cols:
            lines.append(f"  {c}: {r.get(c)}")
    # 各层级
    tpool = summary[summary["business"].astype(str).eq("10000244")]
    lines.append("\n== 10000244 各容量池层级（前 12） ==")
    for _, r in tpool.head(12).iterrows():
        lines.append(f"  {r['pool_level']} {r.get('pool_key')}: 状态={r['capacity_state']} "
                     f"上限={r['raw_capacity_ceiling_mbps']:.0f} 当前={r['current_active_build_bandwidth_mbps']:.0f} "
                     f"富余={r['allocatable_headroom_mbps']:.0f} 过建={r['oversupplied_bandwidth_mbps']:.0f}")
    txt = "\n".join(lines)
    (out / "10000244容量核查（新）.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
