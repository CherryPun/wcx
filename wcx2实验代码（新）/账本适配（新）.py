# -*- coding: utf-8 -*-
"""[wcx2 新增] 账本适配（新）

把现有 V3 pairs（legacy 列名）适配成冻结的账本 schema v1（`ledger_schema（新）.md`），
供 `账本契约检查（新）.py` 在真实数据上跑契约。

映射：
  construction_bandwidth_mbps -> build_bandwidth_mbps
  cum_cost_7d                 -> cost_amount       （列名是历史遗留，实际为单日值）
  cum_revenue_7d              -> revenue_amount
  sample_weight               -> sample_weight
多业务标记：由 `node_day_sampling_audit_v3.csv` 的 clean 行给出（count=1、primary=True）。

用法：
  python "账本适配（新）.py" [--pairs <csv>] [--audit <csv>] [--output <csv>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）"
PAIRS = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"
AUDIT = V3 / "node_day_sampling_audit_v3.csv"
OUTPUT = V3 / "ledger_node_business_day_v1（新）.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    ap.add_argument("--audit", type=Path, default=AUDIT)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    usecols = ["sample_day", "node_id", "business", "construction_bandwidth_mbps",
               "cum_cost_7d", "cum_revenue_7d", "sample_weight"]
    p = pd.read_csv(args.pairs, usecols=usecols, low_memory=False)
    led = p.rename(columns={
        "construction_bandwidth_mbps": "build_bandwidth_mbps",
        "cum_cost_7d": "cost_amount",
        "cum_revenue_7d": "revenue_amount",
    })
    led["business"] = pd.to_numeric(led["business"], errors="coerce").astype("int64")

    if args.audit.exists():
        a = pd.read_csv(args.audit, usecols=["node_id", "sample_day", "status", "active_mainstream_count"], low_memory=False)
        a = a[a["status"] == "clean"]
        led = led.merge(a[["node_id", "sample_day", "active_mainstream_count"]],
                        on=["node_id", "sample_day"], how="left")
    led["active_business_count"] = pd.to_numeric(led.get("active_mainstream_count"), errors="coerce").fillna(1).astype(int)
    led["is_primary"] = led["active_business_count"].eq(1)
    led = led.drop(columns=[c for c in ("active_mainstream_count",) if c in led.columns])

    cols = ["sample_day", "node_id", "business", "build_bandwidth_mbps", "cost_amount",
            "revenue_amount", "sample_weight", "active_business_count", "is_primary"]
    led = led[cols]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[ledger] {len(led):,} 行 -> {args.output.name} | 节点日 {led.groupby(['sample_day','node_id']).ngroups:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
