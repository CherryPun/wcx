# -*- coding: utf-8 -*-
"""[wcx2 新增] 压测特征按日对齐（新）

把 snapshots/bench_YYYYMMDD.csv（178 天历史压测）按 (node_id, sample_day) join 进匹训练 pairs，
生成 as-of 压测特征列，替代"最新值复用整窗"的静态口径。

新增列：
  bench_satisfaction_pct   总极限/应交付 ×100（达成率）
  bench_limit_mbps         总极限带宽
  bench_expected_mbps      应交付带宽
  bench_deliver_ratio      总极限 / 建设带宽（可交付比；虚报/端口不足时显著 <1）
  packet_loss_satisfaction_bucket  分桶（V5 白名单同一列名，供训练直接使用）

用法：python "压测特征对齐（新）.py" [--pairs <csv>] [--output <csv>]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "snapshots"
DEFAULT_PAIRS = ROOT / "recent_month_large_mainstream_v3_6m（新）" / "v1_training_pairs_large_mainstream_recent_1m.csv"
DEFAULT_OUT = ROOT / "recent_month_large_mainstream_v3_6m（新）" / "v1_training_pairs_bench_asof（新）.csv"
BUCKETS = [0, 50, 80, 95, 99.5, 101]


def load_bench() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(SNAP / "bench_*.csv"))):
        d = pd.read_csv(f, low_memory=False, usecols=[
            "node_id", "pressure_snapshot_day", "overall_packet_loss_benchmark_satisfaction_pct",
            "packet_loss_benchmark_bw_total_mbps", "pressure_expected_bw_total_mbps"])
        parts.append(d)
    b = pd.concat(parts, ignore_index=True)
    b["sample_day"] = pd.to_datetime(b["pressure_snapshot_day"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    b = b.rename(columns={
        "overall_packet_loss_benchmark_satisfaction_pct": "bench_satisfaction_pct",
        "packet_loss_benchmark_bw_total_mbps": "bench_limit_mbps",
        "pressure_expected_bw_total_mbps": "bench_expected_mbps",
    })
    return b.groupby(["node_id", "sample_day"], as_index=False).last()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    b = load_bench()
    print(f"[bench] 快照 {b['sample_day'].nunique()} 天 / {len(b):,} 行")
    pairs = pd.read_csv(args.pairs, low_memory=False)
    print(f"[pairs] {len(pairs):,} 行")
    m = pairs.merge(b, on=["node_id", "sample_day"], how="left")
    cov = m["bench_satisfaction_pct"].notna().mean()
    m["bench_deliver_ratio"] = (pd.to_numeric(m["bench_limit_mbps"], errors="coerce")
                                / pd.to_numeric(m["construction_bandwidth_mbps"], errors="coerce"))
    m["packet_loss_satisfaction_bucket"] = pd.cut(
        pd.to_numeric(m["bench_satisfaction_pct"], errors="coerce"), bins=BUCKETS, right=False
    ).astype(str).replace("nan", "missing")
    print(f"[join] 压测覆盖 {cov:.1%} | 达成分位: " +
          ", ".join(f"P{int(q*100)}={m['bench_satisfaction_pct'].quantile(q):.1f}" for q in (0.1, 0.5, 0.9)))
    print(f"[join] 可交付比(极限/建设带宽) 中位 {m['bench_deliver_ratio'].median():.3f} | "
          f"<0.5 占比 {(m['bench_deliver_ratio'] < 0.5).mean():.1%}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[out] -> {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
