# -*- coding: utf-8 -*-
"""[wcx2 新增] 账本契约检查（新）

校验"节点×业务×日"账本（C1 schema v1）的列、键唯一性、数值完整性、取值范围与业务白名单。
fail-fast：任何违规返回退出码 1；同时打印 n/分母，防止"静默丢样本"。

用法：
  python "账本契约检查（新）.py" <ledger.csv> [--allowlist mainstream_business_allowlist.csv]
                                                [--window-start 2026-08-10] [--window-end 2026-09-09]
schema v1：
  sample_day(date) node_id(str) business(int64 canonical) build_bandwidth_mbps(>0)
  cost_amount revenue_amount(可空) sample_weight(>0) active_business_count(>=1) is_primary(bool)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED = ["sample_day", "node_id", "business", "build_bandwidth_mbps", "sample_weight"]
OPTIONAL = ["cost_amount", "revenue_amount", "active_business_count", "is_primary", "effective_support"]
QINIU_CANONICAL = 10000280

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<44} {detail}")
    if not ok:
        fails.append(name)


def load_allowlist(path: Path | None) -> set[int] | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    col = "business" if "business" in df.columns else df.columns[0]
    return set(pd.to_numeric(df[col], errors="coerce").dropna().astype("int64"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ledger", type=Path)
    ap.add_argument("--allowlist", type=Path, default=None)
    ap.add_argument("--window-start", default=None)
    ap.add_argument("--window-end", default=None)
    args = ap.parse_args()

    if not args.ledger.exists():
        print(f"[SKIP] 账本不存在：{args.ledger}")
        return 0
    df = pd.read_csv(args.ledger, low_memory=False)
    print(f"# 账本契约检查：{args.ledger.name}  n={len(df):,} 列={len(df.columns)}")

    missing = [c for c in REQUIRED if c not in df.columns]
    check("必需列齐全", not missing, f"缺={missing}" if missing else " ".join(REQUIRED))
    if missing:
        return finish()

    # 完整性：必需列不得有 NaN（pandas 聚合会静默跳过）
    for c in REQUIRED:
        na = int(df[c].isna().sum())
        check(f"列无空值: {c}", na == 0, f"NaN={na}/{len(df)}")

    dup = int(df.duplicated(["sample_day", "node_id", "business"]).sum())
    check("键唯一 (day,node,business)", dup == 0, f"重复={dup}")

    bw = pd.to_numeric(df["build_bandwidth_mbps"], errors="coerce")
    check("建设带宽 > 0", bool((bw > 0).all()), f"min={bw.min()}")
    w = pd.to_numeric(df["sample_weight"], errors="coerce")
    check("样本权重 > 0", bool((w > 0).all()), f"min={w.min()}")

    if "active_business_count" in df.columns:
        abc = pd.to_numeric(df["active_business_count"], errors="coerce")
        multi = int((abc > 1).sum())
        check("多业务标记存在 (active_business_count>=1)", bool((abc >= 1).all()),
              f"多业务行={multi}（V5 语义需整天剔除，份额模式保留）")
    if "is_primary" in df.columns:
        prim = df.assign(_p=df["is_primary"].astype(str).str.lower().isin(["true", "1"]))
        grp = prim.groupby(["sample_day", "node_id"])["_p"].sum()
        check("每节点日至多一个 is_primary", bool((grp <= 1).all()), f"违例={int((grp > 1).sum())}")
        if "active_business_count" in df.columns:
            abc = pd.to_numeric(df["active_business_count"], errors="coerce")
            single = df[abc == 1]
            missing = int((~single["is_primary"].astype(str).str.lower().isin(["true", "1"])).sum())
            check("单业务节点日必须 is_primary=True", missing == 0, f"缺失={missing}")

    # 分母对账
    print(f"[分母] 节点日 {df.groupby(['sample_day','node_id']).ngroups:,} | "
          f"节点×业务行 {len(df):,} | 天数 {df['sample_day'].nunique()} | 节点 {df['node_id'].nunique():,}")

    if args.window_start and args.window_end:
        days = pd.to_datetime(df["sample_day"], errors="coerce")
        ok = bool((days >= args.window_start).all() and (days <= args.window_end).all())
        check("日期在声明窗口内", ok, f"{days.min().date()}~{days.max().date()}")

    allow = load_allowlist(args.allowlist)
    if allow is not None:
        biz = set(pd.to_numeric(df["business"], errors="coerce").dropna().astype("int64"))
        extra = sorted(biz - allow)
        check("业务在白名单内", not extra, f"越界={extra[:5]}")
    else:
        print("[SKIP] 未提供 allowlist")

    if QINIU_CANONICAL in set(pd.to_numeric(df["business"], errors="coerce").dropna().astype("int64")):
        print(f"[info] 含七牛 canonical {QINIU_CANONICAL}（七牛口径已归一）")

    return finish()


def finish() -> int:
    if fails:
        print("\nFAILED: " + "; ".join(fails))
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
