# -*- coding: utf-8 -*-
"""[wcx2 新增] 上线门槛看板（新）

按 `节点业务推荐模型V5全流程说明.md` §17 的九条门槛，从 V5 产物直接出数（PASS/FAIL/待定）。
用法：python "门槛看板（新）.py" [V5输出目录] [pairs.csv]
默认输出目录：recent_month_large_mainstream_v5_sw_7c2.0（新）；默认 pairs：V3 日粒度 pairs。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "recent_month_large_mainstream_v5_sw_7c2.0（新）"
DEFAULT_PAIRS = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）" / "v1_training_pairs_large_mainstream_recent_1m.csv"
rows: list[tuple[str, str, str, str]] = []


def add(name: str, ok: str, value: str, note: str = "") -> None:
    rows.append((name, ok, value, note))


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    pairs = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PAIRS
    s = json.loads((out / "v5_training_summary.json").read_text(encoding="utf-8"))
    sel, base = s["model_comparison"]["selected"], s["model_comparison"]["baseline"]
    cold = s["model_comparison"].get("cold_start_test", {})

    # 1 画像覆盖：按 V5 **真实加载源**（historical-profiles，节点静态属性）计算，
    #   注意不能用磁盘上的 pairs CSV（那里没有这些列，会造成 0% 误报）。
    prof = globals().get("PROFILES", ROOT / "recent_month_large_1d（新）" / "multibusiness_nodes_large_recent_1m.csv")
    if Path(prof).exists():
        cols = pd.read_csv(prof, nrows=1).columns
        d = pd.read_csv(prof, low_memory=False)
        field_alias = {
            "corenum": ["corenum_bucket", "corenum"],
            "memtotal": ["memtotal_bucket", "memtotal"],
            "totaldisksize": ["totaldisksize_bucket", "totaldisksize"],
            "ipv6": ["ipv6_capability", "dial_ipv6_enable", "join_isipv6schedule"],
        }
        parts, covs = [], []
        for target, alias in field_alias.items():
            hit = next((a for a in alias if a in cols), None)
            cov = float(d[hit].notna().mean()) if hit else 0.0
            covs.append(cov)
            parts.append(f"{target} {cov:.0%}")
        cov = min(covs)
        add("训练核心画像覆盖率 ≥95%", "PASS" if cov >= 0.95 else "FAIL", " / ".join(parts),
            f"源={Path(prof).name}（V5 加载时会补入训练帧）")
    else:
        add("训练核心画像覆盖率 ≥95%", "UNKNOWN", "-", "historical-profiles 缺失")

    # 2 压测 as-of
    align = str(s["training_data"].get("latest_pressure_time_alignment", ""))
    pcov = float(s["training_data"].get("latest_pressure_training_coverage") or 0)
    if pcov == 0:
        add("压测特征历史 as-of 对齐", "FAIL", f"coverage={pcov:.0%}", "本轮未启用该特征（不影响现有指标）")
    else:
        add("压测特征历史 as-of 对齐", "PASS" if "as-of" in align or "as of" in align else "FAIL", align[:40])

    # 3/4 时间外 R²
    add("矿主时间外 R² ≥0.10", "PASS" if sel["miner"]["r2"] >= 0.10 else "FAIL", f'{sel["miner"]["r2"]:.4f}', "全样本口径")
    add("平台时间外 R² ≥0.10", "PASS" if sel["platform"]["r2"] >= 0.10 else "FAIL", f'{sel["platform"]["r2"]:.4f}', "全样本口径")

    # 5 冷启动平台
    pr = cold.get("platform", {}).get("r2")
    add("冷启动平台 R² ≥0", "PASS" if (pr is not None and pr >= 0) else "FAIL", f"{pr:.4f}" if pr is not None else "-")

    # 6 区间覆盖
    mc, pc = sel.get("miner_interval_coverage90"), sel.get("platform_interval_coverage90")
    ok6 = mc is not None and pc is not None and all(0.85 <= v <= 0.95 for v in (mc, pc))
    add("90% 区间覆盖稳定在 85%~95%", "PASS" if ok6 else "FAIL", f"矿主 {mc:.3f} / 平台 {pc:.3f}")

    # 7/8 推荐侧
    rec = out / "v5_node_recommendations.csv"
    if rec.exists():
        r = pd.read_csv(rec, low_memory=False)
        conf = pd.to_numeric(r["recommendation_confidence"], errors="coerce")
        share = float((conf < 0.5).mean())
        add("低置信度节点占比 <50%", "PASS" if share < 0.5 else "FAIL", f"{share:.1%}", "confidence<0.5")
        lo, hi = pd.to_numeric(r["platform_unit_low90_top1"], errors="coerce"), pd.to_numeric(r["platform_unit_high90_top1"], errors="coerce")
        cross = float(((lo <= 0) & (hi >= 0)).mean())
        add("Top1 平台区间跨零 <50%", "PASS" if cross < 0.5 else "FAIL", f"{cross:.1%}")
    else:
        add("低置信度节点占比 <50%", "UNKNOWN", "-", "缺 recommendations")
        add("Top1 平台区间跨零 <50%", "UNKNOWN", "-", "缺 recommendations")

    # 9 滚动回测
    rolls = sorted(ROOT.glob("recent_month_large_mainstream_v5_roll_w*"))
    if len(rolls) >= 3:
        add("滚动回测多窗稳定", "PASS" if len(rolls) >= 4 else "FAIL", f"{len(rolls)} 窗", "见 滚动回测（新）.md（需≥4窗且 bw≥500）")
    else:
        add("滚动回测多窗稳定", "UNKNOWN", f"{len(rolls)} 窗")

    # 10 小流量
    add("严格切换候选通过小流量前瞻", "UNKNOWN", "-", "需权限/组织决策")

    print(f"# 上线门槛看板（输出目录：{out.name}）\n")
    print(f"{'门槛':<26}{'判定':<9}{'数值':<22}说明")
    for n, ok, v, note in rows:
        print(f"{n:<26}{ok:<9}{v:<22}{note}")
    n_pass = sum(1 for _, ok, _, _ in rows if ok == "PASS")
    n_fail = sum(1 for _, ok, _, _ in rows if ok == "FAIL")
    n_unk = sum(1 for _, ok, _, _ in rows if ok == "UNKNOWN")
    print(f"\nPASS {n_pass} / FAIL {n_fail} / 待定 {n_unk} —— 结论：**未达上线门槛，仅可作人工参考**")


if __name__ == "__main__":
    main()
