# -*- coding: utf-8 -*-
"""V2 全库产物验收：对已入库交付产物做阈值断言（无数据即可运行）。

- outputs/exp_metrics.json 与 outputs/large_review_fullpool.csv
- 数值阈值按 2026-09-09 交付基线冻结；变化时先改这里再改文档。
用法：python run_verify_v2.py [--verbose]
可选 --refull：指向全库账/画像的 env 就绪时重跑 stage_e2e.py 后再验（数据缺失则提示跳过）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "07_v2_iteration" / "outputs"

EXP = {
    "test_nodes": (2994, 2994),          # 时间外测试节点 n
    "train_nodes": (11000, 12500),
    "nodes_with_rec": (14000, 16000),
    "hit1": (0.19, 0.23),                # 家族粒 hit@1
    "hit3": (0.49, 0.55),                # 家族粒 hit@3
    "wape": (0.0, 0.46),
    "r2": (0.60, 1.0),
    "profit_positive": (0.90, 1.0),
}
REVIEW_COLS = ["node_id", "true_best", "tb_fam", "top1_business_id", "top1_label", "hit1", "hit3"]


def get_path(d, dotted):
    cur = d
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def chk(name, lo, hi, val, fail):
    ok = (val is not None) and (lo <= float(val) <= hi)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {val} (期望 {lo}~{hi})")
    fail.append((name, val) if not ok else None)
    return ok


def main():
    verbose = "--verbose" in sys.argv
    fail = []
    p_met = OUT / "exp_metrics.json"
    if not p_met.exists():
        print("FAIL 缺 outputs/exp_metrics.json")
        return 1
    m = json.loads(p_met.read_text(encoding="utf-8"))
    print("== exp_metrics.json 阈值断言 ==")
    chk("test_nodes", *EXP["test_nodes"], m.get("test_nodes"), fail)
    chk("train_nodes", *EXP["train_nodes"], m.get("train_nodes"), fail)
    chk("candidate_quality.nodes_with_rec", *EXP["nodes_with_rec"],
        get_path(m, "candidate_quality.nodes_with_rec"), fail)
    chk("candidate_quality.hit@1(家族粒)", *EXP["hit1"],
        get_path(m, "candidate_quality.test_real_best_hit_at_1"), fail)
    chk("candidate_quality.hit@3(家族粒)", *EXP["hit3"],
        get_path(m, "candidate_quality.test_real_best_hit_at_3"), fail)
    chk("candidate_quality.profit_positive", *EXP["profit_positive"],
        get_path(m, "candidate_quality.top1_profit_positive_rate"), fail)
    chk("amount.cost_wape", *EXP["wape"], get_path(m, "amount.cost_wape"), fail)
    chk("amount.cost_r2", *EXP["r2"], get_path(m, "amount.cost_r2"), fail)

    p_rev = OUT / "large_review_fullpool.csv"
    if p_rev.exists():
        import pandas as pd
        df = pd.read_csv(p_rev, dtype=str)
        print("== large_review_fullpool.csv 结构断言 ==")
        miss = [c for c in REVIEW_COLS if c not in df.columns]
        ok_cols = not miss
        print(f"  [{'OK' if ok_cols else 'FAIL'}] 必需列齐全 (缺: {miss})")
        if not ok_cols:
            fail.append(("review_cols", miss))
        ok_n = len(df) == 2994
        print(f"  [{'OK' if ok_n else 'FAIL'}] 行数 {len(df)} == 2994")
        if not ok_n:
            fail.append(("review_rows", len(df)))
        top1_empty = df["top1_business_id"].isna().mean() if "top1_business_id" in df else 1.0
        ok_fill = top1_empty < 0.01
        print(f"  [{'OK' if ok_fill else 'FAIL'}] top1 非空率 {1 - top1_empty:.4f} >= 0.99")
        if not ok_fill:
            fail.append(("review_top1_fill", 1 - top1_empty))
    else:
        print("WARN 缺 outputs/large_review_fullpool.csv，跳过结构断言")

    if "--refull" in sys.argv:
        sub_ = subprocess.run([sys.executable, str(HERE / "stage_e2e.py")], cwd=str(REPO), capture_output=True, text=True)
        if sub_.returncode != 0:
            print("FAIL refull 重跑 stage_e2e.py:", (sub_.stderr or "")[-800:])
            return 1
        print("refull: stage_e2e.py 重跑完成（数字与文档一致性请人工复核）")

    ok = not any(fail)
    print("=" * 40)
    print("V2 产物验收:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
