# -*- coding: utf-8 -*-
"""一键复现校验：在交付目录内重跑关键脚本，并校验关键数字是否与正式产物一致。
步骤：
1. 01_temporal_validation/stage_diag_coldstart.py
2. 01_temporal_validation/stage_p0_temporal.py
3. 02_main_recommendation/stage_e2e.py
4. 02_main_recommendation/build_final_html.py
校验：
- 各脚本正常退出（returncode 0）
- 关键 JSON 指标落在预期容差内
用法：python run_verify_all.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("01_diag", "01_temporal_validation/stage_diag_coldstart.py", "01_temporal_validation/_rerun/coldstart_baseline_metrics.json",
     {"eval_nodes": (999, 999), "global.hit@1": (0.21, 0.22)}),
    ("01_p0", "01_temporal_validation/stage_p0_temporal.py", "01_temporal_validation/_rerun/final_top3_temporal_metrics.json",
     {"real_best_hit_at_1": (0.0, 0.001), "real_best_hit_at_3": (0.003, 0.006)}),
    ("02_main", "02_main_recommendation/stage_e2e.py", "02_main_recommendation/_rerun/e2e_metrics.json",
     {"candidate_quality.test_real_best_hit_at_1": (0.355, 0.365),
      "candidate_quality.test_real_best_hit_at_3": (0.650, 0.660)}),
    ("02_html", "02_main_recommendation/build_final_html.py", "06_html/全节点推荐_最终版.html", None),
]


def get_path(data, dotted):
    cur = data
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def check(expect, actual):
    if expect is None:
        return True
    lo, hi = expect
    return lo <= float(actual) <= hi


def main():
    ok_all = True
    for name, script, out_rel, expects in STEPS:
        print("=" * 60)
        print("运行", name, script)
        r = subprocess.run([PY, str(ROOT / script)], capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()[-1:] + (r.stderr or "").strip().splitlines()[-1:]
        print("  returncode:", r.returncode)
        if r.returncode != 0:
            ok_all = False
            print("  FAILED", "\n".join(tail))
            continue
        if out_rel is None:
            print("  ok (产物存在性由脚本保证)")
            continue
        p = ROOT / out_rel
        if not p.exists():
            print("  缺少产物", p)
            ok_all = False
            continue
        if not (expects or {}):
            print("  ok（产物存在）", p.name)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        row_ok = True
        for dotted, expect in (expects or {}).items():
            actual = get_path(data, dotted)
            good = check(expect, actual)
            row_ok &= good
            print(f"  {dotted}: {actual} 期望[{expect}] -> {'OK' if good else 'MISMATCH'}")
        ok_all &= row_ok
    print("=" * 60)
    print("总体:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
