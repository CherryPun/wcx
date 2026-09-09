# -*- coding: utf-8 -*-
"""冒烟跑通：用入库的小样本子集（v2_data/smoke/）零配置跑整条流水线。

- 只证明"clone 后流水线可跑通 + 结构性合理"，不承诺/不复现全库 21.0 数字。
- 设置 V2_GATE=0（门禁依赖 5% 混合账，冒烟数据无 ant）；MIN_SUPPORT/KNN 用小值保证子集有池。
用法：python run_smoke_v2.py   → 输出 PASS/FAIL
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable

SMOKE_ATTRS = HERE / "smoke" / "smoke_attrs.csv"
SMOKE_OUT = HERE / "smoke" / "smoke_outcomes.csv"
METRICS = HERE / "_rerun" / "e2e_metrics.json"


def main():
    if not (SMOKE_ATTRS.exists() and SMOKE_OUT.exists()):
        print("FAIL 缺 smoke 子集，请先 python make_smoke.py 重建")
        return 1
    env = dict(os.environ)
    env.update({
        "V2_POOL": "nonant",
        "V2_MERGE": "qiniu_name",
        "V2_GATE": "0",                 # 冒烟无 ant/无 5% 混合账
        "V2_MIN_SUPPORT": "2",
        "V2_KNN_K": "10",
        "V2_ATTRS": str(SMOKE_ATTRS),
        "V2_OUTCOMES": str(SMOKE_OUT),
    })
    r = subprocess.run([PY, str(HERE / "stage_e2e.py")], env=env, cwd=str(REPO),
                       capture_output=True, text=True)
    tail = "\n".join((r.stdout or "").strip().splitlines()[-3:])
    if r.returncode != 0:
        print("FAIL stage_e2e.py 退出码非 0")
        print(tail)
        print((r.stderr or "")[-800:])
        return 1
    print(tail)
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    cq = m.get("candidate_quality", {})
    fail = []
    def chk(name, cond, val):
        print(f"  [{'OK' if cond else 'FAIL'}] {name}: {val}")
        if not cond:
            fail.append(name)
    chk("train_nodes>0", m.get("train_nodes", 0) > 0, m.get("train_nodes"))
    chk("test_nodes>0", m.get("test_nodes", 0) > 0, m.get("test_nodes"))
    chk("nodes_with_rec>0", cq.get("nodes_with_rec", 0) > 0, cq.get("nodes_with_rec"))
    chk("候选池>=5", len(m.get("top1_distribution", {})) >= 5, len(m.get("top1_distribution", {})))
    amt = m.get("amount", {})
    chk("金额有限(wape 有限)", amt.get("cost_wape", -1) >= 0 and amt.get("cost_wape", -1) < 10, amt.get("cost_wape"))
    # 空候选率：profile_nonant.csv 的 fallback（测试节点）
    pf = HERE / "_rerun" / "profile_nonant.csv"
    if pf.exists():
        import pandas as pd
        df = pd.read_csv(pf, dtype=str)
        df["fallback"] = pd.to_numeric(df["fallback"], errors="coerce")
        t = df[df["is_test"] == "True"] if "True" in set(df["is_test"]) else df[df["is_test"] == True]  # noqa: E712
        rate = 1 - t["fallback"].mean() if len(t) else 1.0
        chk(f"测试节点候选非空率>=0.9 (n={len(t)})", rate >= 0.9, round(rate, 3))
    ok = not fail
    print("=" * 40)
    print("冒烟流水线:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
