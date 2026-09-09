# -*- coding: utf-8 -*-
"""数据就绪自检：核对复现输入 CSV 的 schema/规模是否符合 data_manifest 预期。

用法（指向全库账/画像即可）：
    python check_data_ready.py
    python check_data_ready.py --outcomes <账.csv> --attrs <画像.csv>
默认按 env V2_OUTCOMES/V2_ATTRS，未设则指向仓库本地 _rerun/exp_*.csv（若存在）。
不做数据入库，只做结构性自检，把 runbook 第 0 步从人工变可执行。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAN = json.loads((HERE / "data_manifest.json").read_text(encoding="utf-8"))
ATTRS_REQ = MAN["inputs"][1]["required_cols"]
OUT_REQ = MAN["inputs"][0]["required_cols"]


def fail(msg):
    print("FAIL", msg)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=os.getenv("V2_OUTCOMES", ""))
    ap.add_argument("--attrs", default=os.getenv("V2_ATTRS", ""))
    a = ap.parse_args()
    out_p = Path(a.outcomes) if a.outcomes else (HERE / "_rerun" / "exp_outcomes_clean.csv")
    att_p = Path(a.attrs) if a.attrs else (HERE / "_rerun" / "exp_attrs.csv")

    ok = True
    for p, req, name, rol in ((att_p, ATTRS_REQ, "画像", "attrs"),
                              (out_p, OUT_REQ, "账", "outcomes")):
        print(f"== {name} {p} ==")
        if not p.exists():
            ok = fail(f"缺文件 {p}（内网拉取/重建后重试，见 EXEC_expansion_runbook.md）")
            continue
        df = pd.read_csv(p, dtype=str, low_memory=False)
        miss = [c for c in req if c not in df.columns]
        if miss:
            ok = fail(f"缺必需列 {miss}")
        else:
            print(f"  OK 必需列齐全 ({len(req)} 列)")
        print(f"  行数 {len(df):,}")
        if rol == "attrs":
            if "bw" not in df.columns:
                ok = fail("数值带宽列必须叫 bw（buildBandwidth 需改名），否则不进相似矩阵")
            ant = df["node_id"].astype(str).str.startswith("ant").sum()
            if ant:
                print(f"  WARN attrs 含 ant 前缀 {ant} 行（nonant 池会过滤，确认预期）")
            lo = MAN["inputs"][1]["row_expected_min"]
            if len(df) < lo:
                ok = fail(f"画像行数 {len(df)} < 期望 {lo}")
            else:
                print(f"  OK 画像规模 >= {lo}")
        else:
            g7 = df["outcome_days"].astype(float).ge(7).sum() if "outcome_days" in df else 0
            print(f"  满7天行 {g7:,}")
            lo = MAN["inputs"][0]["row_expected_min"]
            if g7 < lo:
                ok = fail(f"满7天行 {g7} < 期望 {lo}")
            else:
                print(f"  OK 满7天行 >= {lo}")
            if "business_name" in df.columns:
                print(f"  业务名缺失行 {df['business_name'].isna().sum():,}（七牛归并依赖 business_name，缺失该族不合成）")
    print("=" * 40)
    print("数据就绪:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
