# -*- coding: utf-8 -*-
"""[wcx2 新增] 业务级可信度与候选降级（新）

按业务读取 `v5_business_training_metrics.csv`，判定"证据不足"，产出：
  - 业务级可信度（新）.csv ：逐业务指标 + credibility + 原因
  - 业务阻止清单（新）.txt ：证据不足的业务 ID（每行一个）
用法：python "业务级可信度（新）.py" [V5输出目录]
V5 侧通过环境变量 `V5_BUSINESS_BLOCKLIST` 指向该清单后，命中业务不再给出可执行 Top1（降级为 hold）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "recent_month_large_mainstream_v5_sw_7c2.0（新）"
MIN_TEST_SUPPORT = 100.0
MIN_TEST_ROWS = 50
COV_LO, COV_HI = 0.85, 0.95


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    src = out / "v5_business_training_metrics.csv"
    m = pd.read_csv(src, low_memory=False)
    m = m[m["is_candidate"].astype(str).str.lower().isin(["true", "1"])].copy()

    for c in ("test_effective_support", "test_rows", "miner_r2", "platform_r2", "joint_average_coverage90"):
        m[c] = pd.to_numeric(m[c], errors="coerce")

    reasons = []
    for _, r in m.iterrows():
        why = []
        if not (r["test_effective_support"] >= MIN_TEST_SUPPORT and r["test_rows"] >= MIN_TEST_ROWS):
            why.append("support")
        if (r["miner_r2"] < 0) and (r["platform_r2"] < 0):
            why.append("both_r2_neg")
        if not (COV_LO <= r["joint_average_coverage90"] <= COV_HI):
            why.append("coverage")
        reasons.append("|".join(why))
    m["credibility"] = ["证据不足" if why else "可用" for why in reasons]
    m["insufficient_reasons"] = reasons

    cols = ["business", "business_name", "test_rows", "test_effective_support", "test_nodes",
            "miner_r2", "platform_r2", "joint_average_coverage90", "credibility", "insufficient_reasons"]
    table = m[cols].sort_values(["credibility", "test_effective_support"])
    out_csv = ROOT / "业务级可信度（新）.csv"
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")

    blocked = table[table["credibility"] == "证据不足"]["business"].astype(str).tolist()
    bl_path = ROOT / "业务阻止清单（新）.txt"
    bl_path.write_text("\n".join(blocked) + ("\n" if blocked else ""), encoding="utf-8")

    print(f"{'业务':<10}{'名称':<18}{'test支持':>9}{'矿R2':>8}{'平台R2':>8}{'覆盖':>7}{'判定':>8}  原因")
    for _, r in table.iterrows():
        print(f"{str(r['business']):<10}{str(r['business_name'])[:16]:<18}{r['test_effective_support']:>9.1f}"
              f"{r['miner_r2']:>8.3f}{r['platform_r2']:>8.3f}{r['joint_average_coverage90']:>7.3f}"
              f"{r['credibility']:>8}  {r['insufficient_reasons']}")
    print(f"\n候选业务 {len(table)} 个 | 证据不足 {len(blocked)} 个 -> 写入 {bl_path.name}")
    print("启用降级：$env:V5_BUSINESS_BLOCKLIST=\"" + str(bl_path) + "\"")


if __name__ == "__main__":
    main()
