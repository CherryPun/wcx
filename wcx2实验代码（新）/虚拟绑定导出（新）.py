# -*- coding: utf-8 -*-
"""[wcx2 新增] 虚拟绑定导出（新）

从 Superset `lbattle.jarvis_customer` 导出虚拟业务→真实业务绑定（替代需 NIULINK_AUTH 的接口路径）。
输出与 §5.6 产物同构的 CSV：virtual_business_id, real_business_id, mode, virtual_name。
用法：python "虚拟绑定导出（新）.py"
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "skills" / "superset-sql-query" / "common" / "run_superset_sql.py"
OUT = ROOT / "business_binding_audit"
DAY = "2026-09-13"

SQL = ("SELECT record_id, name, associated_customers FROM lbattle.jarvis_customer "
       "WHERE is_virtual = true")


def parse(js: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in re.finditer(r'"customerId"\s*:\s*\{\s*"\$numberLong"\s*:\s*"(\d+)"\s*\}\s*(?:,\s*"mode"\s*:\s*\{\s*"\$numberLong"\s*:\s*"(\d+)"\s*\})?', js or ""):
        out.append((m.group(1), m.group(2) or "0"))
    return out


def main() -> int:
    OUT.mkdir(exist_ok=True)
    raw = OUT / "_virtual_raw.csv"
    if not raw.exists():
        sqlf = OUT / "_q.sql"
        sqlf.write_text(SQL, encoding="utf-8")
        subprocess.run([sys.executable, str(RUNNER), "--database-id", "19", "--schema", "test",
                        "--sql-file", str(sqlf), "--output-csv", str(raw), "--preview", "1"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
        sqlf.unlink(missing_ok=True)
    if not raw.exists():
        print("查询失败"); return 1
    d = pd.read_csv(raw, dtype=str, low_memory=False)
    rows = []
    for _, r in d.iterrows():
        for real, mode in parse(r.get("associated_customers")):
            rows.append({"virtual_business_id": str(r["record_id"]), "real_business_id": real,
                         "mode": mode, "virtual_name": r.get("name")})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"niulink_virtual_business_bindings_{DAY.replace('-','')}.csv", index=False, encoding="utf-8-sig")
    out.to_csv(OUT / "virtual_bindings_latest.csv", index=False, encoding="utf-8-sig")
    print(f"[out] 虚拟业务 {d['record_id'].nunique()} 个 | 绑定关系 {len(out)} 条")
    print(f"      -> {OUT.name}/virtual_bindings_latest.csv")
    one = out.groupby("virtual_business_id").size()
    print(f"[统计] 唯一绑定(仅1个真实业务) {int((one==1).sum())} 个 | 多义(需消歧) {int((one>1).sum())} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
