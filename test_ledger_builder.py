# -*- coding: utf-8 -*-
"""[wcx2 新增] 账本 v1 构建函数测试（新）：不依赖外部数据源。

运行：MULTIBUSINESS_SUPERSET_COMMON=... python -m unittest test_ledger_builder.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MULTIBUSINESS_SUPERSET_COMMON", str(ROOT / "skills" / "superset-sql-query" / "common"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import build_v3_daily_business_training as v3  # noqa: E402

CHECKER = ROOT / "wcx2实验代码（新）" / "账本契约检查（新）.py"


def fixture_frames():
    pairs = pd.DataFrame({
        "sample_day": ["2026-08-10", "2026-08-10", "2026-08-11"],
        "node_id": ["nodeA", "nodeB", "nodeA"],
        "business": [10000079, 10000280, 10000079],
        "construction_bandwidth_mbps": [5000.0, 100.0, 5000.0],   # nodeB 低于下限
        "cum_cost_7d": [250.0, 78.0, 255.0],
        "cum_revenue_7d": [300.0, 86.0, 305.0],
        "sample_weight": [1.0, 1.0, 0.5],
    })
    audit = pd.DataFrame({
        "node_id": ["nodeA", "nodeB", "nodeA"],
        "sample_day": ["2026-08-10", "2026-08-10", "2026-08-11"],
        "status": ["clean", "clean", "clean"],
        "active_mainstream_count": [1, 1, 1],
    })
    return pairs, audit


class LedgerBuilderTest(unittest.TestCase):
    def test_schema_and_flag(self) -> None:
        pairs, audit = fixture_frames()
        led = v3.build_ledger_v1(pairs, audit, min_build_bandwidth=500.0)
        self.assertEqual(list(led.columns), [
            "sample_day", "node_id", "business", "build_bandwidth_mbps", "cost_amount",
            "revenue_amount", "sample_weight", "active_business_count", "is_primary",
            "bandwidth_below_floor"])
        self.assertEqual(len(led), 3)
        self.assertEqual(int(led["bandwidth_below_floor"].sum()), 1)  # nodeB 100 Mbps
        self.assertTrue(bool(led["is_primary"].all()))
        self.assertAlmostEqual(float(led.loc[0, "cost_amount"]), 250.0)

    def test_multibusiness_marks_non_primary(self) -> None:
        pairs, audit = fixture_frames()
        audit.loc[0, "active_mainstream_count"] = 2
        led = v3.build_ledger_v1(pairs, audit, min_build_bandwidth=0.0)
        self.assertFalse(bool(led.loc[0, "is_primary"]))
        self.assertEqual(int(led.loc[0, "active_business_count"]), 2)

    def test_output_passes_contract_checker(self) -> None:
        pairs, audit = fixture_frames()
        led = v3.build_ledger_v1(pairs, audit, min_build_bandwidth=500.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.csv"
            led.to_csv(path, index=False, encoding="utf-8-sig")
            r = subprocess.run([sys.executable, str(CHECKER), str(path)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
