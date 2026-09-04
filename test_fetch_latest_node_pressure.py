#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import fetch_latest_node_pressure as pressure


class LatestNodePressureTest(unittest.TestCase):
    def test_load_node_ids_deduplicates_multiple_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="latest_pressure_") as temp:
            root = Path(temp)
            first = root / "first.csv"
            second = root / "second.csv"
            pd.DataFrame({"node_id": ["n2", "n1", ""]}).to_csv(first, index=False)
            pd.DataFrame({"node_id": ["n2", "n3"]}).to_csv(second, index=False)
            self.assertEqual(pressure.load_node_ids([first, second]), ["n1", "n2", "n3"])

    def test_sql_selects_latest_valid_snapshot_per_node(self) -> None:
        sql = pressure.latest_pressure_sql(["n1", "n2"], dt.date(2026, 9, 4))
        normalized = " ".join(sql.split()).lower()
        self.assertIn("nj.day = 20260904", normalized)
        self.assertIn("partition by nj._id", normalized)
        self.assertIn("order by nj.hour desc, nj.nodeinfo.lastreporttime desc", normalized)
        self.assertIn("limitbwachieverate is not null", normalized)
        self.assertIn("where rn = 1", normalized)


if __name__ == "__main__":
    unittest.main()
