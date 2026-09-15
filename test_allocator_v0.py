# -*- coding: utf-8 -*-
"""[wcx2 新增] 分配器 v0 测试（新）：合成用例，验证凹目标优于 rank-then-cap 且尊重约束。

运行：python -m unittest test_allocator_v0.py
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "wcx2实验代码（新）" / "分配器v0（新）.py"
_spec = importlib.util.spec_from_file_location("allocator_v0", SCRIPT)
alloc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alloc_mod)


def crafted():
    items = pd.DataFrame({
        "pool": ["p1", "p2", "p1", "p2"],
        "business": [1, 1, 2, 2],
        "max_mbps": [300.0, 300.0, 300.0, 300.0],
        "g_pb": [1.0, 1.0, 0.8, 0.8],
    })
    pool_cap = pd.Series({"p1": 100.0, "p2": 100.0})
    biz_cap = pd.Series({1: 200.0, 2: 200.0})
    knots = pd.DataFrame({"g_mean": [1.0, 0.8], "k1": [50.0, 200.0], "k2": [200.0, 400.0]}, index=[1, 2])
    return items, pool_cap, biz_cap, knots


def concave_value(alloc: pd.Series, knots: pd.DataFrame) -> float:
    total = 0.0
    for biz in knots.index:
        x = float(alloc.xs(biz, level=1).sum()) if biz in set(alloc.index.get_level_values(1)) else 0.0
        k1, k2, g = knots.loc[biz, "k1"], knots.loc[biz, "k2"], knots.loc[biz, "g_mean"]
        total += min(x, k1) * g
        total += min(max(x - k1, 0.0), k2 - k1) * 0.6 * g
        total += max(x - k2, 0.0) * 0.3 * g
    return total


class AllocatorV0Test(unittest.TestCase):
    def test_lp_beats_greedy_under_concavity(self) -> None:
        items, pool_cap, biz_cap, knots = crafted()
        lp = alloc_mod.solve_lp(items, pool_cap, biz_cap, knots)
        gr = alloc_mod.greedy(items, pool_cap, biz_cap)
        self.assertGreater(concave_value(lp, knots), concave_value(gr, knots) + 1e-6,
                           "边际递减下 LP 应优于按单位收益贪心")

    def test_constraints_respected(self) -> None:
        items, pool_cap, biz_cap, knots = crafted()
        lp = alloc_mod.solve_lp(items, pool_cap, biz_cap, knots)
        per_pool = lp.groupby(level=0).sum()
        per_biz = lp.groupby(level=1).sum()
        for p, v in per_pool.items():
            self.assertLessEqual(v, pool_cap[p] + 1e-6)
        for b, v in per_biz.items():
            self.assertLessEqual(v, biz_cap[b] + 1e-6)

    def test_planned_overflow_zero(self) -> None:
        items, pool_cap, biz_cap, knots = crafted()
        lp = alloc_mod.solve_lp(items, pool_cap, biz_cap, knots)
        demand = pd.Series({1: 600.0, 2: 600.0})
        m = alloc_mod.metrics(lp, pool_cap, biz_cap, demand, "lp")
        self.assertEqual(m["planned溢出(Gbps)"], 0.0)
        self.assertGreater(m["扩容缺口(Gbps)"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
