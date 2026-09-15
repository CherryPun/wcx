# -*- coding: utf-8 -*-
"""[wcx2 新增] 成本归因规则回归测试（新）：脱敏合成样本，不依赖桌面 xlsx。

镜像 `成本归因复算（新）.py` 的规则（机房级/业务项级），确保公式不被改坏。
运行：python -m unittest test_attribution_rules.py
"""
from __future__ import annotations

import unittest

import pandas as pd


def recompute(room: pd.DataFrame, item: pd.DataFrame) -> pd.DataFrame:
    """按研发口径复算：调账额、分摊权重、追加成本、调整后成本。"""
    r = room.copy()
    r["复算调账额"] = (
        r["昨日_机房成本"] * r["前日_CDN成本占比"] - r["昨日_CDN成本"]
    ).clip(lower=0)
    i = item.merge(r[["机房名", "复算调账额"]], on="机房名", how="left")
    share = i["计费带宽下降G"] / i.groupby("机房名")["计费带宽下降G"].transform("sum")
    i["复算权重"] = share
    i["复算追加成本"] = i["复算调账额"] * share
    i["复算调整后成本"] = i["昨日_成本"] + i["复算追加成本"]
    return i


def synthetic():
    room = pd.DataFrame({
        "机房名": ["r1", "r2"],
        "昨日_机房成本": [10000.0, 20000.0],
        "前日_CDN成本占比": [0.60, 0.30],
        "昨日_CDN成本": [5000.0, 7000.0],
    })
    item = pd.DataFrame({
        "机房名": ["r1", "r1", "r2"],
        "计费带宽下降G": [10.0, 30.0, 20.0],
        "昨日_成本": [1000.0, 2000.0, 3000.0],
    })
    return room, item


class AttributionRulesTest(unittest.TestCase):
    def test_reallocation_formula(self) -> None:
        room, _ = synthetic()
        r = recompute(room, synthetic()[1])
        # r1: max(0, 10000*0.60 - 5000) = 1000 ; r2: max(0, 20000*0.30 - 7000) = 0（负值截断）
        got = r.groupby("机房名")["复算调账额"].first().to_dict()
        self.assertAlmostEqual(got["r1"], 1000.0, places=6)
        self.assertAlmostEqual(got["r2"], 0.0, places=6)

    def test_weight_allocation_by_bandwidth_drop(self) -> None:
        i = recompute(*synthetic())
        w = i[i["机房名"] == "r1"].set_index("计费带宽下降G")["复算权重"].to_dict()
        self.assertAlmostEqual(w[10.0], 0.25, places=6)
        self.assertAlmostEqual(w[30.0], 0.75, places=6)

    def test_total_reallocation_is_zero_sum(self) -> None:
        """内部调账不改机房总成本：CDN 追加 = 其他业务冲回（方向相反、绝对额相等）。"""
        i = recompute(*synthetic())
        self.assertAlmostEqual(i["复算追加成本"].sum(), 1000.0, places=6)

    def test_adjusted_cost_monotonic(self) -> None:
        i = recompute(*synthetic())
        delta = i["复算调整后成本"] - i["昨日_成本"]
        self.assertTrue((delta >= -1e-9).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
