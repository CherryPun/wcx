#!/usr/bin/env python3
from __future__ import annotations

import unittest

import pandas as pd

import build_business_price_report as report


class BusinessPriceReportTest(unittest.TestCase):
    def test_price_types_remain_separate_and_dominant_uses_weight(self) -> None:
        rows = []
        for day, price_type, weight in [
            ("2026-09-01", "day95avg", 0.7),
            ("2026-09-02", "day95avg", 0.7),
            ("2026-09-03", "month95", 1.0),
        ]:
            rows.append({
                "node_id": day,
                "business": "100",
                "business_name": "业务A",
                "daily_isp": "移动",
                "sample_day": day,
                "sample_weight": weight,
                "cum_cost_7d": 10,
                "cum_revenue_7d": 12,
                "miner_price_type": price_type,
                "miner_price_item_id": price_type,
                "miner_price_item_name": price_type,
                "miner_unit_price": 2400 if price_type == "day95avg" else 3000,
                "miner_price_after_bonus": 0,
                "customer_price_item_id": "revenue",
                "customer_price_item_name": "收入项",
                "customer_unit_price": 2600,
            })

        signatures, dominant = report.build_price_signatures(pd.DataFrame(rows))

        self.assertEqual(len(signatures), 2)
        self.assertEqual(len(dominant), 1)
        self.assertEqual(dominant.iloc[0]["miner_price_type"], "day95avg")
        self.assertAlmostEqual(dominant.iloc[0]["signature_support_share"], 1.4 / 2.4)

    def test_rows_without_any_observed_price_are_excluded(self) -> None:
        facts = pd.DataFrame([{
            "node_id": "n1", "business": "100", "business_name": "业务A",
            "daily_isp": "电信", "sample_day": "2026-09-01", "sample_weight": 1,
            "cum_cost_7d": 1, "cum_revenue_7d": 2,
        }])
        signatures, dominant = report.build_price_signatures(facts)
        self.assertTrue(signatures.empty)
        self.assertTrue(dominant.empty)


if __name__ == "__main__":
    unittest.main()
