#!/usr/bin/env python3
from __future__ import annotations

import unittest

import compare_v5_training_windows as comparison


def fake_summary(max_share: float, low_share: float, businesses: int = 20) -> dict:
    return {
        "date_window": {"start": "2026-08-01", "end": "2026-09-01"},
        "temporal_windows": {"test_start": "2026-08-29", "test_end": "2026-09-01"},
        "training_data": {
            "raw_node_days": 100,
            "effective_run_support": 50,
            "nodes": 20,
            "candidate_businesses": businesses,
        },
        "temporal_validation": {
            "miner": {"rmse": 1.0, "r2": 0.1},
            "platform": {"rmse": 1.0, "r2": 0.1},
        },
        "ranking_and_dr_diagnostics": {"observed_business_hit_at_3": 0.2},
        "recommendation_concentration": {
            "maximum_top1_share": max_share,
            "effective_business_count": 3,
            "concentration_risk": "high" if max_share > 0.7 else "low",
        },
        "credibility": {
            "low_confidence_node_share": low_share,
            "top1_platform_interval_crosses_zero_share": 0.5,
            "level": "low",
        },
    }


class CompareV5TrainingWindowsTest(unittest.TestCase):
    def test_high_concentration_blocks_challenger_promotion(self) -> None:
        result = comparison.compare(fake_summary(0.4, 0.6), fake_summary(0.8, 0.6))
        self.assertFalse(result["decision"]["promote_challenger"])
        self.assertEqual(result["decision"]["selected"], "31-day primary")

    def test_healthy_challenger_is_promoted(self) -> None:
        result = comparison.compare(fake_summary(0.4, 0.6), fake_summary(0.5, 0.62))
        self.assertTrue(result["decision"]["promote_challenger"])


if __name__ == "__main__":
    unittest.main()
