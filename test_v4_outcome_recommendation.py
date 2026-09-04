#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import v4_outcome_recommendation as v4


HERE = Path(__file__).resolve().parent


def synthetic_pairs() -> pd.DataFrame:
    rows = []
    targets = {
        "100": (0.12, 0.01),
        "200": (0.05, 0.10),
        "300": (0.06, 0.04),
    }
    for day in pd.date_range("2026-08-01", periods=12):
        for business, (miner, platform) in targets.items():
            for node_index in range(3):
                bandwidth = 500 + node_index * 100
                rows.append({
                    "node_id": f"n-{business}-{node_index}",
                    "business": business,
                    "business_name": f"业务{business}",
                    "sample_day": day.strftime("%Y-%m-%d"),
                    "sample_weight": 1.0,
                    "cum_cost_7d": miner * bandwidth,
                    "cum_revenue_7d": (miner + platform) * bandwidth,
                    "buildBandwidth": bandwidth,
                    "cost_per_bandwidth_mbps": miner + node_index * 0.001,
                    "profit_per_bandwidth_mbps": platform + node_index * 0.001,
                    "province": "浙江",
                    "city": "杭州",
                    "isp": "电信",
                    "resourcetype": "aggregation",
                    "deliverytype": "aggregation",
                    "nattype": "full",
                    "scheduleisps": "电信",
                    "analysis_transprovrate": 0,
                    "bw": bandwidth,
                })
    return pd.DataFrame(rows)


class V4OutcomeUnitTest(unittest.TestCase):
    def test_model_uses_only_approved_node_profile_features(self) -> None:
        used = {field for _, fields, _ in v4.MODEL_LEVELS for field in fields}
        self.assertEqual(used, {
            "province",
            "isp",
            "network_schedule_type",
            "corenum_bucket",
            "nattype",
            "memtotal_bucket",
            "totaldisksize_bucket",
            "ipv6_capability",
            "bw_bucket",
            "packet_loss_satisfaction_bucket",
        })

    def test_latest_pressure_profile_overrides_older_value_and_keeps_zero(self) -> None:
        with tempfile.TemporaryDirectory(prefix="machine_test_pressure_") as temp:
            path = Path(temp) / "pressure.csv"
            pd.DataFrame([
                {
                    "node_id": "n1",
                    "pressure_snapshot_day": 20260903,
                    "pressure_snapshot_hour": 23,
                    "pressure_last_report_time": 100,
                    "overall_packet_loss_benchmark_satisfaction_pct": 88.0,
                },
                {
                    "node_id": "n1",
                    "pressure_snapshot_day": 20260904,
                    "pressure_snapshot_hour": 9,
                    "pressure_last_report_time": 200,
                    "overall_packet_loss_benchmark_satisfaction_pct": 0.0,
                },
            ]).to_csv(path, index=False)
            enriched = v4.enrich_latest_pressure(pd.DataFrame([{
                "node_id": "n1",
                "overall_packet_loss_benchmark_satisfaction_pct": 55.0,
            }]), path)
            self.assertEqual(
                enriched.loc[0, "overall_packet_loss_benchmark_satisfaction_pct"],
                0.0,
            )
            prepared = v4.prepare_features(enriched)
            self.assertEqual(prepared.loc[0, "packet_loss_satisfaction_bucket"], "<= 50")

    def test_temporal_split_keeps_future_dates_out_of_training(self) -> None:
        pairs = synthetic_pairs()
        pairs["sample_day"] = pd.to_datetime(pairs["sample_day"])
        train, calibration, test, windows = v4.temporal_split(
            pairs, validation_days=4, calibration_days=2
        )
        self.assertLess(train["sample_day"].max(), calibration["sample_day"].min())
        self.assertLess(calibration["sample_day"].max(), test["sample_day"].min())
        self.assertEqual(windows["test_end"], "2026-08-12")

    def test_business_models_predict_miner_and_platform_separately(self) -> None:
        pairs = synthetic_pairs()
        pairs["sample_day"] = pd.to_datetime(pairs["sample_day"])
        pairs[v4.TARGET_MINER] = pairs["cost_per_bandwidth_mbps"]
        pairs[v4.TARGET_PLATFORM] = pairs["profit_per_bandwidth_mbps"]
        prepared = v4.prepare_features(pairs)
        model = v4.fit_model(
            prepared,
            ["100", "200", "300"],
            {"100": "业务100", "200": "业务200", "300": "业务300"},
            min_segment_support=1,
        )
        row = prepared.iloc[0].to_dict()
        a = v4.predict_business(row, "100", model)
        b = v4.predict_business(row, "200", model)
        self.assertGreater(a[v4.TARGET_MINER], b[v4.TARGET_MINER])
        self.assertGreater(b[v4.TARGET_PLATFORM], a[v4.TARGET_PLATFORM])

    def test_concentration_diagnostic_flags_dominant_top1(self) -> None:
        recommendations = pd.DataFrame([
            {"business_top1": "100", "business_name_top1": "业务A"}
            for _ in range(9)
        ] + [{"business_top1": "200", "business_name_top1": "业务B"}])
        _, summary = v4.concentration_diagnostics(recommendations)
        self.assertEqual(summary["concentration_risk"], "high")
        self.assertAlmostEqual(summary["maximum_top1_share"], 0.9)

    def test_action_never_prioritizes_platform_interval_crossing_zero(self) -> None:
        item = {
            "confidence": "high",
            v4.TARGET_PLATFORM: 0.02,
            "platform_low90": -0.01,
        }
        self.assertEqual(v4.recommendation_action(item, 0.20), "谨慎人工复核")
        item[v4.TARGET_PLATFORM] = -0.01
        self.assertEqual(v4.recommendation_action(item, 0.20), "暂不推荐执行")


class V4OutcomeSmokeTest(unittest.TestCase):
    def test_build_and_check_complete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="machine_test_v4_") as temp:
            root = Path(temp)
            pairs = synthetic_pairs()
            pairs_path = root / "pairs.csv"
            nodes_path = root / "nodes.csv"
            map_path = root / "business_map.csv"
            output_dir = root / "output"
            pairs.to_csv(pairs_path, index=False)
            pairs.drop_duplicates("node_id")[[
                "node_id", "province", "city", "isp", "resourcetype", "deliverytype",
                "nattype", "scheduleisps", "analysis_transprovrate", "bw",
            ]].to_csv(nodes_path, index=False)
            pd.DataFrame([
                {"business": business, "business_name": f"业务{business}"}
                for business in ["100", "200", "300"]
            ]).to_csv(map_path, index=False)
            missing_profiles = root / "missing_profiles.csv"

            subprocess.run([
                "python3", str(HERE / "v4_outcome_recommendation.py"), "build",
                "--pairs", str(pairs_path),
                "--historical-profiles", str(missing_profiles),
                "--current-nodes", str(nodes_path),
                "--current-business", str(root / "missing_current_business.csv"),
                "--business-map", str(map_path),
                "--source-summary", str(root / "missing_source_summary.json"),
                "--output-dir", str(output_dir),
                "--validation-days", "4",
                "--calibration-days", "2",
                "--min-business-support", "1",
                "--min-business-nodes", "1",
                "--min-segment-support", "1",
            ], cwd=HERE, check=True, text=True, capture_output=True)

            result = subprocess.run([
                "python3", str(HERE / "v4_outcome_recommendation.py"), "check",
                "--output-dir", str(output_dir),
            ], cwd=HERE, check=True, text=True, capture_output=True)
            self.assertIn("V4 artifacts check passed", result.stdout)
            summary = json.loads((output_dir / v4.OUTPUT_SUMMARY).read_text(encoding="utf-8"))
            self.assertEqual(summary["training_data"]["candidate_businesses"], 3)
            self.assertTrue((output_dir / v4.OUTPUT_REPORT_HTML).exists())
            recommendations = pd.read_csv(output_dir / v4.OUTPUT_RECOMMENDATIONS)
            self.assertTrue({
                "business_top1", "miner_unit_income_top1", "platform_unit_profit_top1",
                "recommendation_confidence", "recommendation_action", "current_business_status",
                "cpu_bucket", "memory_bucket", "disk_bucket", "ipv6_capability",
                "bandwidth_bucket", "packet_loss_satisfaction_pct",
            }.issubset(recommendations.columns))


if __name__ == "__main__":
    unittest.main()
