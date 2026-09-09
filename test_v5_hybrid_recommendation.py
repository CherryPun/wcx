#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import v4_outcome_recommendation as v4
import v5_hybrid_recommendation as v5


class V5HybridRecommendationTest(unittest.TestCase):
    def test_filter_supported_schedule_rows_keeps_only_four_defined_types(self) -> None:
        frame = pd.DataFrame({
            "network_schedule_type": ["本网本省", "异网出省", "混合网络+本省", "未知网络+本省"]
        })
        filtered, excluded = v5.filter_supported_schedule_rows(frame)
        self.assertEqual(filtered["network_schedule_type"].tolist(), ["本网本省", "异网出省"])
        self.assertEqual(excluded, 2)

    def test_filter_training_window_recomputes_consecutive_run_weights(self) -> None:
        frame = pd.DataFrame([
            {"node_id": "n1", "business": "A", "sample_day": f"2026-09-{day:02d}"}
            for day in range(1, 6)
        ])
        frame["sample_weight"] = 0.2
        frame["consecutive_valid_days"] = 5

        filtered = v5.filter_training_window(frame, "2026-09-03", "2026-09-05")

        self.assertEqual(filtered["sample_day"].tolist(), [
            "2026-09-03", "2026-09-04", "2026-09-05",
        ])
        self.assertEqual(filtered["consecutive_valid_days"].tolist(), [3, 3, 3])
        self.assertAlmostEqual(filtered["sample_weight"].sum(), 1.0)

    def test_node_effect_is_business_agnostic_and_shrunk(self) -> None:
        frame = pd.DataFrame({
            "node_id": ["n1", "n1", "n2"],
            "business": ["a", "b", "a"],
            "sample_weight": [1.0, 1.0, 1.0],
            v4.TARGET_MINER: [3.0, 5.0, 2.0],
            v4.TARGET_PLATFORM: [1.0, 3.0, 1.0],
        })
        predictions = {
            v4.TARGET_MINER: np.array([2.0, 4.0, 2.0]),
            v4.TARGET_PLATFORM: np.array([0.0, 2.0, 1.0]),
        }
        effects = v5.fit_node_effects(frame, predictions, alpha=2.0)
        self.assertAlmostEqual(effects[v4.TARGET_MINER]["n1"], 0.5)
        self.assertAlmostEqual(effects[v4.TARGET_PLATFORM]["n1"], 0.5)
        self.assertEqual(effects[v4.TARGET_MINER]["n2"], 0.0)

    @unittest.skipIf(v5.xgb is None, "xgboost is not installed in this interpreter")
    def test_cli_build_and_check_small_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pairs_path = root / "pairs.csv"
            profiles_path = root / "profiles.csv"
            current_path = root / "current.csv"
            map_path = root / "business_map.csv"
            output_dir = root / "output"

            profiles: list[dict[str, object]] = []
            pairs: list[dict[str, object]] = []
            businesses = ["100", "200", "300"]
            provinces = ["江苏", "浙江", "四川"]
            for business_index, business in enumerate(businesses):
                for node_index in range(6):
                    node_id = f"node-{business}-{node_index}"
                    profiles.append({
                        "node_id": node_id,
                        "province": provinces[node_index % 3],
                        "isp": ["电信", "联通", "移动"][node_index % 3],
                        "nattype": ["full", "nat1", "nat4"][node_index % 3],
                        "scheduleisps": ["电信", "联通", "移动"][node_index % 3],
                        "analysis_transprovrate": 0 if node_index % 2 == 0 else 100,
                        "analysis_issupportipv6": node_index % 2 == 0,
                        "bw": 500 + 500 * node_index,
                        "corenum": 8 + 4 * node_index,
                        "memtotal": 16 + 8 * node_index,
                        "totaldisksize": 1000 + 500 * node_index,
                    })
                    for day in range(14):
                        miner = 0.02 + business_index * 0.01 + node_index * 0.001 + day * 0.0001
                        platform = 0.01 + business_index * 0.004 + node_index * 0.0005
                        bandwidth = 500 + 500 * node_index
                        pairs.append({
                            "node_id": node_id,
                            "business": business,
                            "sample_day": f"2026-08-{day + 1:02d}",
                            "cum_cost_7d": miner * bandwidth,
                            "cum_revenue_7d": (miner + platform) * bandwidth,
                            "buildBandwidth": bandwidth,
                            "consecutive_valid_days": 14,
                            "sample_weight": 1 / 14,
                        })
            pd.DataFrame(pairs).to_csv(pairs_path, index=False)
            profile_frame = pd.DataFrame(profiles)
            profile_frame.to_csv(profiles_path, index=False)
            profile_frame.head(9).to_csv(current_path, index=False)
            pd.DataFrame([
                {"business": business, "business_name": f"业务{business}"}
                for business in businesses
            ]).to_csv(map_path, index=False)

            command = [
                sys.executable,
                str(Path(__file__).resolve().parent / "v5_hybrid_recommendation.py"),
                "build",
                "--pairs", str(pairs_path),
                "--historical-profiles", str(profiles_path),
                "--latest-pressure-profiles", str(root / "missing_pressure.csv"),
                "--current-nodes", str(current_path),
                "--current-business", str(root / "missing_current.csv"),
                "--business-map", str(map_path),
                "--source-summary", str(root / "missing_summary.json"),
                "--output-dir", str(output_dir),
                "--validation-days", "4",
                "--calibration-days", "2",
                "--min-business-support", "0.1",
                "--min-business-nodes", "1",
                "--min-segment-support", "0.1",
                "--boost-rounds", "20",
                "--early-stopping-rounds", "5",
            ]
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent, capture_output=True)
            checked = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parent / "v5_hybrid_recommendation.py"),
                 "check", "--output-dir", str(output_dir)],
                check=True,
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
            )
            self.assertIn("V5 artifacts check passed", checked.stdout)
            summary = json.loads((output_dir / v5.OUTPUT_SUMMARY).read_text(encoding="utf-8"))
            self.assertEqual(summary["version"], "v5")
            self.assertEqual(summary["training_data"]["candidate_businesses"], 3)
            self.assertTrue((output_dir / v5.OUTPUT_REPORT_HTML).exists())
            reloaded = v5.load_runtime(output_dir)
            prepared_nodes = v4.prepare_features(profile_frame.head(2))
            reloaded_recommendations = v5.recommend_nodes(prepared_nodes, reloaded)
            self.assertEqual(len(reloaded_recommendations), 2)
            self.assertTrue({
                "business_top1", "business_top2", "business_top3",
                "interval_dominates_current", "cpu_bucket", "memory_bucket", "disk_bucket",
                "ipv6_capability", "bandwidth_bucket", "packet_loss_satisfaction_pct",
            }.issubset(reloaded_recommendations.columns))


if __name__ == "__main__":
    unittest.main()
