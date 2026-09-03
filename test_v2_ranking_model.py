#!/usr/bin/env python3
"""Smoke tests for the V2 profile segment ranker."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import v2_ranking_model as ranking


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "v2_ranking_model.py"


class V2RankingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory(prefix="machine_test_v2_")
        cls.output_dir = Path(cls.tempdir.name)
        subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "build",
                "--limit-nodes",
                "300",
                "--output-dir",
                str(cls.output_dir),
                "--min-business-support",
                "2",
                "--business-map",
                str(HERE / "v1_business_name_map_superset.csv"),
            ],
            cwd=HERE,
            check=True,
            text=True,
            capture_output=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_artifact_check_passes(self) -> None:
        result = subprocess.run(
            ["python3", str(SCRIPT), "check", "--output-dir", str(self.output_dir)],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("V2 artifacts check passed", result.stdout)

    def test_recommendation_columns_include_reasons_and_risk(self) -> None:
        recommendations = pd.read_csv(
            self.output_dir / "v2_node_recommendations.csv",
            dtype={"node_id": "string", "v2_business_top1": "string"},
        )
        required = {
            "node_id",
            "recommendation_mode",
            "v2_business_top1",
            "v2_business_top2",
            "v2_business_top3",
            "v2_reason_top1",
            "v2_risk_level_top1",
            "v2_risk_reasons_top1",
            "v2_top3_detail",
        }
        self.assertTrue(required.issubset(recommendations.columns))
        self.assertGreater(len(recommendations), 0)
        first = recommendations.iloc[0]
        businesses = [str(first[f"v2_business_top{index}"]) for index in range(1, 4)]
        self.assertEqual(len(set(businesses)), 3)
        self.assertFalse(any(value.endswith(".0") for value in businesses))
        self.assertIn(first["v2_risk_level_top1"], {"low", "medium", "high", "unknown"})
        self.assertIn("source_best_rate", first["v2_reason_top1"])

    def test_metrics_and_model_capture_rank_strategy(self) -> None:
        with (self.output_dir / "v2_model_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        self.assertEqual(metrics["version"], "v2")
        self.assertGreater(metrics["test"]["n_nodes"], 0)
        self.assertGreater(metrics["diagnostics"]["business_risk_profiles"], 0)
        self.assertEqual(metrics["diagnostics"]["rank_score_weights"]["source_rate"], 0.6)

        with (self.output_dir / "v2_ranking_model.json").open(encoding="utf-8") as handle:
            model = json.load(handle)
        self.assertEqual(model["model_type"], "profile_segment_ranker")
        self.assertGreater(len(model["feature_fields"]), 0)
        self.assertGreater(len(model["business_risk_profiles"]), 0)

    def test_recommend_node_json(self) -> None:
        recommendations = pd.read_csv(self.output_dir / "v2_node_recommendations.csv", dtype={"node_id": "string"})
        node_id = str(recommendations.iloc[0]["node_id"])
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "recommend-node",
                node_id,
                "--model",
                str(self.output_dir / "v2_ranking_model.json"),
                "--json",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["node_id"], node_id)
        self.assertEqual(payload["recommendation_mode"], "v2_profile_segment_ranker")
        self.assertEqual(len(payload["top3"]), 3)
        self.assertTrue(all(item["business"] for item in payload["top3"]))
        self.assertTrue(all("rank_score_components" in item for item in payload["top3"]))
        self.assertTrue(all(item["risk_level"] in {"low", "medium", "high", "unknown"} for item in payload["top3"]))
        raw_null_markers = {"\\N", "\\\\N", "nan", "NaN", "None", "<NA>"}
        self.assertFalse(any(item["business_name"] in raw_null_markers for item in payload["top3"]))

    def test_recommend_batch_json(self) -> None:
        recommendations = pd.read_csv(self.output_dir / "v2_node_recommendations.csv", dtype={"node_id": "string"})
        node_ids = recommendations["node_id"].head(2).astype(str).tolist()
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "recommend-batch",
                "--node-ids",
                ",".join(node_ids + ["missing-node"]),
                "--model",
                str(self.output_dir / "v2_ranking_model.json"),
                "--json",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["requested"], 3)
        self.assertEqual(payload["matched"], 2)
        self.assertEqual(payload["missing_node_ids"], ["missing-node"])
        self.assertEqual(len(payload["recommendations"]), 2)

    def test_recommend_filter_csv(self) -> None:
        recommendations = pd.read_csv(self.output_dir / "v2_node_recommendations.csv", dtype={"node_id": "string"})
        nodes = pd.read_csv(HERE / "multibusiness_nodes.csv", dtype="string", low_memory=False)
        node = nodes[nodes["node_id"] == recommendations.iloc[0]["node_id"]].iloc[0]
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "recommend-filter",
                "--where",
                f"province={node['province']}",
                "--where",
                f"isp={node['isp']}",
                "--limit",
                "2",
                "--model",
                str(self.output_dir / "v2_ranking_model.json"),
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("node_id", result.stdout)
        self.assertIn("v2_business_top1", result.stdout)

    def test_recommend_segments_csv(self) -> None:
        recommendations = pd.read_csv(self.output_dir / "v2_node_recommendations.csv", dtype={"node_id": "string"})
        nodes = pd.read_csv(HERE / "multibusiness_nodes.csv", dtype="string", low_memory=False)
        node = nodes[nodes["node_id"] == recommendations.iloc[0]["node_id"]].iloc[0]
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "recommend-segments",
                "--where",
                f"province={node['province']}",
                "--where",
                f"isp={node['isp']}",
                "--group-by",
                "city,resourcetype",
                "--min-nodes",
                "1",
                "--limit",
                "3",
                "--model",
                str(self.output_dir / "v2_ranking_model.json"),
                "--recommendations",
                str(self.output_dir / "v2_node_recommendations.csv"),
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("recommended_business", result.stdout)
        self.assertIn("business_share", result.stdout)

    def test_export_condition_report(self) -> None:
        output_csv = self.output_dir / "frontend_report.csv"
        output_json = self.output_dir / "frontend_report.json"
        output_data_js = self.output_dir / "frontend_report_data.js"
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "export-condition-report",
                "--segment",
                "province_isp:province,isp",
                "--min-nodes",
                "1",
                "--model",
                str(self.output_dir / "v2_ranking_model.json"),
                "--recommendations",
                str(self.output_dir / "v2_node_recommendations.csv"),
                "--output-csv",
                str(output_csv),
                "--output-json",
                str(output_json),
                "--output-data-js",
                str(output_data_js),
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertGreater(payload["rows"], 0)
        report = pd.read_csv(output_csv, dtype="string")
        self.assertTrue({
            "segment_level",
            "condition_key",
            "condition_text",
            "node_count",
            "business_top1",
            "business_name_top1",
            "confidence_level",
        }.issubset(report.columns))
        self.assertFalse(report["business_name_top1"].fillna("").eq("").any())
        with output_json.open(encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["rows"], len(report))
        self.assertTrue(output_data_js.exists())
        self.assertIn("window.CONDITION_REPORT_DATA", output_data_js.read_text(encoding="utf-8")[:80])


class V2RankingUnitTest(unittest.TestCase):
    def test_business_display_name_uses_readable_missing_label(self) -> None:
        self.assertEqual(ranking.business_display_name("10000251", ""), "未命名业务")
        self.assertEqual(ranking.business_display_name("10000251", "business_id:10000251"), "未命名业务")
        self.assertEqual(ranking.business_display_name("10000069", "腾讯汇聚P2P"), "腾讯汇聚P2P")

    def test_infer_node_size_type_separates_large_nodes_and_small_boxes(self) -> None:
        self.assertEqual(
            ranking.infer_node_size_type({
                "deliverytype": "smallBox",
                "resourcetype": "aggregation",
                "device_type": "micro.B",
                "bw": "10",
            }),
            ranking.NODE_SIZE_SMALL_BOX,
        )
        self.assertEqual(
            ranking.infer_node_size_type({
                "deliverytype": "idc",
                "resourcetype": "dedicated",
                "device_type": "jarvis.A",
                "bw": "5000",
                "corenum": "64",
            }),
            ranking.NODE_SIZE_LARGE,
        )

    def test_network_schedule_type_combines_isp_and_province_scope(self) -> None:
        cases = [
            ({"isp": "移动", "scheduleisps": "", "analysis_transprovrate": 0}, "本网本省"),
            ({"isp": "电信", "scheduleisps": "电信", "analysis_transprovrate": 100}, "本网出省"),
            ({"isp": "移动", "scheduleisps": "电信", "analysis_transprovrate": 0}, "异网本省"),
            ({"isp": "联通", "scheduleisps": "电信", "analysis_transprovrate": 100}, "异网出省"),
        ]
        for profile, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(ranking.infer_network_schedule_type(profile), expected)

    def test_empty_schedule_is_canonicalized_to_local_isp(self) -> None:
        self.assertEqual(
            ranking.canonical_schedule_isps({"isp": "移动", "scheduleisps": ""}),
            "移动",
        )
        self.assertEqual(
            ranking.canonical_schedule_isps({"isp": "中国联通", "scheduleisps": '["联通"]'}),
            "联通",
        )

    def test_invalid_transprov_value_is_not_treated_as_mixed_province(self) -> None:
        self.assertEqual(
            ranking.infer_network_schedule_type({
                "isp": "移动",
                "scheduleisps_text": "移动|电信",
                "analysis_transprovrate": 50,
            }),
            "混合网络+未知省份",
        )
        self.assertEqual(
            ranking.infer_network_schedule_type({
                "isp": "移动",
                "scheduleisps": "电信",
                "analysis_transprovrate": 10,
            }),
            "异网+未知省份",
        )
        self.assertEqual(
            ranking.infer_node_size_type({
                "deliverytype": "aggregation",
                "analysis_supply_side_delivery_type": "汇聚",
                "device_type": "jarvis.A",
                "bw": "300",
            }),
            ranking.NODE_SIZE_LARGE,
        )

    def test_profile_range_risk_flags_ignore_zero_quality_thresholds(self) -> None:
        model = {
            "business_risk_profiles": {
                "100": {
                    "support_nodes": "100",
                    "maxioutil_count": "100",
                    "maxioutil_p90": "0",
                    "maxioutil_p95": "0",
                }
            },
            "risk_min_profile_count": 20,
        }
        flags = ranking.profile_range_risk_flags({"maxioutil": "1"}, "100", model)
        self.assertEqual(flags, [])

    def test_profile_range_risk_flags_find_business_outliers(self) -> None:
        model = {
            "business_risk_profiles": {
                "100": {
                    "support_nodes": "100",
                    "bw_count": "100",
                    "bw_p05": "500",
                    "bw_p50": "1000",
                    "quality_rtt_count": "100",
                    "quality_rtt_p90": "100",
                    "quality_rtt_p95": "150",
                }
            },
            "risk_min_profile_count": 20,
        }
        flags = ranking.profile_range_risk_flags({"bw": "200", "quality_rtt": "200"}, "100", model)
        levels = [level for level, _ in flags]
        reasons = ";".join(reason for _, reason in flags)
        self.assertIn("high", levels)
        self.assertIn("bw=200", reasons)
        self.assertIn("quality_rtt=200", reasons)


if __name__ == "__main__":
    unittest.main()
