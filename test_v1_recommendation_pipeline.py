#!/usr/bin/env python3
"""Smoke tests for the V1 recommendation pipeline.

These tests intentionally avoid pytest and sklearn/scipy so they can run in the
current project environment with only the standard library plus pandas.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import v1_recommendation_pipeline as pipeline


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "v1_recommendation_pipeline.py"


class V1PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory(prefix="machine_test_v1_")
        cls.output_dir = Path(cls.tempdir.name)
        subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "build",
                "--limit-nodes",
                "500",
                "--output-dir",
                str(cls.output_dir),
                "--min-business-support",
                "3",
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
        self.assertIn("V1 artifacts check passed", result.stdout)

    def test_recommendation_columns_and_top3(self) -> None:
        recommendations = pd.read_csv(
            self.output_dir / "v1_node_recommendations.csv",
            dtype={"node_id": "string", "combined_top1": "string", "combined_top2": "string", "combined_top3": "string"},
        )
        required = {
            "node_id",
            "recommendation_mode",
            "combined_top1",
            "combined_top2",
            "combined_top3",
            "combined_top3_detail",
            "recommendation_reason_top1",
            "risk_level",
            "risk_reasons",
            "missing_profile_fields",
        }
        self.assertTrue(required.issubset(recommendations.columns))
        self.assertGreater(len(recommendations), 0)
        first = recommendations.iloc[0]
        businesses = [first["combined_top1"], first["combined_top2"], first["combined_top3"]]
        self.assertEqual(len(set(businesses)), 3)
        self.assertFalse(any(str(value).endswith(".0") for value in businesses))
        detail_parts = str(first["combined_top3_detail"]).split(";")
        self.assertLessEqual(len(detail_parts), 3)
        self.assertTrue(all(":" in part for part in detail_parts if part))
        self.assertTrue(set(recommendations["recommendation_mode"]).issubset({
            "observed_node_outcome",
            "model_segment_fallback",
        }))

    def test_business_names_do_not_leak_raw_null_markers(self) -> None:
        recommendations = pd.read_csv(self.output_dir / "v1_node_recommendations.csv", dtype="string")
        raw_null_markers = {"\\N", "\\\\N", "nan", "NaN", "None", "<NA>"}
        for column in [name for name in recommendations.columns if "business_name" in name]:
            values = set(recommendations[column].fillna("").astype(str).unique())
            self.assertFalse(values & raw_null_markers)

    def test_correction_columns_and_actions(self) -> None:
        corrections = pd.read_csv(
            self.output_dir / "v1_existing_node_correction_candidates.csv",
            dtype={"node_id": "string", "current_business": "string", "combined_top1": "string"},
        )
        required = {
            "node_id",
            "current_business",
            "current_business_source",
            "current_business_score_source",
            "current_eq_top1",
            "current_in_top3",
            "suggested_action",
            "reason_summary",
        }
        self.assertTrue(required.issubset(corrections.columns))
        self.assertGreater(len(corrections), 0)
        allowed = {
            "current_business_unknown",
            "keep_top1",
            "keep_in_top3_review",
            "review_switch_candidate",
            "review_current_business_not_in_model",
            "observe",
        }
        self.assertTrue(set(corrections["suggested_action"].dropna()).issubset(allowed))

    def test_metrics_json(self) -> None:
        with (self.output_dir / "v1_model_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        self.assertEqual(metrics["version"], "v1")
        self.assertGreater(metrics["diagnostics"]["training_pair_rows"], 0)
        self.assertGreater(metrics["diagnostics"]["candidate_businesses"], 0)
        self.assertGreater(metrics["diagnostics"]["business_risk_profile_rows"], 0)
        warning = metrics["diagnostics"]["current_business_warning"]
        self.assertIn("historical latest business_online_day", warning)
        self.assertIn("current business policy", warning)

    def test_business_risk_profile(self) -> None:
        profile = pd.read_csv(self.output_dir / "v1_business_risk_profile.csv", dtype="string")
        required = {
            "business",
            "business_name",
            "support_nodes",
            "combined_score_mean",
            "cum_profit_7d_mean",
        }
        self.assertTrue(required.issubset(profile.columns))
        self.assertGreater(len(profile), 0)

    def test_recommend_node_json(self) -> None:
        recommendations = pd.read_csv(self.output_dir / "v1_node_recommendations.csv", dtype={"node_id": "string"})
        node_id = str(recommendations.iloc[0]["node_id"])
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "recommend-node",
                node_id,
                "--model",
                str(self.output_dir / "v1_business_score_model.json"),
                "--json",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["node_id"], node_id)
        self.assertIn(payload["recommendation_mode"], {"observed_node_outcome", "model_segment_fallback"})
        self.assertIn(payload["risk_level"], {"low", "medium", "high", "unknown"})
        self.assertGreaterEqual(len(payload["top3"]), 1)
        self.assertLessEqual(len(payload["top3"]), 3)
        self.assertTrue(all(item["business"] for item in payload["top3"]))
        self.assertTrue(all("reason" in item for item in payload["top3"]))
        raw_null_markers = {"\\N", "\\\\N", "nan", "NaN", "None", "<NA>"}
        self.assertFalse(any(item["business_name"] in raw_null_markers for item in payload["top3"]))

    def test_recommend_filter(self) -> None:
        nodes = pd.read_csv(HERE / "multibusiness_nodes.csv", dtype="string")
        province = str(nodes["province"].dropna().iloc[0])
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "recommend-filter",
                "--where",
                f"province={province}",
                "--limit",
                "3",
                "--model",
                str(self.output_dir / "v1_business_score_model.json"),
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("node_id", result.stdout)
        self.assertIn("combined_top1", result.stdout)

    def test_audit_node_json(self) -> None:
        recommendations = pd.read_csv(self.output_dir / "v1_node_recommendations.csv", dtype={"node_id": "string"})
        node_id = str(recommendations.iloc[0]["node_id"])
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "audit-node",
                node_id,
                "--output-dir",
                str(self.output_dir),
                "--json",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["node_id"], node_id)
        self.assertIn("current_business", payload)
        self.assertIn("recommendation", payload)
        self.assertIn("decision", payload)
        self.assertIn("risk", payload)
        self.assertGreaterEqual(len(payload["recommendation"]["top3"]), 1)

    def test_correction_candidates_filter(self) -> None:
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "correction-candidates",
                "--output-dir",
                str(self.output_dir),
                "--action",
                "review_switch_candidate",
                "--limit",
                "5",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("node_id", result.stdout)
        self.assertIn("suggested_action", result.stdout)


class V1PipelineUnitTest(unittest.TestCase):
    def test_clean_cell_normalizes_raw_null_markers(self) -> None:
        for value in [None, float("nan"), "\\N", "\\\\N", "nan", "NaN", "None", "<NA>"]:
            self.assertEqual(pipeline.clean_cell(value), "")

    def test_clean_cell_preserves_integer_like_ids(self) -> None:
        self.assertEqual(pipeline.clean_cell("10000184.0"), "10000184")

    def test_derive_current_business_from_wide(self) -> None:
        wide = pd.DataFrame([
            {
                "nodeId": "node-a",
                "customerId": "100",
                "customerName": "业务A",
                "day": "2026-08-18",
                "state": "online",
                "peak95": "10",
                "revenue_finalAmount": "1",
                "cost_finalAmount": "0.4",
                "profit_profitAmount": "0.6",
            },
            {
                "nodeId": "node-a",
                "customerId": "100",
                "customerName": "业务A",
                "day": "2026-08-19",
                "state": "online",
                "peak95": "20",
                "revenue_finalAmount": "2",
                "cost_finalAmount": "0.8",
                "profit_profitAmount": "1.2",
            },
            {
                "nodeId": "node-a",
                "customerId": "200",
                "customerName": "业务B",
                "day": "2026-08-19",
                "state": "online",
                "peak95": "999",
                "revenue_finalAmount": "99",
                "cost_finalAmount": "1",
                "profit_profitAmount": "98",
            },
            {
                "nodeId": "node-b",
                "customerId": "300",
                "customerName": "业务C",
                "day": "2026-08-19",
                "state": "offline",
                "peak95": "999",
            },
            {
                "nodeId": "node-b",
                "customerId": "400",
                "customerName": "业务D",
                "day": "2026-08-19",
                "state": "online",
                "peak95": "1",
            },
        ])
        current = pipeline.derive_current_business_from_wide(wide, lookback_days=7, min_active_days=2)
        selected = current.set_index("node_id")
        self.assertEqual(selected.loc["node-a", "current_business"], "100")
        self.assertEqual(selected.loc["node-a", "current_business_confidence"], "wide_high")
        self.assertEqual(selected.loc["node-b", "current_business"], "400")
        self.assertEqual(selected.loc["node-b", "current_business_confidence"], "wide_medium")

    def test_normalize_current_business_aliases_and_business_map(self) -> None:
        frame = pd.DataFrame([{"nodeId": "node-a", "customerId": "100.0", "day": "2026-08-19"}])
        current = pipeline.normalize_current_business_frame(frame, business_name_map={"100": "业务A"})
        row = current.iloc[0]
        self.assertEqual(row["node_id"], "node-a")
        self.assertEqual(row["current_business"], "100")
        self.assertEqual(row["current_business_name"], "业务A")

    def test_current_business_override_falls_back_to_proxy_for_missing_nodes(self) -> None:
        nodes = pd.DataFrame([
            {"node_id": "node-a", "province": "江苏", "isp": "电信", "resourcetype": "aggregation"},
            {"node_id": "node-b", "province": "浙江", "isp": "联通", "resourcetype": "aggregation"},
        ])
        outcomes = pd.DataFrame([
            {
                "node_id": "node-a",
                "business": "100",
                "business_name": "业务A",
                "business_online_day": "2026-08-18",
                "cum_cost_7d": 1,
                "cum_revenue_7d": 2,
                "outcome_distinct_days": 1,
                "business_active_days": 1,
            },
            {
                "node_id": "node-b",
                "business": "200",
                "business_name": "业务B",
                "business_online_day": "2026-08-18",
                "cum_cost_7d": 1,
                "cum_revenue_7d": 2,
                "outcome_distinct_days": 1,
                "business_active_days": 1,
            },
        ])
        recommendations = pd.DataFrame([
            {"node_id": "node-a", "recommended_business_top1": "100", "recommended_score_top1": 1.0},
            {"node_id": "node-b", "recommended_business_top1": "200", "recommended_score_top1": 1.0},
        ])
        current = pd.DataFrame([
            {
                "node_id": "node-a",
                "current_business": "100",
                "current_business_name": "业务A",
                "current_business_source": "authoritative_test",
                "current_business_confidence": "authoritative",
            }
        ])
        model = {
            "candidate_businesses": ["100", "200"],
            "global_scores": {
                "100": {"score": 1, "support": 1, "business_name": "业务A"},
                "200": {"score": 1, "support": 1, "business_name": "业务B"},
            },
            "segment_levels": [],
            "feature_fields": [],
        }
        corrections = pipeline.build_correction_candidates(
            nodes,
            pipeline.prepare_outcomes(outcomes),
            recommendations,
            model,
            min_switch_gap=0.05,
            top_k=3,
            current_business=current,
        ).set_index("node_id")
        self.assertEqual(corrections.loc["node-a", "current_business_source"], "authoritative_test")
        self.assertEqual(
            corrections.loc["node-b", "current_business_source"],
            pipeline.CURRENT_BUSINESS_HISTORY_SOURCE,
        )

    def test_unit_yield_uses_construction_bandwidth_not_actual_bandwidth(self) -> None:
        outcomes = pd.DataFrame([
            {
                "node_id": "node-a",
                "business": "100",
                "cum_cost_7d": 100,
                "cum_revenue_7d": 120,
                "buildBandwidth": 500,
                "bw": 800,
                "actualbandwidth": 2000,
            },
            {
                "node_id": "node-b",
                "business": "100",
                "cum_cost_7d": 100,
                "cum_revenue_7d": 120,
                "buildBandwidth": 1000,
                "bw": 800,
                "actualbandwidth": 100,
            },
        ])
        prepared = pipeline.prepare_outcomes(outcomes, target_mode="unit_bandwidth")
        self.assertEqual(prepared["construction_bandwidth_mbps"].tolist(), [500.0, 1000.0])
        self.assertEqual(prepared["cost_per_bandwidth_mbps"].tolist(), [0.2, 0.1])

    def test_construction_bandwidth_falls_back_to_nominal_bw(self) -> None:
        frame = pd.DataFrame([
            {"buildBandwidth": None, "bw": 600, "actualbandwidth": 2000},
            {"buildBandwidth": 0, "bw": 800, "actualbandwidth": 3000},
        ])
        self.assertEqual(pipeline.construction_bandwidth_mbps(frame).tolist(), [600.0, 800.0])

    def test_weighted_group_summary_gives_each_run_its_effective_weight(self) -> None:
        frame = pd.DataFrame([
            {"business": "100", "combined_score": 0.0, "sample_weight": 0.25},
            {"business": "100", "combined_score": 1.0, "sample_weight": 0.75},
        ])

        summary = pipeline.weighted_group_summary(
            frame,
            ["business"],
            ["combined_score"],
        ).iloc[0]

        self.assertEqual(summary["raw_support"], 2)
        self.assertAlmostEqual(summary["effective_support"], 1.0)
        self.assertAlmostEqual(summary["combined_score"], 0.75)


if __name__ == "__main__":
    unittest.main()
