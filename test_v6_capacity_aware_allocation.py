#!/usr/bin/env python3
from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

import v6_capacity_aware_allocation as v6


class V6CapacityAwareAllocationTest(unittest.TestCase):
    def test_current_capacity_snapshot_is_joined_by_node_day_and_business(self) -> None:
        recommendation = self.recommendation("n1", 0.4)
        recommendation["current_business_day"] = "2026-09-01"
        facts = pd.DataFrame([{
            "node_id": "n1", "sample_day": "2026-09-01", "business": "A",
            "daily_province": "浙江", "daily_isp": "移动",
            "network_schedule_type": "异网出省", "schedule_target_isp": "联通",
            "buildBandwidth": 100,
        }])

        output = v6.attach_current_capacity_snapshot(
            pd.DataFrame([recommendation]), facts,
        ).iloc[0]

        self.assertEqual(output["capacity_current_province"], "浙江")
        self.assertEqual(output["capacity_current_isp"], "移动")
        self.assertEqual(output["capacity_current_network_schedule_type"], "异网出省")
        self.assertEqual(output["capacity_current_schedule_target_isp"], "联通")
        self.assertEqual(output["capacity_target_schedule_target_isp"], "联通")
        self.assertEqual(output["capacity_current_build_bandwidth_mbps"], 100)

    def test_daily_utilization_uses_observed_bandwidth_coverage(self) -> None:
        facts = pd.DataFrame([
            {
                "node_id": "n1", "sample_day": "2026-09-01", "business": "A",
                "business_name": "业务A", "daily_isp": "移动", "daily_province": "浙江",
                "network_schedule_type": "本网本省", "buildBandwidth": 100,
                "capacity_observed": True, "capacity_peak95_mbps": 50,
                "node_utilization": 0.5, "capacity_peak95_source": "peak95",
                "miner_income": 10, "platform_revenue": 20, "platform_profit": 10,
            },
            {
                "node_id": "n2", "sample_day": "2026-09-01", "business": "A",
                "business_name": "业务A", "daily_isp": "移动", "daily_province": "浙江",
                "network_schedule_type": "本网本省", "buildBandwidth": 100,
                "capacity_observed": False, "capacity_peak95_mbps": None,
                "node_utilization": None, "capacity_peak95_source": "missing",
                "miner_income": 10, "platform_revenue": 20, "platform_profit": 10,
            },
        ])

        pools = v6.aggregate_capacity_pools(facts)
        row = pools[pools["pool_level"].eq("business")].iloc[0]

        self.assertEqual(row["active_build_bandwidth_mbps"], 200)
        self.assertEqual(row["observed_build_bandwidth_mbps"], 100)
        self.assertEqual(row["bandwidth_observation_coverage"], 0.5)
        self.assertEqual(row["observed_utilization"], 0.5)
        self.assertEqual(row["estimated_total_peak95_mbps"], 100)

    def test_capacity_pool_splits_same_cross_isp_type_by_target_carrier(self) -> None:
        facts = pd.DataFrame([
            {
                "node_id": node_id, "sample_day": "2026-09-01", "business": "A",
                "business_name": "业务A", "daily_isp": "移动",
                "daily_province": "浙江", "network_schedule_type": "异网出省",
                "schedule_target_isp": target_isp, "buildBandwidth": 100,
                "capacity_observed": True, "capacity_peak95_mbps": 50,
                "node_utilization": 0.5, "capacity_peak95_source": "peak95",
                "miner_income": 10, "platform_revenue": 20, "platform_profit": 10,
            }
            for node_id, target_isp in [("n1", "电信"), ("n2", "联通")]
        ])

        pools = v6.aggregate_capacity_pools(facts)
        detailed = pools[
            pools["pool_level"].eq("business_province_isp_schedule")
        ]

        self.assertEqual(len(detailed), 2)
        self.assertEqual(
            set(detailed["pool_key"]),
            {"A|浙江|移动|异网出省|电信", "A|浙江|移动|异网出省|联通"},
        )

    def test_capacity_ceiling_uses_recent_p75_and_protects_current_load(self) -> None:
        daily = pd.DataFrame([
            {
                "sample_day": f"2026-09-{day:02d}", "business": "A", "business_name": "业务A",
                "node_count": 10, "active_build_bandwidth_mbps": 1000,
                "estimated_total_peak95_mbps": 600, "observed_utilization": 0.6,
                "bandwidth_observation_coverage": 0.95, "low_utilization_node_share": 0.2,
                "platform_profit_per_build_mbps": 0.02,
            }
            for day in range(1, 8)
        ])

        summary, correlations = v6.build_business_capacity_summary(daily, 0.7, 14, 7, 0.8)
        row = summary.iloc[0]

        self.assertTrue(bool(row["capacity_evidence_sufficient"]))
        self.assertAlmostEqual(row["raw_capacity_ceiling_mbps"], 600 / 0.7)
        self.assertEqual(row["operational_capacity_ceiling_mbps"], 1000)
        self.assertGreater(row["oversupplied_bandwidth_mbps"], 0)
        self.assertEqual(len(correlations), 1)

    def test_global_optimizer_keeps_only_one_switch_when_destination_is_full(self) -> None:
        recommendations = pd.DataFrame([
            self.recommendation("n1", 0.4),
            self.recommendation("n2", 0.3),
        ])
        capacity = pd.DataFrame([
            {
                "business": "A", "business_name": "业务A",
                "current_active_build_bandwidth_mbps": 120,
                "operational_capacity_ceiling_mbps": 120,
                "capacity_evidence_sufficient": True,
            },
            {
                "business": "B", "business_name": "业务B",
                "current_active_build_bandwidth_mbps": 0,
                "operational_capacity_ceiling_mbps": 60,
                "capacity_evidence_sufficient": True,
            },
        ])

        allocations, diagnostics = v6.optimize_allocations(recommendations, capacity, 0.03)

        self.assertEqual(int(allocations["planned_change"].sum()), 1)
        changed = allocations[allocations["planned_change"]].iloc[0]
        self.assertEqual(changed["node_id"], "n1")
        self.assertEqual(changed["planned_business"], "B")
        self.assertEqual(diagnostics["eligible_nodes"], 2)

    def test_capacity_fallback_uses_top2_when_top1_is_full(self) -> None:
        first = self.recommendation("n1", 0.4)
        second = self.recommendation("n2", 0.3)
        second.update({
            "business_top2": "C", "business_name_top2": "业务C",
            "combined_score_top2": 0.7, "confidence_top2": "medium",
            "platform_unit_profit_top2": 0.015,
        })
        recommendations = pd.DataFrame([first, second])
        capacity = pd.DataFrame([
            {
                "business": "A", "business_name": "业务A",
                "current_active_build_bandwidth_mbps": 120,
                "operational_capacity_ceiling_mbps": 120,
                "capacity_evidence_sufficient": True,
            },
            {
                "business": "B", "business_name": "业务B",
                "current_active_build_bandwidth_mbps": 0,
                "operational_capacity_ceiling_mbps": 60,
                "capacity_evidence_sufficient": True,
            },
            {
                "business": "C", "business_name": "业务C",
                "current_active_build_bandwidth_mbps": 0,
                "operational_capacity_ceiling_mbps": 60,
                "capacity_evidence_sufficient": True,
            },
        ])

        allocations, _ = v6.optimize_allocations(recommendations, capacity, 0.03)

        self.assertEqual(int(allocations["planned_change"].sum()), 2)
        fallback = allocations[allocations["node_id"].eq("n2")].iloc[0]
        self.assertEqual(fallback["planned_business"], "C")
        self.assertEqual(fallback["capacity_aware_rank"], 2)

    def test_exact_region_pool_limit_can_force_top2(self) -> None:
        first = self.recommendation("n1", 0.4)
        second = self.recommendation("n2", 0.3)
        second.update({
            "business_top2": "C", "business_name_top2": "业务C",
            "combined_score_top2": 0.7, "confidence_top2": "medium",
            "platform_unit_profit_top2": 0.015,
        })
        capacity = pd.DataFrame([
            *self.capacity_pools("A", "业务A", 120, 120),
            *self.capacity_pools("B", "业务B", 0, 120, exact_cap=60),
            *self.capacity_pools("C", "业务C", 0, 60),
        ])

        allocations, diagnostics = v6.optimize_allocations(
            pd.DataFrame([first, second]), capacity, 0.03,
        )

        self.assertEqual(int(allocations["planned_change"].sum()), 2)
        self.assertEqual(
            allocations.set_index("node_id").loc["n1", "planned_business"], "B",
        )
        fallback = allocations.set_index("node_id").loc["n2"]
        self.assertEqual(fallback["planned_business"], "C")
        self.assertEqual(fallback["capacity_aware_rank"], 2)
        self.assertEqual(
            fallback["capacity_most_specific_level"],
            "business_province_isp_schedule",
        )
        self.assertEqual(diagnostics["capacity_constraints"], 12)

    def test_missing_exact_pool_falls_back_to_schedule_pool(self) -> None:
        capacity = pd.DataFrame([
            *self.capacity_pools("A", "业务A", 60, 60),
            *self.capacity_pools(
                "B", "业务B", 0, 60,
                exact_evidence=False,
            ),
        ])

        allocations, _ = v6.optimize_allocations(
            pd.DataFrame([self.recommendation("n1", 0.4)]), capacity, 0.03,
        )

        row = allocations.iloc[0]
        self.assertTrue(bool(row["planned_change"]))
        self.assertEqual(row["planned_business"], "B")
        self.assertEqual(
            row["capacity_most_specific_level"], "business_isp_schedule",
        )
        self.assertEqual(
            row["capacity_constraint_levels"],
            "business > business_isp > business_isp_schedule",
        )

    def test_greedy_fallback_also_respects_exact_region_pool(self) -> None:
        first = self.recommendation("n1", 0.4)
        second = self.recommendation("n2", 0.3)
        second.update({
            "business_top2": "C", "business_name_top2": "业务C",
            "combined_score_top2": 0.7, "confidence_top2": "medium",
            "platform_unit_profit_top2": 0.015,
        })
        capacity = pd.DataFrame([
            *self.capacity_pools("A", "业务A", 120, 120),
            *self.capacity_pools("B", "业务B", 0, 120, exact_cap=60),
            *self.capacity_pools("C", "业务C", 0, 60),
        ])

        with mock.patch.object(v6.importlib.util, "find_spec", return_value=None):
            allocations, diagnostics = v6.optimize_allocations(
                pd.DataFrame([first, second]), capacity, 0.03,
            )

        self.assertEqual(diagnostics["solver"], "deterministic_capacity_greedy")
        self.assertEqual(
            allocations.set_index("node_id").loc["n1", "planned_business"], "B",
        )
        self.assertEqual(
            allocations.set_index("node_id").loc["n2", "planned_business"], "C",
        )

    def test_current_load_is_released_from_observed_snapshot_pool(self) -> None:
        recommendation = self.recommendation("n1", 0.4)
        recommendation.update({
            "network_schedule_type": "本网出省",
            "capacity_current_province": "浙江",
            "capacity_current_isp": "移动",
            "capacity_current_network_schedule_type": "本网本省",
        })
        business_b = self.capacity_pools("B", "业务B", 0, 60)
        for row in business_b:
            row["pool_key"] = str(row["pool_key"]).replace("本网本省", "本网出省")
        capacity = pd.DataFrame([
            *self.capacity_pools("A", "业务A", 60, 60),
            *business_b,
        ])
        recommendations = pd.DataFrame([recommendation])

        allocations, _ = v6.optimize_allocations(recommendations, capacity, 0.03)
        pool_summary = v6.allocation_pool_summary(
            recommendations, allocations, capacity,
        ).set_index("pool_constraint_id")

        self.assertTrue(bool(allocations.iloc[0]["planned_change"]))
        self.assertEqual(
            pool_summary.loc[
                "business_isp_schedule::A|移动|本网本省|移动", "planned_load_mbps"
            ],
            0,
        )
        self.assertEqual(
            pool_summary.loc[
                "business_isp_schedule::B|移动|本网出省|移动", "planned_load_mbps"
            ],
            60,
        )

    def test_node_report_separates_review_and_capacity_blocked_nodes(self) -> None:
        recommendations = pd.DataFrame([
            self.recommendation("n1", 0.4),
            self.recommendation("n2", 0.3),
        ])
        capacity = pd.DataFrame([
            *self.capacity_pools("A", "业务A", 120, 120),
            *self.capacity_pools("B", "业务B", 0, 120, exact_cap=60),
        ])
        allocations, _ = v6.optimize_allocations(recommendations, capacity, 0.03)
        pool_summary = v6.allocation_pool_summary(
            recommendations, allocations, capacity,
        )

        report = v6.build_node_capacity_report(allocations, pool_summary)

        self.assertEqual(
            report["capacity_report_category"].value_counts().to_dict(),
            {"建议切换-Top1": 1, "容量约束保留": 1},
        )
        blocked = report[report["capacity_report_category"].eq("容量约束保留")].iloc[0]
        self.assertGreater(blocked["capacity_blocking_pool_count"], 0)
        self.assertEqual(
            blocked["capacity_bottleneck_level"],
            "business_province_isp_schedule",
        )
        self.assertIn("小于节点", blocked["capacity_diagnostic"])

    @staticmethod
    def capacity_pools(
        business: str,
        name: str,
        current: float,
        cap: float,
        exact_cap: float | None = None,
        include_exact: bool = True,
        exact_evidence: bool = True,
    ) -> list[dict[str, object]]:
        definitions = [
            ("business", business),
            ("business_isp", f"{business}|移动"),
            ("business_isp_schedule", f"{business}|移动|本网本省|移动"),
        ]
        if include_exact:
            definitions.append((
                "business_province_isp_schedule",
                f"{business}|浙江|移动|本网本省|移动",
            ))
        return [{
            "pool_level": level,
            "pool_key": key,
            "business": business,
            "business_name": name,
            "current_active_build_bandwidth_mbps": current,
            "operational_capacity_ceiling_mbps": (
                exact_cap
                if level == "business_province_isp_schedule" and exact_cap is not None
                else cap
            ),
            "capacity_evidence_sufficient": (
                exact_evidence
                if level == "business_province_isp_schedule"
                else True
            ),
        } for level, key in definitions]

    @staticmethod
    def recommendation(node_id: str, gain: float) -> dict[str, object]:
        return {
            "node_id": node_id, "province": "浙江", "isp": "移动",
            "network_schedule_type": "本网本省", "schedule_target_isp": "移动",
            "build_bandwidth_mbps": 60,
            "current_business": "A", "current_business_name": "业务A",
            "current_business_status": "current_not_top1",
            "current_predicted_combined_score": 0.5,
            "recommendation_action": "谨慎人工复核",
            "business_top1": "B", "business_name_top1": "业务B",
            "combined_score_top1": 0.5 + gain, "confidence_top1": "medium",
            "platform_unit_profit_top1": 0.02,
            "business_top2": "A", "business_name_top2": "业务A",
            "combined_score_top2": 0.5, "confidence_top2": "medium",
            "platform_unit_profit_top2": 0.01,
            "business_top3": "", "business_name_top3": "",
            "combined_score_top3": None, "confidence_top3": "low",
            "platform_unit_profit_top3": None,
        }


if __name__ == "__main__":
    unittest.main()
