#!/usr/bin/env python3
from __future__ import annotations

import unittest

import pandas as pd

import build_v3_daily_business_training as v3


class V3DailyBusinessTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.allowlist = pd.DataFrame([
            {"business": "10000278", "business_name": "七牛虚拟A", "brand": "七牛"},
            {"business": "10000280", "business_name": "七牛CDN-ZJ月95", "brand": "七牛"},
            {"business": "10000069", "business_name": "腾讯汇聚P2P", "brand": "腾讯"},
            {"business": "10000064", "business_name": "百度小度", "brand": "百度"},
        ])

    def test_qiniu_virtual_rows_and_bound_revenue_become_one_fact(self) -> None:
        raw = pd.DataFrame([
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": "10000278",
                "state": "online", "stage": "inService", "buildBandwidth": 500,
                "cost_finalAmount": 10, "revenue_finalAmount": 2,
                "vendorSuggestCustomersName": '["10000069:腾讯汇聚P2P"]',
                "scheduleISPs": '["联通"]', "transProvRate": 100,
            },
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": "10000280",
                "state": "online", "stage": "inService", "buildBandwidth": 500,
                "cost_finalAmount": 5, "revenue_finalAmount": 3,
                "vendorSuggestCustomersName": '["10000069:腾讯汇聚P2P"]',
                "scheduleISPs": '["联通"]', "transProvRate": 100,
            },
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": "10000069",
                "state": None, "stage": None, "buildBandwidth": 500,
                "cost_finalAmount": 0, "revenue_finalAmount": 30,
            },
        ])
        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {})
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts.iloc[0]["business"], v3.QINIU_BUSINESS_ID)
        self.assertEqual(facts.iloc[0]["business_name"], v3.QINIU_BUSINESS_NAME)
        self.assertEqual(facts.iloc[0]["cum_cost_7d"], 15)
        self.assertEqual(facts.iloc[0]["cum_revenue_7d"], 35)
        self.assertEqual(facts.iloc[0]["daily_scheduleisps"], "联通")
        self.assertEqual(audit.iloc[0]["status"], "clean")

    def test_multiple_active_mainstream_businesses_are_excluded(self) -> None:
        raw = pd.DataFrame([
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": business,
                "state": "online", "stage": "inService", "buildBandwidth": 500,
                "cost_finalAmount": 10, "revenue_finalAmount": 20,
            }
            for business in ["10000064", "10000069"]
        ])
        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {})
        self.assertTrue(facts.empty)
        self.assertEqual(audit.iloc[0]["status"], "excluded_multiple_active_mainstream_businesses")

    def test_unattributed_financial_business_excludes_whole_day(self) -> None:
        raw = pd.DataFrame([
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": "10000064",
                "state": "online", "stage": "inService", "buildBandwidth": 500,
                "cost_finalAmount": 10, "revenue_finalAmount": 20,
            },
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": "10000069",
                "state": None, "stage": None, "buildBandwidth": 500,
                "cost_finalAmount": 0, "revenue_finalAmount": 1,
            },
        ])
        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {})
        self.assertTrue(facts.empty)
        self.assertEqual(audit.iloc[0]["status"], "excluded_unattributed_financial_rows")

    def test_training_nodes_use_model_profile_field_names(self) -> None:
        facts = pd.DataFrame([{
            "node_id": "n1",
            "sample_day": "2026-09-01",
            "daily_province": "浙江",
            "daily_city": "杭州",
            "daily_isp": "电信",
            "daily_nattype": "full",
            "daily_resourcetype": "aggregation",
            "daily_deliverytype": "aggregation",
            "daily_scheduleisps": "联通",
            "daily_transprovrate": 100,
            "buildBandwidth": 500,
        }])
        nodes = v3.build_training_nodes(facts)
        self.assertEqual(nodes.iloc[0]["resourcetype"], "aggregation")
        self.assertEqual(nodes.iloc[0]["nattype"], "full")
        self.assertEqual(nodes.iloc[0]["deliverytype"], "aggregation")
        self.assertEqual(nodes.iloc[0]["bw"], 500)

    def test_empty_schedule_and_transprov_mean_local_network_and_province(self) -> None:
        raw = pd.DataFrame([{
            "nodeId": "n1", "day": "2026-09-01", "customerId": "10000064",
            "state": "online", "stage": "inService", "buildBandwidth": 500,
            "isp": "移动", "scheduleISPs": "", "transProvRate": None,
            "cost_finalAmount": 10, "revenue_finalAmount": 20,
        }])
        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {})
        self.assertEqual(audit.iloc[0]["status"], "clean")
        self.assertEqual(facts.iloc[0]["daily_scheduleisps"], "移动")
        self.assertEqual(facts.iloc[0]["daily_transprovrate"], 0)

    def test_multiple_schedule_isps_keep_only_the_first_carrier(self) -> None:
        raw = pd.DataFrame([{
            "nodeId": "n1", "day": "2026-09-01", "customerId": "10000064",
            "state": "online", "stage": "inService", "buildBandwidth": 500,
            "isp": "移动", "scheduleISPs": '["联通", "电信"]',
            "transProvRate": 100,
            "cost_finalAmount": 10, "revenue_finalAmount": 20,
        }])

        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {})

        self.assertEqual(audit.iloc[0]["status"], "clean")
        self.assertEqual(facts.iloc[0]["daily_scheduleisps"], "联通")

    def test_intermediate_transprov_rate_is_excluded(self) -> None:
        raw = pd.DataFrame([{
            "nodeId": "n1", "day": "2026-09-01", "customerId": "10000064",
            "state": "online", "stage": "inService", "buildBandwidth": 500,
            "isp": "移动", "scheduleISPs": "电信", "transProvRate": 50,
            "cost_finalAmount": 10, "revenue_finalAmount": 20,
        }])
        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {})
        self.assertTrue(facts.empty)
        self.assertEqual(audit.iloc[0]["status"], "excluded_invalid_transprov_rate")

    def test_virtual_business_is_resolved_by_unique_real_financial_row(self) -> None:
        raw = pd.DataFrame([
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": "90000001",
                "state": "online", "stage": "inService", "buildBandwidth": 500,
                "cost_finalAmount": 10, "revenue_finalAmount": 0,
            },
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": "10000064",
                "state": "", "stage": "", "buildBandwidth": 500,
                "cost_finalAmount": 0, "revenue_finalAmount": 20,
            },
        ])
        bindings = {"90000001": {"10000064", "10000069"}}

        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {}, bindings)

        self.assertEqual(audit.iloc[0]["status"], "clean")
        self.assertEqual(facts.iloc[0]["business"], "10000064")
        self.assertEqual(facts.iloc[0]["cum_cost_7d"], 10)
        self.assertEqual(facts.iloc[0]["cum_revenue_7d"], 20)

    def test_ambiguous_virtual_business_excludes_whole_day(self) -> None:
        raw = pd.DataFrame([{
            "nodeId": "n1", "day": "2026-09-01", "customerId": "90000001",
            "state": "online", "stage": "inService", "buildBandwidth": 500,
            "cost_finalAmount": 10, "revenue_finalAmount": 0,
        }] + [
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": business,
                "state": "", "stage": "", "buildBandwidth": 500,
                "cost_finalAmount": 0, "revenue_finalAmount": 20,
            }
            for business in ["10000064", "10000069"]
        ])
        bindings = {"90000001": {"10000064", "10000069"}}

        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {}, bindings)

        self.assertTrue(facts.empty)
        self.assertEqual(
            audit.iloc[0]["status"],
            "excluded_ambiguous_virtual_business_binding",
        )

    def test_consecutive_business_days_share_one_unit_of_weight(self) -> None:
        facts = pd.DataFrame([
            {"node_id": "n1", "business": "A", "sample_day": "2026-09-01"},
            {"node_id": "n1", "business": "A", "sample_day": "2026-09-02"},
            {"node_id": "n1", "business": "A", "sample_day": "2026-09-03"},
            {"node_id": "n1", "business": "A", "sample_day": "2026-09-05"},
            {"node_id": "n1", "business": "B", "sample_day": "2026-09-06"},
            {"node_id": "n1", "business": "B", "sample_day": "2026-09-07"},
        ])

        weighted = v3.add_consecutive_sample_weights(facts)

        self.assertEqual(weighted["consecutive_valid_days"].tolist(), [3, 3, 3, 1, 2, 2])
        self.assertEqual(weighted["sample_weight"].round(6).tolist(), [
            0.333333, 0.333333, 0.333333, 1.0, 0.5, 0.5,
        ])
        run_weights = weighted.groupby("consecutive_business_run")["sample_weight"].sum()
        self.assertTrue((run_weights.round(6) == 1.0).all())

    def test_qiniu_duplicate_ratio_is_reconstructed_once_not_summed(self) -> None:
        raw = pd.DataFrame([
            {
                "nodeId": "n1", "day": "2026-09-01", "customerId": business,
                "state": "online", "stage": "inService", "buildBandwidth": 500,
                "cost_finalAmount": 5, "revenue_finalAmount": 10,
                "peak95Ratio": 70,
            }
            for business in ["10000278", "10000280"]
        ])

        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {})

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts.iloc[0]["capacity_peak95_mbps"], 350)
        self.assertEqual(facts.iloc[0]["capacity_peak95_source"], "peak95Ratio_reconstructed")
        self.assertEqual(facts.iloc[0]["capacity_peak95_observation_count"], 2)
        self.assertFalse(bool(audit.iloc[0]["capacity_peak95_conflict"]))

    def test_direct_peak_has_priority_over_analyze_and_ratio(self) -> None:
        raw = pd.DataFrame([{
            "nodeId": "n1", "day": "2026-09-01", "customerId": "10000064",
            "state": "online", "stage": "inService", "buildBandwidth": 500,
            "cost_finalAmount": 10, "revenue_finalAmount": 20,
            "peak95": 320_000_000, "analyzePeak95": 310_000_000, "peak95Ratio": 60,
        }])

        facts, _, _ = v3.build_daily_facts(raw, self.allowlist, {})

        self.assertEqual(facts.iloc[0]["capacity_peak95_mbps"], 320)
        self.assertEqual(facts.iloc[0]["capacity_peak95_source"], "peak95")

    def test_outlier_direct_peak_falls_back_to_ratio(self) -> None:
        raw = pd.DataFrame([{
            "nodeId": "n1", "day": "2026-09-01", "customerId": "10000064",
            "state": "online", "stage": "inService", "buildBandwidth": 500,
            "cost_finalAmount": 10, "revenue_finalAmount": 20,
            "peak95": 2_000_000_000, "peak95Ratio": 60,
        }])

        facts, audit, _ = v3.build_daily_facts(raw, self.allowlist, {})

        self.assertEqual(facts.iloc[0]["capacity_peak95_mbps"], 300)
        self.assertEqual(facts.iloc[0]["capacity_peak95_source"], "peak95Ratio_reconstructed")
        self.assertEqual(audit.iloc[0]["capacity_peak95_outlier_rows"], 1)


if __name__ == "__main__":
    unittest.main()
