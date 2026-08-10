#!/usr/bin/env python3
"""重建跨日期多业务节点的最佳业务推荐模型。

口径：
1. 宽表历史上至少出现 3 个业务，且这些业务的首次出现日不少于 3 个不同日期；
2. 至少有一条历史记录 state=online，作为“已上线”条件；
3. 每个节点-业务以该业务首次出现日为起点，聚合后续 7 个自然日；
4. 节点级 80/20 切分，健康约束只使用训练集建立的业务历史可行域。

该脚本只执行 SELECT，凭据由 SupersetSQLClient 从 ~/.codex/.superset_env 读取。
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient


BASE = "http://superset.yzh-logverse.k8s.qiniu.io"
VM_BASE = "https://vm-select.mvm.qiniu.io/select/293:0/prometheus/api/v1"
PROM_FEATURE_CHUNK_SIZE = 500
HERE = Path(__file__).resolve().parent

START_DAY = dt.date(2026, 3, 16)
END_DAY = dt.date(2026, 8, 4)
ATTRIBUTE_DAY = "20260805"
ATTRIBUTE_LOOKBACK_DAYS = int(os.environ.get("MULTIBUSINESS_ATTRIBUTE_LOOKBACK_DAYS", "1"))
SUFFIXES = tuple(f"{value:02x}" for value in range(13))
NODE_BUCKETS = tuple("0123456789abcdefy")
NODE_SUB_BUCKETS = tuple("0123456789abcdef")
SAMPLE_RATE = len(SUFFIXES) / 256
MIN_BUSINESS_COUNT = 3
MIN_BUSINESS_SUPPORT = int(os.environ.get("MULTIBUSINESS_MIN_SUPPORT", "30"))
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
QUERY_WORKERS = int(os.environ.get("MULTIBUSINESS_QUERY_WORKERS", "3"))
ATTRIBUTE_BATCH_SIZE = 1200
OUTCOME_QUERY_LIMIT = 3_000_000

OUTPUT_PAIRS = HERE / "multibusiness_eligible_pairs.csv"
OUTPUT_OUTCOMES = HERE / "multibusiness_outcomes.csv"
OUTPUT_NODES = HERE / "multibusiness_nodes.csv"
OUTPUT_PROM = HERE / "multibusiness_prometheus_probe.json"
OUTPUT_RECOMMENDATIONS = HERE / "multibusiness_recommendations.csv"
OUTPUT_BASELINE_RECOMMENDATIONS = HERE / "multibusiness_recommendations_pre_day_core_baseline.csv"
OUTPUT_FEATURE_COMPARISON = HERE / "multibusiness_feature_comparison_pre_day.json"
OUTPUT_MULTISEED = HERE / "multibusiness_multiseed_validation_pre_day.json"
OUTPUT_METRICS = HERE / "multibusiness_metrics.json"
OUTPUT_REPORT = HERE / "多业务节点最佳业务重建报告.md"
# 该分区在本次重建中已由 Superset 返回 HIVE_CURSOR_ERROR，并在属性拉取时排除。
# 缓存复用时仍保留这条数据质量记录，避免报告因不重复拉取而丢失。
UNREADABLE_NODE_ANALYSIS_PARTITIONS: set[tuple[str, str]] = {("20260627", "05")}

CAT_FEATURES = [
    "vendorid", "deliverytype", "resourcetype", "dialtype", "nattype",
    "scheduleisps", "regsource", "customermode", "province", "isp", "city",
    "device_type", "arch_type", "isvm", "qoskiller_status", "os", "arch",
    "node_state", "node_stage", "node_status", "hardwaretype",
    "node_manufacturer", "node_model",
    "node_analysis_stage", "idc_id", "idc_name", "analysis_nodecustmertype",
    "analysis_nodedeliverytype", "analysis_tcpnattype", "analysis_udpnattype",
    "analysis_cgroupversion", "analysis_lastdeploystate", "analysis_lastdeployscenario",
    "analysis_cooperationtype", "analysis_supply_side_delivery_type", "analysis_isroot",
    "analysis_issupportipv6", "analysis_upnpstate", "dial_is_normal", "dial_ipv6_enable",
    "dial_on_physical_nic", "dial_lbtype", "dial_lbtype_v6", "join_ismanaged",
    "join_isminorisp", "join_isbantransprov", "join_isipv6schedule",
    "join_natforwardenable", "join_idcbindtype", "join_deploystate",
]
NUM_FEATURES = [
    "bw", "bandwidth", "actualbandwidth", "netbenchlimitbandwidth",
    "cpuload1", "cpuload5", "cpuload15", "corenum", "cpuuser", "cpusystem",
    "cpunice", "cpuidle", "cpuiowait", "cpuirq", "cpusoftirq", "cpusteal",
    "memtotal", "memused", "memfree", "membufferd", "memcached",
    "join_cpu_load1", "join_cpu_load5", "join_cpu_load15", "join_cpu_corenumber",
    "join_cpu_totalcores", "join_cpu_totalphysicals", "join_cpu_totalthreads",
    "join_cpu_idle", "join_memtotal", "join_memused", "join_memfree",
    "join_membuffers", "join_memcached", "mem_used_ratio", "retrans",
    "v4pingloss", "v6pingloss",
    "v4pingavgrtt", "v4pingbestrtt", "v4pingworstrtt", "v6pingavgrtt",
    "v6pingbestrtt", "v6pingworstrtt", "totaldisksize", "totaldiskused",
    "hdddisksize", "hdddiskused", "ssddisksize", "ssddiskused",
    "systemdisksize", "systemdiskused", "disk_used_ratio", "hdd_used_ratio",
    "ssd_used_ratio", "system_disk_used_ratio", "maxioutil", "quality_cpuutil",
    "quality_retransrate", "quality_pinglossrate", "quality_rtt",
    "sevendayavg95ratio", "join_disks_totalsize", "join_disks_total_size",
    "join_disks_total_used", "join_disks_total_usedrate", "join_disks_hdd_size",
    "join_disks_hdd_used", "join_disks_hdd_usedrate", "join_disks_ssd_size",
    "join_disks_ssd_used", "join_disks_ssd_usedrate", "join_disks_system_size",
    "join_disks_system_used", "join_disks_system_usedrate", "join_net_pingrtt",
    "join_net_tcpretransrate", "join_net_pinglossrate", "join_actualbw",
    "join_limitbw", "join_actualbw_achieverate", "join_actualbw_rtt",
    "join_actualbw_tcpretr_rate", "join_limitbw_achieverate",
    "join_netbench_bandwidth", "join_netcardbw", "join_procbw", "join_diskdivbw",
    "cpu_load1_per_core", "smart_available_spare", "smart_bad_block_count",
    "smart_case_temperature", "smart_critical_warning", "zfs_pool_online",
    "zfs_dataset_read", "zfs_dataset_write", "prom_retrans_ratio",
    "analysis_transprovrate", "analysis_usb_bw", "analysis_bw_num",
    "analysis_serviceduration", "analysis_yesterday_p95_bw",
    "analysis_yesterday_online_duration", "analysis_yesterday_evening_online_duration",
    "analysis_yesterday_outline_count", "analysis_yesterday_evening_outline_count",
    "analysis_yesterday_snapshot_bw", "analysis_yesterday_snapshot_netbench_bw",
    "analysis_dby_online_duration", "analysis_dby_evening_online_duration",
    "analysis_dby_outline_count", "analysis_dby_evening_outline_count",
    "analysis_dby_p95_bw", "analysis_dby_snapshot_bw", "analysis_dby_snapshot_netbench_bw",
    "analysis_dby2_p95_bw", "analysis_dby2_online_duration",
    "analysis_dby2_outline_count", "analysis_dby2_evening_outline_count",
    "analysis_dby2_snapshot_netbench_bw", "analysis_out_bandwidth_total",
    "analysis_out_bandwidth_node", "analysis_out_limit_bandwidth_node",
    "analysis_out_bandwidth_udp_node", "analysis_out_limit_bandwidth_udp_node",
    "analysis_out_bandwidth_udp_total", "analysis_udp_actual_bandwidth",
    "analysis_udp_netbench_limit_bandwidth", "analysis_conn_succeed_bw_num",
    "analysis_abnormal_line_num", "analysis_ping_loss_stats_total",
    "analysis_ping_loss_p5_abnormal", "analysis_ping_loss_p10_abnormal",
    "analysis_retrans_stats_total", "analysis_retrans_p5_abnormal",
    "analysis_cpuutil_stats_total", "analysis_cpuutil_p70_abnormal",
    "analysis_ipv4_packet_count", "analysis_ipv6_packet_count", "analysis_disk_div_bw",
    "join_uptime", "join_duration_time", "join_has_errors", "join_iptable_abnormal",
    "join_biz_bw", "join_yesterday_avg_peak_ratio", "join_quality_retrans_abnormal",
    "join_quality_retrans_total", "join_quality_pingloss_p5_abnormal",
    "join_quality_pingloss_p5_total", "join_quality_pingloss_p10_abnormal",
    "join_quality_pingloss_p10_total", "join_quality_cpuutil_p70_abnormal",
    "join_quality_cpuutil_p70_total", "dial_account_count", "yesterday_online_ratio",
    "dby_online_ratio", "analysis_retrans_abnormal_ratio",
    "analysis_pingloss_abnormal_ratio", "analysis_cpuutil_abnormal_ratio",
    "analysis_actual_bw_ratio", "analysis_udp_actual_bw_ratio",
    "analysis_out_limit_bw_ratio",
]

# 预测模型只允许使用上线前即可确定的节点固有属性：节点配置、地域、设备信息、
# 硬件容量和名义网络配置。运行后产生的状态、利用率、质量、在线时长和业务流量
# 仍保留在属性快照中，但只能用于健康准入，不能进入收益预测模型。
NODE_INTRINSIC_CAT_FEATURES = [
    "vendorid", "deliverytype", "resourcetype", "dialtype", "nattype",
    "scheduleisps", "regsource", "customermode", "province", "isp", "city",
    "device_type", "arch_type", "isvm", "qoskiller_status", "os", "arch",
    "hardwaretype", "node_manufacturer", "node_model", "idc_id", "idc_name",
    "analysis_nodecustmertype", "analysis_nodedeliverytype",
    "analysis_tcpnattype", "analysis_udpnattype", "analysis_cgroupversion",
    "analysis_lastdeploystate", "analysis_lastdeployscenario",
    "analysis_cooperationtype", "analysis_supply_side_delivery_type",
    "analysis_isroot", "analysis_issupportipv6", "analysis_upnpstate",
    "dial_ipv6_enable", "dial_on_physical_nic", "dial_lbtype", "dial_lbtype_v6",
    "join_ismanaged", "join_isminorisp", "join_isbantransprov",
    "join_isipv6schedule", "join_natforwardenable", "join_idcbindtype",
    "join_deploystate",
]
NODE_INTRINSIC_NUM_FEATURES = [
    # node_analysis_data 中的容量字段与 node_join 中的硬件容量字段均保留，
    # 它们描述节点能力，不代表上线后的实际使用量。
    "bw", "corenum", "memtotal", "totaldisksize", "hdddisksize",
    "ssddisksize", "systemdisksize", "join_cpu_corenumber",
    "join_cpu_totalcores", "join_cpu_totalphysicals", "join_cpu_totalthreads",
    "join_memtotal", "join_disks_totalsize", "join_disks_total_size",
    "join_disks_hdd_size", "join_disks_ssd_size", "join_disks_system_size",
]
CORE_NODE_INTRINSIC_CAT_FEATURES = [
    "vendorid", "deliverytype", "resourcetype", "dialtype", "nattype",
    "scheduleisps", "regsource", "customermode", "province", "isp", "city",
    "device_type", "arch_type", "isvm", "qoskiller_status", "os", "arch",
    "hardwaretype", "node_manufacturer", "node_model",
]
CORE_NODE_INTRINSIC_NUM_FEATURES = list(NODE_INTRINSIC_NUM_FEATURES)
HEALTH_METRICS = [
    "cpu_load1_per_core", "mem_used_ratio", "disk_used_ratio", "retrans",
    "v4pingloss", "v6pingloss", "quality_retransrate", "quality_pinglossrate",
    "maxioutil", "smart_available_spare", "smart_bad_block_count",
    "smart_case_temperature", "smart_critical_warning", "zfs_pool_online",
    "zfs_dataset_read", "zfs_dataset_write",
    "prom_retrans_ratio", "dial_is_normal", "analysis_retrans_abnormal_ratio",
]

# 这些是可审计的初始硬阈值，不是已被因果验证的业务准入标准。
# 业务特定上下界会从训练集的历史已运行记录按 1%/99% 分位数建立。
GLOBAL_HEALTH_RULES = {
    "cpu_load1_per_core_max": 1.5,
    "mem_used_ratio_max": 0.95,
    "disk_used_ratio_max": 0.95,
    "retrans_max": 5.0,
    "v4pingloss_max": 5.0,
    "v6pingloss_max": 5.0,
    "quality_retransrate_max": 5.0,
    "quality_pinglossrate_max": 5.0,
    "maxioutil_max": 95.0,
    "smart_available_spare_min": 10.0,
    "smart_bad_block_count_max": 0.0,
    "smart_critical_warning_max": 0.0,
    "zfs_pool_online_required": 1,
    "prom_retrans_ratio_max": 5.0,
    "dial_is_normal_required": 1,
    "analysis_retrans_abnormal_ratio_max": 0.5,
}

INTERACTION_FEATURES = [
    "bw", "bandwidth", "actualbandwidth", "netbenchlimitbandwidth",
    "cpuload1", "cpuload5", "cpuload15", "corenum", "cpuidle",
    "join_cpu_load1", "join_cpu_load5", "join_cpu_load15", "join_cpu_corenumber",
    "join_cpu_totalcores", "join_cpu_totalthreads", "join_cpu_idle",
    "memtotal", "memused", "memfree", "membufferd", "memcached",
    "join_memtotal", "join_memused", "join_memfree", "join_membuffers",
    "join_memcached", "mem_used_ratio", "retrans", "v4pingloss", "v6pingloss",
    "v4pingavgrtt", "v4pingworstrtt", "v6pingavgrtt", "v6pingworstrtt",
    "totaldisksize", "totaldiskused", "hdddisksize", "hdddiskused",
    "ssddisksize", "ssddiskused", "systemdisksize", "systemdiskused",
    "disk_used_ratio", "hdd_used_ratio", "ssd_used_ratio",
    "system_disk_used_ratio", "maxioutil", "quality_cpuutil",
    "quality_retransrate", "quality_pinglossrate", "quality_rtt",
    "sevendayavg95ratio", "join_disks_total_size", "join_disks_total_used",
    "join_disks_total_usedrate", "join_disks_hdd_usedrate", "join_disks_ssd_usedrate",
    "join_disks_system_usedrate", "join_net_pingrtt", "join_net_tcpretransrate",
    "join_net_pinglossrate", "join_actualbw", "join_limitbw",
    "join_actualbw_achieverate", "join_actualbw_rtt", "join_actualbw_tcpretr_rate",
    "join_limitbw_achieverate", "join_netbench_bandwidth", "join_netcardbw",
    "join_procbw", "join_diskdivbw", "cpu_load1_per_core", "smart_available_spare",
    "smart_bad_block_count", "smart_case_temperature", "smart_critical_warning",
    "zfs_pool_online", "zfs_dataset_read", "zfs_dataset_write", "prom_retrans_ratio",
]

# 这些指标只能取到上线后的运行时快照或历史观测值，不是节点固有属性。
# 它们保留在节点属性和健康准入检查中，但不进入历史收益模型，避免未来状态泄漏到目标。
VM_RUNTIME_ONLY_FEATURES = {
    "smart_available_spare", "smart_bad_block_count", "smart_case_temperature",
    "smart_critical_warning", "zfs_pool_online", "zfs_dataset_read",
    "zfs_dataset_write", "prom_retrans_ratio",
}
# 当前没有业务方或历史样本支持的可审计硬阈值；有值也只能人工审核，不能自动放行。
HEALTH_MANUAL_ONLY_METRICS = {
    "smart_case_temperature", "zfs_dataset_read", "zfs_dataset_write",
}
NODE_INTRINSIC_INTERACTION_FEATURES = [
    column for column in INTERACTION_FEATURES if column in NODE_INTRINSIC_NUM_FEATURES
]
MODEL_NUM_FEATURES = list(NODE_INTRINSIC_NUM_FEATURES)
MODEL_INTERACTION_FEATURES = list(NODE_INTRINSIC_INTERACTION_FEATURES)

# 核心基线沿用扩展前已经使用的字段集合，确保特征增益比较只改变字段，
# 不改变节点筛选、目标窗口、候选业务和节点级切分口径。
CORE_CAT_FEATURES = list(CORE_NODE_INTRINSIC_CAT_FEATURES)
CORE_NUM_FEATURES = list(CORE_NODE_INTRINSIC_NUM_FEATURES)
CORE_MODEL_NUM_FEATURES = list(CORE_NODE_INTRINSIC_NUM_FEATURES)
CORE_INTERACTION_FEATURES = [
    column for column in NODE_INTRINSIC_INTERACTION_FEATURES
    if column in CORE_MODEL_NUM_FEATURES
]

FEATURE_SET_CONFIGS = {
    "core_pre_day": {
        "label": "上线日前一天核心字段基线",
        "categorical": CORE_CAT_FEATURES,
        "numeric": CORE_NUM_FEATURES,
        "model_numeric": CORE_MODEL_NUM_FEATURES,
        "interaction": CORE_INTERACTION_FEATURES,
    },
    "enriched_pre_day": {
        "label": "上线日前一天节点固有属性增强字段",
        "categorical": NODE_INTRINSIC_CAT_FEATURES,
        "numeric": NODE_INTRINSIC_NUM_FEATURES,
        "model_numeric": MODEL_NUM_FEATURES,
        "interaction": MODEL_INTERACTION_FEATURES,
    },
}

VALIDATION_SEEDS = tuple(
    int(value.strip())
    for value in os.environ.get("MULTIBUSINESS_VALIDATION_SEEDS", "42,52,62,72,82").split(",")
    if value.strip()
)


def sql_quote(value: Any) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def node_filter(suffix: str, bucket: str, sub_bucket: str | None = None) -> str:
    conditions = [f"RIGHT(nodeId, 2) = {sql_quote(suffix)}"]
    if bucket == "y":
        conditions.append("SUBSTR(nodeId, 4, 1) = 'y'")
        conditions.append(f"SUBSTR(nodeId, 6, 1) = {sql_quote(sub_bucket or '')}")
    else:
        conditions.append(f"SUBSTR(nodeId, 4, 1) = {sql_quote(bucket)}")
    return " AND ".join(conditions)


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip()


def run_sql(sql: str, database_id: int, schema: str) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(3):
        client = SupersetSQLClient(
            base_url=BASE,
            database_id=database_id,
            query_limit=OUTCOME_QUERY_LIMIT,
        )
        try:
            return client.execute_sql(sql=sql, schema=schema)
        except Exception as exc:
            last_error = exc
            if attempt < 2 and ("502" in str(exc) or "503" in str(exc) or "504" in str(exc)):
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"SQL query failed after retries: {last_error}")


def query_eligible_pairs(suffix: str, bucket: str, sub_bucket: str | None = None) -> list[dict[str, Any]]:
    start = START_DAY.isoformat()
    end = END_DAY.isoformat()
    last_full_day = (END_DAY - dt.timedelta(days=6)).isoformat()
    sql = f"""
    WITH first_business AS (
      SELECT
        nodeId AS node_id,
        customerId AS business,
        MIN(day) AS first_business_day,
        COUNT(DISTINCT day) AS active_days,
        MAX(CASE WHEN state = 'online' THEN 1 ELSE 0 END) AS ever_online
      FROM node_day_ops_wide_full
        WHERE day BETWEEN DATE '{start}' AND DATE '{end}'
        AND customerId > 0
        AND {node_filter(suffix, bucket, sub_bucket)}
      GROUP BY nodeId, customerId
    ), full_window_business AS (
      SELECT *
      FROM first_business
      WHERE first_business_day <= DATE '{last_full_day}'
    ), eligible_nodes AS (
      SELECT
        node_id,
        MIN(first_business_day) AS online_day,
        COUNT(*) AS business_count,
        COUNT(DISTINCT first_business_day) AS business_first_day_count,
        MAX(ever_online) AS ever_online
      FROM full_window_business
      GROUP BY node_id
      HAVING COUNT(*) >= {MIN_BUSINESS_COUNT}
         AND COUNT(DISTINCT first_business_day) >= {MIN_BUSINESS_COUNT}
         AND MAX(ever_online) = 1
    )
    SELECT
      p.node_id,
      p.business,
      p.first_business_day AS business_online_day,
      p.active_days AS business_active_days,
      e.online_day,
      e.business_count,
      e.business_first_day_count,
      e.ever_online
    FROM full_window_business p
    JOIN eligible_nodes e ON p.node_id = e.node_id
    ORDER BY p.node_id, p.first_business_day, p.business
    """
    return run_sql(sql, database_id=19, schema="test")


def query_outcomes(suffix: str, bucket: str, sub_bucket: str | None = None) -> list[dict[str, Any]]:
    start = START_DAY.isoformat()
    end = END_DAY.isoformat()
    last_full_day = (END_DAY - dt.timedelta(days=6)).isoformat()
    sql = f"""
    WITH first_business AS (
      SELECT
        nodeId AS node_id,
        customerId AS business,
        MIN(day) AS first_business_day,
        COUNT(DISTINCT day) AS active_days,
        MAX(CASE WHEN state = 'online' THEN 1 ELSE 0 END) AS ever_online
      FROM node_day_ops_wide_full
        WHERE day BETWEEN DATE '{start}' AND DATE '{end}'
        AND customerId > 0
        AND {node_filter(suffix, bucket, sub_bucket)}
      GROUP BY nodeId, customerId
    ), full_window_business AS (
      SELECT *
      FROM first_business
      WHERE first_business_day <= DATE '{last_full_day}'
    ), eligible_nodes AS (
      SELECT
        node_id,
        MIN(first_business_day) AS online_day,
        COUNT(*) AS business_count,
        COUNT(DISTINCT first_business_day) AS business_first_day_count,
        MAX(ever_online) AS ever_online
      FROM full_window_business
      GROUP BY node_id
      HAVING COUNT(*) >= {MIN_BUSINESS_COUNT}
         AND COUNT(DISTINCT first_business_day) >= {MIN_BUSINESS_COUNT}
         AND MAX(ever_online) = 1
    )
    SELECT
      p.node_id,
      CAST(p.business AS VARCHAR) AS business,
      p.first_business_day AS business_online_day,
      p.active_days AS business_active_days,
      e.online_day,
      e.business_count,
      e.business_first_day_count,
      e.ever_online,
      MIN(COALESCE(CAST(t.customerName AS VARCHAR), '')) AS business_name,
      COALESCE(SUM(CAST(t.cost_finalAmount AS DOUBLE)), 0.0) AS cum_cost_7d,
      COALESCE(SUM(CAST(t.revenue_finalAmount AS DOUBLE)), 0.0) AS cum_revenue_7d,
      COUNT(t.day) AS outcome_days,
      COUNT(DISTINCT t.day) AS outcome_distinct_days
    FROM full_window_business p
    JOIN eligible_nodes e ON p.node_id = e.node_id
    LEFT JOIN node_day_ops_wide_full t
      ON t.nodeId = p.node_id
     AND t.customerId = p.business
     AND t.day >= p.first_business_day
     AND t.day <= DATE_ADD(p.first_business_day, INTERVAL 6 DAY)
    GROUP BY p.node_id, p.business, p.first_business_day, p.active_days,
             e.online_day, e.business_count, e.business_first_day_count, e.ever_online
    ORDER BY p.node_id, p.first_business_day, p.business
    """
    return run_sql(sql, database_id=19, schema="test")


def fetch_suffix_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_parts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    outcome_parts: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def fetch_one(suffix: str, bucket: str, sub_bucket: str | None) -> tuple[str, str, str, list[dict[str, Any]], list[dict[str, Any]]]:
        outcomes = query_outcomes(suffix, bucket, sub_bucket)
        pair_fields = [
            "node_id", "business", "business_online_day", "business_active_days",
            "online_day", "business_count", "business_first_day_count", "ever_online",
        ]
        pairs = [{field: row.get(field) for field in pair_fields} for row in outcomes]
        return suffix, bucket, sub_bucket or "", pairs, outcomes

    with ThreadPoolExecutor(max_workers=QUERY_WORKERS) as executor:
        futures = {
            executor.submit(fetch_one, suffix, bucket, sub_bucket): (suffix, bucket, sub_bucket)
            for suffix in SUFFIXES
            for bucket in NODE_BUCKETS
            for sub_bucket in (NODE_SUB_BUCKETS if bucket == "y" else (None,))
        }
        for future in as_completed(futures):
            suffix, bucket, sub_bucket, pairs, outcomes = future.result()
            pair_parts[(suffix, bucket, sub_bucket)] = pairs
            outcome_parts[(suffix, bucket, sub_bucket)] = outcomes
            if pairs or outcomes:
                label = f"{bucket}{sub_bucket}" if sub_bucket else bucket
                print(f"suffix {suffix} bucket {label}: eligible_pairs={len(pairs):,}, outcomes={len(outcomes):,}")

    pair_rows = [
        row for suffix in SUFFIXES for bucket in NODE_BUCKETS
        for sub_bucket in (NODE_SUB_BUCKETS if bucket == "y" else (None,))
        for row in pair_parts.get((suffix, bucket, sub_bucket or ""), [])
    ]
    outcome_rows = [
        row for suffix in SUFFIXES for bucket in NODE_BUCKETS
        for sub_bucket in (NODE_SUB_BUCKETS if bucket == "y" else (None,))
        for row in outcome_parts.get((suffix, bucket, sub_bucket or ""), [])
    ]
    pairs = pd.DataFrame(pair_rows)
    outcomes = pd.DataFrame(outcome_rows)
    if pairs.empty or outcomes.empty:
        raise RuntimeError("宽表筛选结果为空，拒绝继续训练。")
    pairs.to_csv(OUTPUT_PAIRS, index=False)
    outcomes.to_csv(OUTPUT_OUTCOMES, index=False)
    print(f"wrote {OUTPUT_PAIRS}: {len(pairs):,} rows")
    print(f"wrote {OUTPUT_OUTCOMES}: {len(outcomes):,} rows")
    return pairs, outcomes


def read_or_fetch_outcomes() -> tuple[pd.DataFrame, pd.DataFrame]:
    if OUTPUT_PAIRS.exists() and OUTPUT_OUTCOMES.exists() and os.environ.get("MULTIBUSINESS_REFRESH") != "1":
        pairs = pd.read_csv(OUTPUT_PAIRS, low_memory=False)
        outcomes = pd.read_csv(OUTPUT_OUTCOMES, low_memory=False)
        print("reuse cached multibusiness wide-table outputs")
        return pairs, outcomes
    return fetch_suffix_data()


def normalize_attribute_targets(node_targets: pd.DataFrame) -> pd.DataFrame:
    required = {"node_id", "online_day"}
    missing = required - set(node_targets.columns)
    if missing:
        raise RuntimeError(f"属性目标缺少字段: {sorted(missing)}")
    targets = node_targets[["node_id", "online_day"]].copy()
    targets["node_id"] = targets["node_id"].astype(str)
    parsed = pd.to_datetime(targets["online_day"], errors="coerce")
    if parsed.isna().any():
        bad = targets.loc[parsed.isna(), "node_id"].head(5).tolist()
        raise RuntimeError(f"属性目标上线日无法解析，示例节点: {bad}")
    targets["online_day"] = parsed.dt.strftime("%Y-%m-%d")
    attribute_dates = parsed - pd.to_timedelta(ATTRIBUTE_LOOKBACK_DAYS, unit="D")
    targets["attribute_day"] = attribute_dates.dt.strftime("%Y-%m-%d")
    targets["target_day"] = attribute_dates.dt.strftime("%Y%m%d")
    targets = targets.drop_duplicates("node_id").sort_values("node_id").reset_index(drop=True)
    if targets["node_id"].duplicated().any():
        raise RuntimeError("同一节点存在多个上线日，拒绝静默选择属性日期。")
    return targets


def attribute_target_sql_parts(node_targets: pd.DataFrame) -> tuple[str, str, str, str]:
    targets = normalize_attribute_targets(node_targets)
    ids = ",\n".join(sql_quote(value) for value in targets["node_id"])
    string_days = ",\n".join(sql_quote(value) for value in targets["target_day"])
    integer_days = ",\n".join(str(int(value)) for value in targets["target_day"])
    values = ",\n".join(
        f"({sql_quote(row.node_id)}, {sql_quote(row.online_day)}, {sql_quote(row.target_day)})"
        for row in targets.itertuples(index=False)
    )
    return ids, string_days, integer_days, values


def extract_bad_node_analysis_partitions(error: Exception) -> set[tuple[str, str]]:
    text = str(error)
    return {
        (day, hour)
        for day, hour in re.findall(
            r"node_analysis_data/day=(\d{8})/hour=(\d{2})/",
            text,
        )
    }


def attribute_sql_for_node_analysis(
    node_targets: pd.DataFrame,
    excluded_partitions: set[tuple[str, str]] | None = None,
) -> str:
    targets = normalize_attribute_targets(node_targets)
    if targets["target_day"].nunique() != 1:
        raise RuntimeError("node_analysis_data 属性批次必须只包含一个 day 分区。")
    ids = ",\n".join(sql_quote(value) for value in targets["node_id"])
    target_day = str(targets["target_day"].iloc[0])
    partition_guards = [
        f"NOT (nd.day = {sql_quote(day)} AND nd.hour = {sql_quote(hour)})"
        for day, hour in sorted(excluded_partitions or set())
        if day == target_day
    ]
    excluded_sql = "".join(f"\n        AND {guard}" for guard in partition_guards)
    return f"""
    SELECT * FROM (
      SELECT
        nd.nodeid AS node_id,
        nd.day AS analysis_attribute_day,
        nd.hour AS attribute_hour,
        nd.os, nd.arch, nd.isvm, nd.province, nd.city, nd.isp, nd.nattype,
        nd.dialtype, nd.resourcetype, nd.hardwaretype,
        nd.bandwidth, nd.actualbandwidth, nd.netbenchlimitbandwidth,
        nd.state AS node_state,
        nd.cpuload1, nd.cpuload5, nd.cpuload15, nd.corenum, nd.cpuuser,
        nd.cpusystem, nd.cpunice, nd.cpuidle, nd.cpuiowait, nd.cpuirq,
        nd.cpusoftirq, nd.cpusteal, nd.memtotal, nd.memused, nd.memfree,
        nd.membufferd, nd.memcached, nd.retrans, nd.v4pingloss, nd.v6pingloss,
        nd.v4pingavgrtt, nd.v4pingbestrtt, nd.v4pingworstrtt, nd.v6pingavgrtt,
        nd.v6pingbestrtt, nd.v6pingworstrtt, nd.totaldisksize,
        nd.totaldiskused, nd.hdddisksize, nd.hdddiskused, nd.ssddisksize,
        nd.ssddiskused, nd.systemdisksize, nd.systemdiskused, nd.maxioutil,
        nd.externalip,
        nd.idcid AS idc_id, nd.idcname AS idc_name,
        nd.stage AS node_analysis_stage, nd.nodecustmertype AS analysis_nodecustmertype,
        nd.nodedeliverytype AS analysis_nodedeliverytype,
        nd.tcpnattype AS analysis_tcpnattype, nd.udpnattype AS analysis_udpnattype,
        nd.cgroupversion AS analysis_cgroupversion,
        nd.lastdeploystate AS analysis_lastdeploystate,
        nd.lastdeployscenario AS analysis_lastdeployscenario,
        nd.cooperationtype AS analysis_cooperationtype,
        nd.supplysidedeliverytype AS analysis_supply_side_delivery_type,
        nd.isroot AS analysis_isroot, nd.issupportipv6 AS analysis_issupportipv6,
        nd.upnpstate AS analysis_upnpstate,
        nd.transprovrate AS analysis_transprovrate, nd.usbw AS analysis_usb_bw,
        nd.bwnum AS analysis_bw_num, nd.serviceduration AS analysis_serviceduration,
        nd.yesterdayp95bw AS analysis_yesterday_p95_bw,
        nd.yesterdayonlineduration AS analysis_yesterday_online_duration,
        nd.yesterdayeveningonlineduration AS analysis_yesterday_evening_online_duration,
        nd.yesterdayoutlinecount AS analysis_yesterday_outline_count,
        nd.yesterdayeveningoutlinecount AS analysis_yesterday_evening_outline_count,
        nd.yesterdaysnapbandwidth AS analysis_yesterday_snapshot_bw,
        nd.yesterdaysnapnetbenchbandwidth AS analysis_yesterday_snapshot_netbench_bw,
        nd.dbyonlineduration AS analysis_dby_online_duration,
        nd.dbyeveningonlineduration AS analysis_dby_evening_online_duration,
        nd.dbyoutlinecount AS analysis_dby_outline_count,
        nd.dbyeveningoutlinecount AS analysis_dby_evening_outline_count,
        nd.dbyp95bw AS analysis_dby_p95_bw,
        nd.dbysnapbandwidth AS analysis_dby_snapshot_bw,
        nd.dbysnapnetbenchbandwidth AS analysis_dby_snapshot_netbench_bw,
        nd.dby2p95bw AS analysis_dby2_p95_bw,
        nd.dby2onlineduration AS analysis_dby2_online_duration,
        nd.dby2outlinecount AS analysis_dby2_outline_count,
        nd.dby2eveningoutlinecount AS analysis_dby2_evening_outline_count,
        nd.dby2snapnetbenchbandwidth AS analysis_dby2_snapshot_netbench_bw,
        nd.outbandwidthtotal AS analysis_out_bandwidth_total,
        nd.outbandwidthbynode AS analysis_out_bandwidth_node,
        nd.outlimitbandwidthbynode AS analysis_out_limit_bandwidth_node,
        nd.outbandwidthudpbynode AS analysis_out_bandwidth_udp_node,
        nd.outlimitbandwidthudpbynode AS analysis_out_limit_bandwidth_udp_node,
        nd.outbandwidthudptotal AS analysis_out_bandwidth_udp_total,
        nd.udpactualbandwidth AS analysis_udp_actual_bandwidth,
        nd.udpnetbenchlimitbandwidth AS analysis_udp_netbench_limit_bandwidth,
        nd.connsucceedbwnum AS analysis_conn_succeed_bw_num,
        nd.abnormallinenum AS analysis_abnormal_line_num,
        nd.pinglossstatstotal AS analysis_ping_loss_stats_total,
        nd.pinglossp5statsabnormal AS analysis_ping_loss_p5_abnormal,
        nd.pinglossp10statsabnormal AS analysis_ping_loss_p10_abnormal,
        nd.retranstatstotal AS analysis_retrans_stats_total,
        nd.retranp5statsabnormal AS analysis_retrans_p5_abnormal,
        nd.cpuutilstatstotal AS analysis_cpuutil_stats_total,
        nd.cpuutilp70statsabnormal AS analysis_cpuutil_p70_abnormal,
        nd.ipv4packetcount AS analysis_ipv4_packet_count,
        nd.ipv6packetcount AS analysis_ipv6_packet_count,
        nd.diskdivbw AS analysis_disk_div_bw,
        ROW_NUMBER() OVER (
          PARTITION BY nd.nodeid
          ORDER BY TRY_CAST(nd.hour AS INTEGER) DESC, nd.time DESC
        ) AS rn
      FROM node_analysis_data nd
      WHERE nd.day = {sql_quote(target_day)}
        {excluded_sql}
        AND nd.nodeid IN ({ids})
    ) ranked
    WHERE rn = 1
    """


def attribute_sql_for_node_join(node_targets: pd.DataFrame) -> str:
    targets = normalize_attribute_targets(node_targets)
    if targets["target_day"].nunique() != 1:
        raise RuntimeError("node_join 属性批次必须只包含一个 day 分区。")
    ids = ",\n".join(sql_quote(value) for value in targets["node_id"])
    target_day = int(targets["target_day"].iloc[0])
    return f"""
    SELECT * FROM (
      SELECT
        nj._id AS node_id,
        nj.day AS join_attribute_day,
        nj.nodestaticinfo.vendorid AS vendorid,
        nj.nodestaticinfo.nominalinfo.bandwidth AS bw,
        nj.nodestaticinfo.nominalinfo.deliverytype AS deliverytype,
        nj.nodestaticinfo.nominalinfo.resourcetype AS join_resourcetype,
        nj.nodestaticinfo.nominalinfo.dialtype AS join_dialtype,
        nj.nodestaticinfo.nominalinfo.nattype AS join_nattype,
        element_at(nj.nodestaticinfo.nominalinfo.scheduleisps, 1) AS scheduleisps,
        nj.nodestaticinfo.regsource AS regsource,
        nj.nodestaticinfo.customermode AS customermode,
        nj.nodestaticinfo.nominalinfo.province AS join_province,
        nj.nodestaticinfo.nominalinfo.isp AS join_isp,
        nj.nodestaticinfo.nominalinfo.city AS join_city,
        nj.nodestaticinfo.nominalinfo.ismanaged AS join_ismanaged,
        nj.nodestaticinfo.nominalinfo.isminorisp AS join_isminorisp,
        nj.nodestaticinfo.nominalinfo.isbantransprov AS join_isbantransprov,
        nj.nodestaticinfo.nominalinfo.isipv6schedule AS join_isipv6schedule,
        nj.nodestaticinfo.nominalinfo.natforwardenable AS join_natforwardenable,
        nj.nodestaticinfo.idcbindtype AS join_idcbindtype,
        nj.nodestaticinfo.deploystate AS join_deploystate,
        nj.nodeinfo.devicetype AS device_type,
        nj.nodestaticinfo.nominalinfo.nodearchtype AS join_arch_type,
        CAST(nj.nodestaticinfo.nominalinfo.iscloudvm AS VARCHAR) AS join_isvm,
        nj.nodestaticinfo.nodeqoskillerconfig.status AS qoskiller_status,
        nj.nodeinfo.os.name AS join_os,
        nj.nodeinfo.arch AS join_arch,
        nj.nodeinfo.manufacturer AS join_manufacturer,
        nj.nodeinfo.model AS join_model,
        nj.nodeinfo.status AS node_status,
        nj.nodestaticinfo.stage AS node_stage,
        nj.nodeinfo.uptime AS join_uptime,
        nj.nodeinfo.durationtime AS join_duration_time,
        nj.nodeinfo.haserrors AS join_has_errors,
        nj.nodeinfo.iptableabnormal AS join_iptable_abnormal,
        nj.nodeinfo.bizbw AS join_biz_bw,
        nj.nodeinfo.yesterdayavgpeakratio AS join_yesterday_avg_peak_ratio,
        nj.nodeinfo.sevendayavg95ratio AS sevendayavg95ratio,
        nj.nodeinfo.qualityinfosummary.cpuutil AS quality_cpuutil,
        nj.nodeinfo.qualityinfosummary.retransrate AS quality_retransrate,
        nj.nodeinfo.qualityinfosummary.pinglossrate AS quality_pinglossrate,
        nj.nodeinfo.qualityinfosummary.rtt AS quality_rtt,
        nj.nodeinfo.qualityinfosummary.retranstats.abnormal AS join_quality_retrans_abnormal,
        nj.nodeinfo.qualityinfosummary.retranstats.total AS join_quality_retrans_total,
        nj.nodeinfo.qualityinfosummary.pinglossp5stats.abnormal AS join_quality_pingloss_p5_abnormal,
        nj.nodeinfo.qualityinfosummary.pinglossp5stats.total AS join_quality_pingloss_p5_total,
        nj.nodeinfo.qualityinfosummary.pinglossp10stats.abnormal AS join_quality_pingloss_p10_abnormal,
        nj.nodeinfo.qualityinfosummary.pinglossp10stats.total AS join_quality_pingloss_p10_total,
        nj.nodeinfo.qualityinfosummary.cpuutilp70stats.abnormal AS join_quality_cpuutil_p70_abnormal,
        nj.nodeinfo.qualityinfosummary.cpuutilp70stats.total AS join_quality_cpuutil_p70_total,
        nj.nodeinfo.cpu.load.load1 AS join_cpu_load1,
        nj.nodeinfo.cpu.load.load5 AS join_cpu_load5,
        nj.nodeinfo.cpu.load.load15 AS join_cpu_load15,
        nj.nodeinfo.cpu.corenumber AS join_cpu_corenumber,
        nj.nodeinfo.cpu.totalcores AS join_cpu_totalcores,
        nj.nodeinfo.cpu.totalphysicals AS join_cpu_totalphysicals,
        nj.nodeinfo.cpu.totalthreads AS join_cpu_totalthreads,
        nj.nodeinfo.cpu.idle AS join_cpu_idle,
        nj.nodeinfo.memory.memtotal AS join_memtotal,
        nj.nodeinfo.memory.used AS join_memused,
        nj.nodeinfo.memory.memfree AS join_memfree,
        nj.nodeinfo.memory.buffers AS join_membuffers,
        nj.nodeinfo.memory.cached AS join_memcached,
        nj.nodeinfo.disks.totalsize AS join_disks_totalsize,
        nj.nodeinfo.disks.totalsummary.size AS join_disks_total_size,
        nj.nodeinfo.disks.totalsummary.used AS join_disks_total_used,
        nj.nodeinfo.disks.totalsummary.usedrate AS join_disks_total_usedrate,
        nj.nodeinfo.disks.hddsummary.size AS join_disks_hdd_size,
        nj.nodeinfo.disks.hddsummary.used AS join_disks_hdd_used,
        nj.nodeinfo.disks.hddsummary.usedrate AS join_disks_hdd_usedrate,
        nj.nodeinfo.disks.ssdsummary.size AS join_disks_ssd_size,
        nj.nodeinfo.disks.ssdsummary.used AS join_disks_ssd_used,
        nj.nodeinfo.disks.ssdsummary.usedrate AS join_disks_ssd_usedrate,
        nj.nodeinfo.disks.systemdisksummary.size AS join_disks_system_size,
        nj.nodeinfo.disks.systemdisksummary.used AS join_disks_system_used,
        nj.nodeinfo.disks.systemdisksummary.usedrate AS join_disks_system_usedrate,
        nj.nodeinfo.netquality.pingrtt AS join_net_pingrtt,
        nj.nodeinfo.netquality.tcpretransrate AS join_net_tcpretransrate,
        nj.nodeinfo.netquality.pinglossrate AS join_net_pinglossrate,
        nj.nodeinfo.netbenchresults.actualbw AS join_actualbw,
        nj.nodeinfo.netbenchresults.limitbw AS join_limitbw,
        nj.nodeinfo.netbenchresults.actualbwachieverate AS join_actualbw_achieverate,
        nj.nodeinfo.netbenchresults.actualbwrtt AS join_actualbw_rtt,
        nj.nodeinfo.netbenchresults.actualbwtcprtrate AS join_actualbw_tcpretr_rate,
        nj.nodeinfo.netbenchresults.limitbwachieverate AS join_limitbw_achieverate,
        -- 部分历史 Parquet 分区将 bandwidth 声明为 bigint，Trino 无法跨分区读取；
        -- actualbw 是同一 netbench 结果中的可读带宽字段，作为该特征的保守回退。
        nj.nodeinfo.netbenchresults.actualbw AS join_netbench_bandwidth,
        nj.nodeinfo.netcardbw AS join_netcardbw,
        nj.nodeinfo.procbw AS join_procbw,
        nj.nodeinfo.diskdivbw AS join_diskdivbw,
        ROW_NUMBER() OVER (
          PARTITION BY nj._id
          ORDER BY nj.hour DESC, nj.nodeinfo.lastreporttime DESC
        ) AS rn
      FROM node_join nj
      WHERE nj.day = {target_day}
        AND nj._id IN ({ids})
    ) ranked
    WHERE rn = 1
    """


def attribute_sql_for_dial_acc(node_targets: pd.DataFrame) -> str:
    targets = normalize_attribute_targets(node_targets)
    if targets["target_day"].nunique() != 1:
        raise RuntimeError("dial_acc 属性批次必须只包含一个 day 分区。")
    ids = ",\n".join(sql_quote(value) for value in targets["node_id"])
    target_day = int(targets["target_day"].iloc[0])
    return f"""
    SELECT * FROM (
      SELECT
        da._id AS node_id,
        da.day AS dial_attribute_day,
        da.hour AS dial_attribute_hour,
        CASE
          WHEN da.dialisnormal THEN 1.0
          WHEN da.dialisnormal = false THEN 0.0
          ELSE NULL
        END AS dial_is_normal,
        da.ipv6enable AS dial_ipv6_enable,
        da.dialonphysicalnic AS dial_on_physical_nic,
        da.lbtype AS dial_lbtype,
        da.lbtypev6 AS dial_lbtype_v6,
        cardinality(da.accounts) AS dial_account_count,
        ROW_NUMBER() OVER (
          PARTITION BY da._id
          ORDER BY da.hour DESC, da.lastupdatenano DESC
        ) AS rn
      FROM dial_acc da
      WHERE da.day = {target_day}
        AND da._id IN ({ids})
    ) ranked
    WHERE rn = 1
    """


def fetch_attributes(node_targets: pd.DataFrame) -> pd.DataFrame:
    targets = normalize_attribute_targets(node_targets)
    jobs: list[pd.DataFrame] = []
    for _, day_targets in targets.groupby("target_day", sort=True):
        jobs.extend(
            day_targets.iloc[index:index + ATTRIBUTE_BATCH_SIZE].copy()
            for index in range(0, len(day_targets), ATTRIBUTE_BATCH_SIZE)
        )

    def fetch_one(batch: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        excluded_partitions: set[tuple[str, str]] = set()
        for attempt in range(5):
            try:
                analysis = run_sql(
                    attribute_sql_for_node_analysis(batch, excluded_partitions),
                    database_id=2,
                    schema="jarvis",
                )
                break
            except Exception as exc:
                bad_partitions = extract_bad_node_analysis_partitions(exc)
                new_partitions = bad_partitions - excluded_partitions
                if not new_partitions or attempt == 4:
                    raise
                excluded_partitions.update(new_partitions)
                UNREADABLE_NODE_ANALYSIS_PARTITIONS.update(new_partitions)
                print(
                    "skip unreadable node_analysis_data partitions for batch: "
                    + ",".join(f"day={day}/hour={hour}" for day, hour in sorted(new_partitions))
                )
        join = run_sql(attribute_sql_for_node_join(batch), database_id=2, schema="jarvis")
        dial = run_sql(attribute_sql_for_dial_acc(batch), database_id=2, schema="jarvis")
        return analysis, join, dial

    analysis_rows: list[dict[str, Any]] = []
    join_rows: list[dict[str, Any]] = []
    dial_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=QUERY_WORKERS) as executor:
        futures = [executor.submit(fetch_one, batch) for batch in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            analysis, join, dial = future.result()
            analysis_rows.extend(analysis)
            join_rows.extend(join)
            dial_rows.extend(dial)
            print(
                f"attribute batch {index}/{len(jobs)}: analysis={len(analysis):,}, "
                f"node_join={len(join):,}, dial_acc={len(dial):,}"
            )

    analysis = pd.DataFrame(analysis_rows)
    join = pd.DataFrame(join_rows)
    dial = pd.DataFrame(dial_rows)
    if analysis.empty and join.empty:
        raise RuntimeError("node_analysis_data 与 node_join 均未返回属性，拒绝训练。")
    if not analysis.empty:
        analysis = analysis.drop_duplicates("node_id")
    if not join.empty:
        join = join.drop_duplicates("node_id")
    if not dial.empty:
        dial = dial.drop_duplicates("node_id")
    nodes = targets[["node_id", "online_day", "attribute_day"]].copy()
    if not analysis.empty:
        nodes = nodes.merge(analysis, on="node_id", how="left")
    if not join.empty:
        nodes = nodes.merge(join, on="node_id", how="left")
    if not dial.empty:
        nodes = nodes.merge(dial, on="node_id", how="left")

    for left, right in [
        ("resourcetype", "join_resourcetype"), ("dialtype", "join_dialtype"),
        ("nattype", "join_nattype"), ("province", "join_province"),
        ("isp", "join_isp"), ("city", "join_city"), ("arch", "join_arch_type"),
        ("isvm", "join_isvm"), ("os", "join_os"), ("arch", "join_arch"),
        ("node_manufacturer", "join_manufacturer"), ("node_model", "join_model"),
        ("cpuload1", "join_cpu_load1"), ("cpuload5", "join_cpu_load5"),
        ("cpuload15", "join_cpu_load15"), ("corenum", "join_cpu_corenumber"),
        ("memtotal", "join_memtotal"), ("memused", "join_memused"),
        ("memfree", "join_memfree"), ("membufferd", "join_membuffers"),
        ("memcached", "join_memcached"), ("totaldisksize", "join_disks_total_size"),
        ("totaldiskused", "join_disks_total_used"), ("hdddisksize", "join_disks_hdd_size"),
        ("hdddiskused", "join_disks_hdd_used"), ("ssddisksize", "join_disks_ssd_size"),
        ("ssddiskused", "join_disks_ssd_used"), ("systemdisksize", "join_disks_system_size"),
        ("systemdiskused", "join_disks_system_used"), ("retrans", "join_net_tcpretransrate"),
        ("v4pingloss", "join_net_pinglossrate"), ("v6pingloss", "join_net_pinglossrate"),
        ("v4pingavgrtt", "join_net_pingrtt"), ("v6pingavgrtt", "join_net_pingrtt"),
        ("actualbandwidth", "join_actualbw"), ("netbenchlimitbandwidth", "join_limitbw"),
    ]:
        if left not in nodes:
            nodes[left] = np.nan
        if right in nodes:
            left_values = nodes[left].replace("", np.nan)
            nodes[left] = left_values.fillna(nodes[right])
    for column in CAT_FEATURES:
        if column not in nodes:
            nodes[column] = ""
    for column in NUM_FEATURES:
        if column not in nodes:
            nodes[column] = np.nan
    nodes["node_status"] = nodes["node_status"].fillna(nodes.get("node_state", np.nan))
    nodes["attribute_snapshot_mode"] = f"online_day_minus_{ATTRIBUTE_LOOKBACK_DAYS}d"
    nodes["health_snapshot_day"] = nodes["attribute_day"]
    nodes.to_csv(OUTPUT_NODES, index=False)
    print(
        f"wrote {OUTPUT_NODES}: {len(nodes):,} nodes; "
        f"attribute snapshot=online_day_minus_{ATTRIBUTE_LOOKBACK_DAYS}d"
    )
    return nodes


def add_derived_features(nodes: pd.DataFrame) -> pd.DataFrame:
    frame = nodes.copy()
    for column in NUM_FEATURES:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in [
        "retrans", "v4pingloss", "v6pingloss", "v4pingavgrtt", "v4pingbestrtt",
        "v4pingworstrtt", "v6pingavgrtt", "v6pingbestrtt", "v6pingworstrtt",
        "quality_retransrate", "quality_pinglossrate", "quality_rtt",
        "join_net_pingrtt", "join_net_tcpretransrate", "join_net_pinglossrate",
        "join_actualbw", "join_limitbw", "join_actualbw_achieverate",
        "join_actualbw_rtt", "join_actualbw_tcpretr_rate", "join_limitbw_achieverate",
        "analysis_transprovrate", "analysis_yesterday_p95_bw",
        "analysis_yesterday_online_duration", "analysis_yesterday_evening_online_duration",
        "analysis_yesterday_outline_count", "analysis_yesterday_evening_outline_count",
        "analysis_yesterday_snapshot_bw", "analysis_yesterday_snapshot_netbench_bw",
        "analysis_dby_online_duration", "analysis_dby_evening_online_duration",
        "analysis_dby_outline_count", "analysis_dby_evening_outline_count",
        "analysis_dby_p95_bw", "analysis_dby_snapshot_bw",
        "analysis_dby_snapshot_netbench_bw", "analysis_dby2_p95_bw",
        "analysis_dby2_online_duration", "analysis_dby2_outline_count",
        "analysis_dby2_evening_outline_count", "analysis_dby2_snapshot_netbench_bw",
        "analysis_out_bandwidth_total", "analysis_out_bandwidth_node",
        "analysis_out_limit_bandwidth_node", "analysis_out_bandwidth_udp_node",
        "analysis_out_limit_bandwidth_udp_node", "analysis_out_bandwidth_udp_total",
        "analysis_udp_actual_bandwidth", "analysis_udp_netbench_limit_bandwidth",
        "analysis_ping_loss_stats_total", "analysis_ping_loss_p5_abnormal",
        "analysis_ping_loss_p10_abnormal", "analysis_retrans_stats_total",
        "analysis_retrans_p5_abnormal", "analysis_cpuutil_stats_total",
        "analysis_cpuutil_p70_abnormal", "analysis_ipv4_packet_count",
        "analysis_ipv6_packet_count", "analysis_disk_div_bw", "join_uptime",
        "join_duration_time", "join_biz_bw", "join_yesterday_avg_peak_ratio",
        "join_quality_retrans_abnormal", "join_quality_retrans_total",
        "join_quality_pingloss_p5_abnormal", "join_quality_pingloss_p5_total",
        "join_quality_pingloss_p10_abnormal", "join_quality_pingloss_p10_total",
        "join_quality_cpuutil_p70_abnormal", "join_quality_cpuutil_p70_total",
    ]:
        if column in frame:
            frame[column] = frame[column].mask(frame[column] < 0)
    for column in CAT_FEATURES:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).replace({"nan": "", "None": ""})
    valid_cores = frame["corenum"].where(frame["corenum"] > 0)
    frame["cpu_load1_per_core"] = frame["cpuload1"] / valid_cores
    frame["mem_used_ratio"] = frame["memused"] / frame["memtotal"].replace(0, np.nan)
    frame["disk_used_ratio"] = frame["totaldiskused"] / frame["totaldisksize"].replace(0, np.nan)
    frame["hdd_used_ratio"] = frame["hdddiskused"] / frame["hdddisksize"].replace(0, np.nan)
    frame["ssd_used_ratio"] = frame["ssddiskused"] / frame["ssddisksize"].replace(0, np.nan)
    frame["system_disk_used_ratio"] = frame["systemdiskused"] / frame["systemdisksize"].replace(0, np.nan)
    frame["yesterday_online_ratio"] = frame["analysis_yesterday_online_duration"] / 86400.0
    frame["dby_online_ratio"] = frame["analysis_dby_online_duration"] / 86400.0
    frame["analysis_retrans_abnormal_ratio"] = (
        frame["analysis_retrans_p5_abnormal"]
        / frame["analysis_retrans_stats_total"].where(frame["analysis_retrans_stats_total"] > 0)
    )
    frame["analysis_pingloss_abnormal_ratio"] = (
        frame["analysis_ping_loss_p5_abnormal"]
        / frame["analysis_ping_loss_stats_total"].where(frame["analysis_ping_loss_stats_total"] > 0)
    )
    frame["analysis_cpuutil_abnormal_ratio"] = (
        frame["analysis_cpuutil_p70_abnormal"]
        / frame["analysis_cpuutil_stats_total"].where(frame["analysis_cpuutil_stats_total"] > 0)
    )
    frame["analysis_actual_bw_ratio"] = (
        frame["actualbandwidth"]
        / frame["bandwidth"].where(frame["bandwidth"] > 0)
    )
    frame["analysis_udp_actual_bw_ratio"] = (
        frame["analysis_udp_actual_bandwidth"]
        / frame["analysis_udp_netbench_limit_bandwidth"].where(
            frame["analysis_udp_netbench_limit_bandwidth"] > 0
        )
    )
    frame["analysis_out_limit_bw_ratio"] = (
        frame["analysis_out_limit_bandwidth_node"]
        / frame["analysis_out_bandwidth_node"].where(frame["analysis_out_bandwidth_node"] > 0)
    )
    return frame


def vm_query_payload(query: str, timeout: float = 30.0) -> dict[str, Any]:
    data = urllib.parse.urlencode({"query": query}).encode()
    request = urllib.request.Request(
        VM_BASE + "/query",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return {"status": "error", "http_status": exc.code, "response": body}
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "response": str(exc)}
    return payload


def vm_query(query: str, timeout: float = 30.0) -> dict[str, Any]:
    payload = vm_query_payload(query, timeout=timeout)
    if payload.get("status") != "success":
        return {"status": "error", "response": payload}
    return {"status": "success", "result_count": len(payload.get("data", {}).get("result", []))}


def fetch_prometheus_features(node_ids: list[str]) -> pd.DataFrame:
    """按 node_id 分批读取 SMART/ZFS/Prom 重传的真实值并聚合到节点级。"""
    metrics = {
        "smart_available_spare": "jarvis_disk_available_spare",
        "smart_bad_block_count": "jarvis_disk_bad_block_count",
        "smart_case_temperature": "jarvis_disk_temperature",
        "smart_critical_warning": "jarvis_disk_critical_warning",
        "zfs_pool_state": "node_zfs_zpool_state",
        "zfs_dataset_read": "node_zfs_zpool_dataset_nread",
        "zfs_dataset_write": "node_zfs_zpool_dataset_nwritten",
        "prom_retrans_ratio": "jarvis_node_tcp_retran_ratio_5m",
    }
    chunks = [node_ids[index:index + PROM_FEATURE_CHUNK_SIZE] for index in range(0, len(node_ids), PROM_FEATURE_CHUNK_SIZE)]

    def fetch_one(metric_name: str, metric: str, ids: list[str]) -> tuple[str, dict[str, Any]]:
        regex = "|".join(re.escape(node_id) for node_id in ids)
        return metric_name, vm_query_payload(f'{metric}{{node_id=~"{regex}"}}', timeout=60.0)

    raw: dict[str, list[dict[str, Any]]] = {name: [] for name in metrics}
    jobs = [
        (metric_name, metric, chunk)
        for metric_name, metric in metrics.items()
        for chunk in chunks
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_one, metric_name, metric, chunk) for metric_name, metric, chunk in jobs]
        for future in as_completed(futures):
            metric_name, payload = future.result()
            if payload.get("status") == "success":
                raw[metric_name].extend(payload.get("data", {}).get("result", []))

    values: dict[str, dict[str, list[float]]] = {}
    zfs_states: dict[str, list[str]] = {}
    for metric_name, items in raw.items():
        for item in items:
            labels = item.get("metric", {})
            node_id = labels.get("node_id")
            if not node_id:
                continue
            try:
                value = float(item.get("value", [None, "nan"])[1])
            except (TypeError, ValueError):
                continue
            if metric_name == "zfs_pool_state":
                zfs_states.setdefault(node_id, []).append(str(labels.get("state", "")))
                continue
            if metric_name == "prom_retrans_ratio":
                value *= 100.0
            values.setdefault(node_id, {}).setdefault(metric_name, []).append(value)

    rows: list[dict[str, Any]] = []
    for node_id in node_ids:
        item: dict[str, Any] = {"node_id": node_id}
        node_values = values.get(node_id, {})
        for metric_name, metric_values in node_values.items():
            if metric_name == "smart_available_spare":
                item[metric_name] = min(metric_values)
            elif metric_name in {"smart_bad_block_count", "zfs_dataset_read", "zfs_dataset_write"}:
                item[metric_name] = sum(metric_values)
            elif metric_name in {"smart_case_temperature", "smart_critical_warning", "prom_retrans_ratio"}:
                item[metric_name] = max(metric_values)
            else:
                item[metric_name] = float(np.mean(metric_values))
        if node_id in zfs_states:
            states = zfs_states[node_id]
            item["zfs_pool_online"] = float(all(state == "online" for state in states))
        rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame({"node_id": node_ids})
    for metric_name in metrics:
        if metric_name not in result.columns:
            result[metric_name] = np.nan
    if "zfs_pool_online" not in result.columns:
        result["zfs_pool_online"] = np.nan
    return result


def enrich_nodes_with_prometheus(nodes: pd.DataFrame, node_ids: list[str]) -> pd.DataFrame:
    prom = fetch_prometheus_features(node_ids)
    merged = nodes.drop(columns=[column for column in prom.columns if column != "node_id" and column in nodes.columns], errors="ignore").merge(prom, on="node_id", how="left")
    merged.to_csv(OUTPUT_NODES, index=False)
    coverage = {column: float(merged[column].notna().mean()) for column in prom.columns if column != "node_id"}
    print(f"Prometheus feature coverage: {coverage}")
    return merged


def probe_prometheus(node_ids: list[str]) -> dict[str, Any]:
    sample = node_ids[:80]
    regex = "|".join(re.escape(node_id) for node_id in sample)
    metrics = {
        "node_cpu_seconds_total": "node_cpu_seconds_total",
        "node_memory_MemTotal_bytes": "node_memory_MemTotal_bytes",
        "node_disk_read_bytes_total": "node_disk_read_bytes_total",
        "smart_available_spare": "jarvis_disk_available_spare",
        "smart_bad_block_count": "jarvis_disk_bad_block_count",
        "smart_case_temperature": "jarvis_disk_temperature",
        "smart_critical_warning": "jarvis_disk_critical_warning",
        "zfs_pool_state": "node_zfs_zpool_state",
        "zfs_dataset_read": "node_zfs_zpool_dataset_nread",
        "zfs_dataset_write": "node_zfs_zpool_dataset_nwritten",
        "prom_retrans_ratio": "jarvis_node_tcp_retran_ratio_5m",
    }
    result: dict[str, Any] = {
        "evaluation_time_bj": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "sample_node_count": len(sample),
        "label_assumption": "按 node_id 直接匹配；未使用不可证实的 ID 变换",
        "metrics": {},
    }
    for name, metric in metrics.items():
        payload = vm_query(f'{metric}{{node_id=~"{regex}"}}')
        result["metrics"][name] = payload
    result["mapping_usable"] = any(
        value.get("status") == "success" and value.get("result_count", 0) > 0
        for value in result["metrics"].values()
    )
    with open(OUTPUT_PROM, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"wrote {OUTPUT_PROM}; direct node_id mapping usable={result['mapping_usable']}")
    return result


class RobustEncoder:
    def __init__(
        self,
        cat_features: list[str],
        num_features: list[str],
        business_col: str,
        interaction_features: list[str],
    ):
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.business_col = business_col
        self.interaction_features = list(interaction_features)
        self.cat_onehot: OneHotEncoder | None = None
        self.business_onehot: OneHotEncoder | None = None
        self.numeric_fill: dict[str, float] = {}
        self.numeric_scale: dict[str, float] = {}
        self.output_names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "RobustEncoder":
        cat_frame = frame[self.cat_features].fillna("__MISSING__").astype(str)
        business_frame = frame[[self.business_col]].fillna("__MISSING__").astype(str)
        try:
            self.cat_onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            self.business_onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            self.cat_onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)
            self.business_onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)
        self.cat_onehot.fit(cat_frame)
        self.business_onehot.fit(business_frame)
        self.numeric_fill = {}
        self.numeric_scale = {}
        for column in self.num_features:
            values = pd.to_numeric(frame[column], errors="coerce")
            self.numeric_fill[column] = float(values.median()) if values.notna().any() else 0.0
            observed = values.dropna()
            if len(observed) >= 2:
                scale = float(observed.quantile(0.75) - observed.quantile(0.25))
                if not math.isfinite(scale) or scale <= 1e-12:
                    scale = float(observed.std(ddof=0))
            else:
                scale = 1.0
            self.numeric_scale[column] = scale if math.isfinite(scale) and scale > 1e-12 else 1.0
        names = list(self.cat_onehot.get_feature_names_out(self.cat_features))
        names.extend(self.business_onehot.get_feature_names_out([self.business_col]))
        names.extend(self.num_features)
        names.extend(f"{column}__missing" for column in self.num_features)
        names.extend(
            f"business_x_{column}__{business}"
            for column in self.interaction_features
            for business in self.business_onehot.categories_[0]
        )
        self.output_names = names
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if self.cat_onehot is None or self.business_onehot is None:
            raise RuntimeError("encoder is not fitted")
        cat_frame = frame[self.cat_features].fillna("__MISSING__").astype(str)
        business_frame = frame[[self.business_col]].fillna("__MISSING__").astype(str)
        cat_matrix = self.cat_onehot.transform(cat_frame)
        business_matrix = self.business_onehot.transform(business_frame)
        numeric_parts = []
        missing_parts = []
        for column in self.num_features:
            values = pd.to_numeric(frame[column], errors="coerce")
            missing_parts.append(values.isna().astype(float).to_numpy().reshape(-1, 1))
            scaled = (
                values.fillna(self.numeric_fill[column]).to_numpy(dtype=float)
                - self.numeric_fill[column]
            ) / self.numeric_scale[column]
            numeric_parts.append(scaled.reshape(-1, 1))
        numeric_matrix = np.concatenate(numeric_parts, axis=1)
        missing_matrix = np.concatenate(missing_parts, axis=1)
        interaction_parts = []
        for column in self.interaction_features:
            index = self.num_features.index(column)
            interaction_parts.append(business_matrix.multiply(numeric_matrix[:, index].reshape(-1, 1)))
        return sparse.hstack(
            [cat_matrix, business_matrix, sparse.csr_matrix(numeric_matrix),
             sparse.csr_matrix(missing_matrix), *interaction_parts],
            format="csr",
        )


class BusinessClassifier:
    """按节点历史最佳业务训练分类器，用于直接优化节点内 Top-1 推荐。"""

    def __init__(
        self,
        cat_features: list[str] | None = None,
        num_features: list[str] | None = None,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.cat_features = list(cat_features or CAT_FEATURES)
        self.num_features = list(num_features or MODEL_NUM_FEATURES)
        self.random_seed = int(random_seed)
        self.cat_encoder: OneHotEncoder | None = None
        self.numeric_fill: dict[str, float] = {}
        self.feature_names: list[str] = []
        self.cost_classifier: ExtraTreesClassifier | None = None
        self.profit_classifier: ExtraTreesClassifier | None = None

    def _fit_encoder(self, frame: pd.DataFrame) -> None:
        cat_frame = frame[self.cat_features].fillna("__MISSING__").astype(str)
        try:
            self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
        self.cat_encoder.fit(cat_frame)
        self.feature_names = list(self.num_features)
        self.feature_names.extend(f"{column}__missing" for column in self.num_features)
        self.feature_names.extend(self.cat_encoder.get_feature_names_out(self.cat_features))
        self.numeric_fill = {}
        for column in self.num_features:
            values = pd.to_numeric(frame[column], errors="coerce")
            self.numeric_fill[column] = float(values.median()) if values.notna().any() else 0.0

    def _transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if self.cat_encoder is None:
            raise RuntimeError("business classifier encoder is not fitted")
        cat_frame = frame[self.cat_features].fillna("__MISSING__").astype(str)
        cat_matrix = self.cat_encoder.transform(cat_frame)
        numeric_parts = []
        missing_parts = []
        for column in self.num_features:
            values = pd.to_numeric(frame[column], errors="coerce")
            missing_parts.append(values.isna().astype(float).to_numpy().reshape(-1, 1))
            numeric_parts.append(
                values.fillna(self.numeric_fill[column]).to_numpy(dtype=float).reshape(-1, 1)
            )
        numeric_matrix = sparse.csr_matrix(np.concatenate(numeric_parts, axis=1))
        missing_matrix = sparse.csr_matrix(np.concatenate(missing_parts, axis=1))
        return sparse.hstack([numeric_matrix, missing_matrix, cat_matrix], format="csr")

    def _classifier(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=int(os.environ.get("MULTIBUSINESS_TREE_ESTIMATORS", "200")),
            max_depth=int(os.environ.get("MULTIBUSINESS_TREE_MAX_DEPTH", "18")),
            min_samples_leaf=int(os.environ.get("MULTIBUSINESS_TREE_MIN_SAMPLES_LEAF", "5")),
            max_features=os.environ.get("MULTIBUSINESS_TREE_MAX_FEATURES", "sqrt"),
            n_jobs=int(os.environ.get("MULTIBUSINESS_TREE_JOBS", "-1")),
            random_state=self.random_seed,
        )

    def fit(self, frame: pd.DataFrame) -> "BusinessClassifier":
        train = frame.copy()
        train["business"] = train["business"].astype(str)
        node_frame = train.drop_duplicates("node_id").copy()
        self._fit_encoder(node_frame)
        x = self._transform(node_frame)
        labels: dict[str, pd.Series] = {}
        for target in ["cum_cost_7d", "cum_profit_7d"]:
            best_rows = train.loc[train.groupby("node_id")[target].idxmax()]
            labels[target] = best_rows.set_index("node_id")["business"].astype(str)
        node_ids = node_frame["node_id"].astype(str)
        for target, attr in [
            ("cum_cost_7d", "cost_classifier"),
            ("cum_profit_7d", "profit_classifier"),
        ]:
            label_values = node_ids.map(labels[target]).fillna("").to_numpy()
            valid = label_values != ""
            classifier = self._classifier().fit(x[valid], label_values[valid])
            setattr(self, attr, classifier)
        return self

    @staticmethod
    def _candidate_probabilities(
        classifier: ExtraTreesClassifier,
        probabilities: np.ndarray,
        businesses: list[str],
    ) -> np.ndarray:
        output = np.zeros((len(probabilities), len(businesses)), dtype=float)
        class_index = {str(value): index for index, value in enumerate(classifier.classes_)}
        for index, business in enumerate(businesses):
            source_index = class_index.get(str(business))
            if source_index is not None:
                output[:, index] = probabilities[:, source_index]
        return output

    def score(self, nodes: pd.DataFrame, businesses: list[str]) -> pd.DataFrame:
        if self.cost_classifier is None or self.profit_classifier is None:
            raise RuntimeError("business classifier is not fitted")
        base = nodes[["node_id"] + self.cat_features + self.num_features].drop_duplicates("node_id")
        x = self._transform(base)
        cost = self._candidate_probabilities(
            self.cost_classifier,
            self.cost_classifier.predict_proba(x),
            businesses,
        )
        profit = self._candidate_probabilities(
            self.profit_classifier,
            self.profit_classifier.predict_proba(x),
            businesses,
        )
        business_array = np.asarray(businesses, dtype=object)
        return pd.DataFrame({
            "node_id": np.repeat(base["node_id"].to_numpy(), len(businesses)),
            "business": np.tile(business_array, len(base)),
            "pred_cost": cost.reshape(-1),
            "pred_profit": profit.reshape(-1),
        })


class Recommender:
    def __init__(
        self,
        cat_features: list[str] | None = None,
        num_features: list[str] | None = None,
        interaction_features: list[str] | None = None,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.cat_features = list(cat_features or CAT_FEATURES)
        self.num_features = list(num_features or MODEL_NUM_FEATURES)
        self.interaction_features = list(interaction_features or MODEL_INTERACTION_FEATURES)
        self.random_seed = int(random_seed)
        self.encoder = RobustEncoder(
            self.cat_features,
            self.num_features,
            "business",
            self.interaction_features,
        )
        self.cost_model: Ridge | None = None
        self.op_model: Ridge | None = None
        self.business_classifier: BusinessClassifier | None = None
        self.businesses: list[str] = []
        self.target_transform = os.environ.get("MULTIBUSINESS_TARGET_TRANSFORM", "signed_log")
        self.cost_target_max = 0.0
        self.profit_target_abs_max = 0.0

    def _transform_cost(self, values: pd.Series | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if self.target_transform == "none":
            return array
        return np.log1p(np.clip(array, 0.0, None))

    def _inverse_cost(self, values: np.ndarray) -> np.ndarray:
        if self.target_transform == "none":
            return np.asarray(values, dtype=float)
        upper = math.log1p(max(self.cost_target_max, 0.0))
        transformed = np.clip(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=upper, neginf=0.0), 0.0, upper)
        return np.maximum(np.expm1(transformed), 0.0)

    def _transform_profit(self, values: pd.Series | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if self.target_transform == "none":
            return array
        return np.sign(array) * np.log1p(np.abs(array))

    def _inverse_profit(self, values: np.ndarray) -> np.ndarray:
        if self.target_transform == "none":
            return np.asarray(values, dtype=float)
        array = np.asarray(values, dtype=float)
        upper = math.log1p(max(self.profit_target_abs_max, 0.0))
        transformed = np.clip(np.nan_to_num(array, nan=0.0, posinf=upper, neginf=-upper), -upper, upper)
        return np.sign(transformed) * np.expm1(np.abs(transformed))

    def fit(self, frame: pd.DataFrame) -> "Recommender":
        train = frame.copy()
        train["business"] = train["business"].astype(str)
        self.cost_target_max = float(pd.to_numeric(train["cum_cost_7d"], errors="coerce").fillna(0.0).max())
        self.profit_target_abs_max = float(pd.to_numeric(train["cum_profit_7d"], errors="coerce").fillna(0.0).abs().max())
        self.encoder.fit(train)
        x = self.encoder.transform(train)
        common = {
            "alpha": float(os.environ.get("MULTIBUSINESS_RIDGE_ALPHA", "10.0")),
            "solver": "lsqr",
        }
        self.cost_model = Ridge(**common).fit(x, self._transform_cost(train["cum_cost_7d"]))
        self.op_model = Ridge(**common).fit(x, self._transform_profit(train["cum_profit_7d"]))
        self.business_classifier = BusinessClassifier(
            self.cat_features,
            self.num_features,
            self.random_seed,
        ).fit(train)
        self.businesses = sorted(train["business"].unique())
        return self

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.cost_model is None or self.op_model is None:
            raise RuntimeError("model is not fitted")
        x = self.encoder.transform(frame)
        return pd.DataFrame({
            "pred_cost": self._inverse_cost(self.cost_model.predict(x)),
            "pred_profit": self._inverse_profit(self.op_model.predict(x)),
        }, index=frame.index)

    def score(self, nodes: pd.DataFrame, businesses: list[str]) -> pd.DataFrame:
        if self.business_classifier is None:
            raise RuntimeError("model is not fitted")
        return self.business_classifier.score(nodes, businesses)

    def predict_amounts(self, nodes: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if pairs.empty:
            return pd.DataFrame(columns=["pred_cost", "pred_profit"])
        base = nodes[["node_id"] + self.cat_features + self.num_features].drop_duplicates("node_id")
        frame = pairs[["node_id", "business"]].merge(base, on="node_id", how="left", sort=False)
        return self.predict_frame(frame)


def build_health_bounds(train: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    bounds: dict[str, dict[str, dict[str, float]]] = {}
    metric_columns = list(HEALTH_METRICS)
    for business, group in train.groupby("business"):
        bounds[str(business)] = {}
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values) < 20:
                continue
            bounds[str(business)][metric] = {
                "q01": float(values.quantile(0.01)),
                "q99": float(values.quantile(0.99)),
                "n": int(len(values)),
            }
    return bounds


def health_status(row: pd.Series, business: str, bounds: dict[str, dict[str, dict[str, float]]]) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    unknown: list[str] = []
    values = {metric: pd.to_numeric(row.get(metric), errors="coerce") for metric in HEALTH_METRICS}
    checks = [
        ("cpu_load1_per_core", "cpu_load1_per_core_max", "CPU负载/核超过阈值", lambda v, t: v > t),
        ("mem_used_ratio", "mem_used_ratio_max", "内存使用率超过阈值", lambda v, t: v > t),
        ("disk_used_ratio", "disk_used_ratio_max", "磁盘使用率超过阈值", lambda v, t: v > t),
        ("retrans", "retrans_max", "node_analysis重传超过阈值", lambda v, t: v > t),
        ("v4pingloss", "v4pingloss_max", "IPv4丢包超过阈值", lambda v, t: v > t),
        ("v6pingloss", "v6pingloss_max", "IPv6丢包超过阈值", lambda v, t: v > t),
        ("quality_retransrate", "quality_retransrate_max", "node_join重传超过阈值", lambda v, t: v > t),
        ("quality_pinglossrate", "quality_pinglossrate_max", "node_join丢包超过阈值", lambda v, t: v > t),
        ("maxioutil", "maxioutil_max", "磁盘IO利用率超过阈值", lambda v, t: v > t),
        ("smart_available_spare", "smart_available_spare_min", "SMART可用spare低于阈值", lambda v, t: v < t),
        ("smart_bad_block_count", "smart_bad_block_count_max", "SMART坏块超过阈值", lambda v, t: v > t),
        ("smart_critical_warning", "smart_critical_warning_max", "SMART critical warning超过阈值", lambda v, t: v > t),
        ("prom_retrans_ratio", "prom_retrans_ratio_max", "Prometheus重传超过阈值", lambda v, t: v > t),
        ("dial_is_normal", "dial_is_normal_required", "拨号状态异常", lambda v, t: v != t),
        (
            "analysis_retrans_abnormal_ratio",
            "analysis_retrans_abnormal_ratio_max",
            "历史重传异常比例超过阈值",
            lambda v, t: v > t,
        ),
    ]
    for metric, threshold_key, label, predicate in checks:
        value = values[metric]
        threshold = GLOBAL_HEALTH_RULES[threshold_key]
        if pd.isna(value):
            unknown.append(metric)
        elif predicate(float(value), threshold):
            if threshold_key == "dial_is_normal_required":
                comparator = "!="
            else:
                comparator = "<" if threshold_key.endswith("_min") else ">"
            reasons.append(f"{label}({float(value):.4g}{comparator}{threshold:g})")
    zfs = values["zfs_pool_online"]
    if pd.isna(zfs):
        unknown.append("zfs_pool_online")
    elif int(zfs) != GLOBAL_HEALTH_RULES["zfs_pool_online_required"]:
        reasons.append("ZFS池非online")

    for metric in sorted(HEALTH_MANUAL_ONLY_METRICS):
        if pd.isna(values[metric]):
            unknown.append(metric)
        else:
            unknown.append(f"{metric}_threshold_not_configured")

    business_bounds = bounds.get(str(business), {})
    for metric, setting in business_bounds.items():
        value = values.get(metric)
        if value is None or pd.isna(value):
            continue
        if metric == "smart_available_spare":
            if value < setting["q01"]:
                reasons.append(f"业务历史可行域外:{metric}={float(value):.4g}<{setting['q01']:.4g}")
        elif metric in {"smart_bad_block_count", "smart_critical_warning", "zfs_pool_online"}:
            if value > setting["q99"]:
                reasons.append(f"业务历史可行域外:{metric}={float(value):.4g}>{setting['q99']:.4g}")
        elif value > setting["q99"]:
            reasons.append(f"业务历史可行域外:{metric}={float(value):.4g}>{setting['q99']:.4g}")
    if reasons:
        return False, reasons, "blocked"
    if unknown:
        return True, [f"指标缺失未判定:{','.join(sorted(set(unknown)))}"], "needs_manual_check"
    return True, [], "pass"


def recommend(model: Recommender, nodes: pd.DataFrame, businesses: list[str], bounds: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    scored = model.score(nodes, businesses)
    outputs: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    for node_id, group in scored.groupby("node_id", sort=True):
        feasible = []
        health_meta: dict[str, tuple[bool, list[str], str]] = {}
        node_row = nodes.loc[nodes["node_id"] == node_id].iloc[0]
        for business in businesses:
            ok, reasons, status = health_status(node_row, business, bounds)
            health_meta[business] = (ok, reasons, status)
            if ok:
                feasible.append(business)
        feasible_group = group[group["business"].isin(feasible)]
        if feasible_group.empty:
            miner_best = ""
            operator_best = ""
            miner_row = None
            operator_row = None
            recommendation_status = "no_feasible_business"
        else:
            miner_row = feasible_group.sort_values(["pred_cost", "business"], ascending=[False, True]).iloc[0]
            operator_row = feasible_group.sort_values(["pred_profit", "business"], ascending=[False, True]).iloc[0]
            miner_best = str(miner_row["business"])
            operator_best = str(operator_row["business"])
            recommendation_status = "needs_manual_check" if any(health_meta[b][2] == "needs_manual_check" for b in feasible) else "pass"
        blocked = [f"{b}:{'|'.join(health_meta[b][1])}" for b in businesses if not health_meta[b][0]]
        output = {
            "node_id": node_id,
            "miner_best": miner_best if recommendation_status == "pass" else "",
            "miner_pred_cost": np.nan,
            "miner_model_probability": float(miner_row["pred_cost"]) if miner_row is not None else np.nan,
            "operator_best": operator_best if recommendation_status == "pass" else "",
            "operator_pred_profit": np.nan,
            "operator_model_probability": float(operator_row["pred_profit"]) if operator_row is not None else np.nan,
            "miner_best_provisional": miner_best,
            "operator_best_provisional": operator_best,
            "feasible_business_count": len(feasible),
            "blocked_business_count": len(blocked),
            "recommendation_status": recommendation_status,
            "blocked_business_reasons": ";".join(blocked),
        }
        for mode, metric in [("miner", "pred_cost"), ("operator", "pred_profit")]:
            top = feasible_group.sort_values([metric, "business"], ascending=[False, True]).head(3) if not feasible_group.empty else feasible_group
            output[f"{mode}_top3"] = ";".join(f"{row.business}:{float(getattr(row, metric)):.6g}" for row in top.itertuples())
        output_index = len(outputs)
        if miner_row is not None:
            selected_pairs.append({
                "output_index": output_index,
                "role": "miner",
                "node_id": str(node_id),
                "business": str(miner_row["business"]),
            })
        if operator_row is not None:
            selected_pairs.append({
                "output_index": output_index,
                "role": "operator",
                "node_id": str(node_id),
                "business": str(operator_row["business"]),
            })
        outputs.append(output)
    if selected_pairs:
        selected_frame = pd.DataFrame(selected_pairs)
        amounts = model.predict_amounts(nodes, selected_frame)
        for selected, prediction in zip(selected_pairs, amounts.itertuples(index=False)):
            if selected["role"] == "miner":
                outputs[selected["output_index"]]["miner_pred_cost"] = float(prediction.pred_cost)
            else:
                outputs[selected["output_index"]]["operator_pred_profit"] = float(prediction.pred_profit)
    return pd.DataFrame(outputs)


def bootstrap_difference(model_hit: np.ndarray, baseline_hit: np.ndarray, seed: int = RANDOM_SEED) -> dict[str, float]:
    if len(model_hit) == 0:
        return {"difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value": 1.0}
    difference = model_hit.astype(float) - baseline_hit.astype(float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(difference, size=(2000, len(difference)), replace=True).mean(axis=1)
    permutations = np.empty(2000)
    for index in range(2000):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(difference))
        permutations[index] = float((difference * signs).mean())
    exceedances = int((np.abs(permutations) >= abs(float(difference.mean()))).sum())
    p_value = float((exceedances + 1) / (len(permutations) + 1))
    return {
        "difference": float(difference.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "p_value": p_value,
    }


def evaluate(
    split_name: str,
    model: Recommender,
    nodes: pd.DataFrame,
    outcomes: pd.DataFrame,
    businesses: list[str],
    bounds: dict[str, dict[str, dict[str, float]]],
    baseline_miner_business: str,
    baseline_operator_business: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    split_nodes = nodes[nodes["node_id"].isin(outcomes["node_id"].unique())].copy()
    rec = recommend(model, split_nodes, businesses, bounds).set_index("node_id")
    predictions = model.predict_frame(outcomes)
    actual_cost = pd.to_numeric(outcomes["cum_cost_7d"], errors="coerce").fillna(0.0).to_numpy()
    actual_profit = pd.to_numeric(outcomes["cum_profit_7d"], errors="coerce").fillna(0.0).to_numpy()
    predicted_cost = predictions["pred_cost"].to_numpy()
    predicted_profit = predictions["pred_profit"].to_numpy()
    hit_cost: list[bool] = []
    hit_op: list[bool] = []
    evaluation_node_ids: list[str] = []
    base_cost: list[bool] = []
    base_op: list[bool] = []
    base_cost_raw: list[bool] = []
    base_op_raw: list[bool] = []
    cost_regret: list[float] = []
    op_regret: list[float] = []
    cost_capture: list[float] = []
    op_capture: list[float] = []
    op_nonpositive_best = 0
    op_near_zero_best = 0
    cost_covered = 0
    op_covered = 0
    no_rec = 0
    for node_id, group in outcomes.groupby("node_id"):
        if node_id not in rec.index:
            continue
        evaluation_node_ids.append(str(node_id))
        group = group.set_index("business")
        costs = pd.to_numeric(group["cum_cost_7d"], errors="coerce").fillna(0.0)
        profits = pd.to_numeric(group["cum_profit_7d"], errors="coerce").fillna(0.0)
        true_cost = str(costs.idxmax())
        true_op = str(profits.idxmax())
        row = rec.loc[node_id]
        miner = str(row["miner_best_provisional"])
        operator = str(row["operator_best_provisional"])
        node_row = split_nodes.loc[split_nodes["node_id"] == node_id].iloc[0]
        baseline_miner_allowed = health_status(node_row, baseline_miner_business, bounds)[0]
        baseline_operator_allowed = health_status(node_row, baseline_operator_business, bounds)[0]
        hit_cost.append(miner == true_cost)
        hit_op.append(operator == true_op)
        base_cost.append(baseline_miner_allowed and str(baseline_miner_business) == true_cost)
        base_op.append(baseline_operator_allowed and str(baseline_operator_business) == true_op)
        base_cost_raw.append(str(baseline_miner_business) == true_cost)
        base_op_raw.append(str(baseline_operator_business) == true_op)
        if not miner:
            no_rec += 1
        elif miner in group.index:
            best = float(costs.max())
            selected = float(costs.loc[miner])
            cost_covered += 1
            cost_regret.append(max(0.0, (best - selected) / max(abs(best), 1e-9)))
            cost_capture.append(selected / best if best > 0 else 1.0)
        if not operator:
            no_rec += 1
        elif operator in group.index:
            best = float(profits.max())
            selected = float(profits.loc[operator])
            op_covered += 1
            if best > 1e-6:
                op_regret.append(max(0.0, (best - selected) / best))
                op_capture.append(selected / best)
            elif best > 0:
                op_near_zero_best += 1
            else:
                op_nonpositive_best += 1

    metrics = {
        "split": split_name,
        "n_nodes": len(hit_cost),
        "miner_hit_rate_at_1": float(np.mean(hit_cost)) if hit_cost else 0.0,
        "operator_hit_rate_at_1": float(np.mean(hit_op)) if hit_op else 0.0,
        "miner_hit_rate_at_3": None,
        "operator_hit_rate_at_3": None,
        "miner_realized_coverage": cost_covered / len(hit_cost) if hit_cost else 0.0,
        "operator_realized_coverage": op_covered / len(hit_op) if hit_op else 0.0,
        "miner_realized_regret": float(np.mean(cost_regret)) if cost_regret else None,
        "operator_realized_regret": float(np.mean(op_regret)) if op_regret else None,
        "miner_value_capture": float(np.mean(cost_capture)) if cost_capture else None,
        "operator_value_capture": float(np.mean(op_capture)) if op_capture else None,
        "operator_nonpositive_best_count": op_nonpositive_best,
        "operator_near_zero_best_count": op_near_zero_best,
        "no_recommendation_events": no_rec,
        "baseline_business_miner": str(baseline_miner_business),
        "baseline_business_operator": str(baseline_operator_business),
        "baseline_miner_hit_rate_unfiltered": float(np.mean(base_cost_raw)) if base_cost_raw else 0.0,
        "baseline_operator_hit_rate_unfiltered": float(np.mean(base_op_raw)) if base_op_raw else 0.0,
        "baseline_miner_hit_rate_health_aware": float(np.mean(base_cost)) if base_cost else 0.0,
        "baseline_operator_hit_rate_health_aware": float(np.mean(base_op)) if base_op else 0.0,
        "automatic_pass_node_count": int((rec["recommendation_status"] == "pass").sum()),
        "manual_check_node_count": int((rec["recommendation_status"] == "needs_manual_check").sum()),
        "no_feasible_business_node_count": int((rec["recommendation_status"] == "no_feasible_business").sum()),
        "blocked_business_events": int(rec["blocked_business_count"].sum()),
        "regression": {
            "cost_mae": float(mean_absolute_error(actual_cost, predicted_cost)),
            "cost_rmse": float(math.sqrt(mean_squared_error(actual_cost, predicted_cost))),
            "cost_r2": float(r2_score(actual_cost, predicted_cost)) if np.ptp(actual_cost) > 0 else None,
            "profit_mae": float(mean_absolute_error(actual_profit, predicted_profit)),
            "profit_rmse": float(math.sqrt(mean_squared_error(actual_profit, predicted_profit))),
            "profit_r2": float(r2_score(actual_profit, predicted_profit)) if np.ptp(actual_profit) > 0 else None,
        },
        "model_minus_baseline_miner": bootstrap_difference(np.array(hit_cost), np.array(base_cost)),
        "model_minus_baseline_operator": bootstrap_difference(np.array(hit_op), np.array(base_op)),
    }
    signals = {
        "node_id": np.asarray(evaluation_node_ids, dtype=object),
        "miner_hit": np.asarray(hit_cost, dtype=bool),
        "operator_hit": np.asarray(hit_op, dtype=bool),
        "baseline_miner_hit": np.asarray(base_cost, dtype=bool),
        "baseline_operator_hit": np.asarray(base_op, dtype=bool),
    }
    return metrics, rec.reset_index(), signals


def prepare_frames(pairs: pd.DataFrame, outcomes: pd.DataFrame, nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = outcomes.copy()
    outcomes["node_id"] = outcomes["node_id"].astype(str)
    outcomes["business"] = outcomes["business"].astype(str)
    for column in ["cum_cost_7d", "cum_revenue_7d"]:
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce").fillna(0.0)
    outcomes["cum_profit_7d"] = outcomes["cum_revenue_7d"] - outcomes["cum_cost_7d"]
    nodes = add_derived_features(nodes)
    model_df = outcomes.merge(nodes, on="node_id", how="inner")
    model_df = model_df.dropna(subset=["node_id", "business"]).copy()
    return pairs, outcomes, model_df


def split_node_sets(node_ids: list[str], seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(node_ids), dtype=object)
    rng.shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * TRAIN_RATIO))
    return set(shuffled[:train_count]), set(shuffled[train_count:])


def best_business_by_node(frame: pd.DataFrame, target: str) -> pd.Series:
    best_indices = frame.groupby("node_id")[target].idxmax()
    return frame.loc[best_indices].set_index("node_id")["business"].astype(str)


def feature_importance_for_model(model: Recommender) -> dict[str, Any]:
    coefficient = np.asarray(model.op_model.coef_, dtype=float) if model.op_model is not None else np.array([])
    order = np.argsort(np.abs(coefficient))[::-1][:30]
    regression_importance = {
        model.encoder.output_names[index]: float(coefficient[index])
        for index in order
        if abs(coefficient[index]) > 0
    }
    classifier_importance: dict[str, dict[str, float]] = {}
    if model.business_classifier is not None:
        for target, classifier in [
            ("miner_cost_best", model.business_classifier.cost_classifier),
            ("operator_profit_best", model.business_classifier.profit_classifier),
        ]:
            if classifier is None:
                continue
            importances = np.asarray(classifier.feature_importances_, dtype=float)
            order = np.argsort(importances)[::-1][:30]
            classifier_importance[target] = {
                model.business_classifier.feature_names[index]: float(importances[index])
                for index in order
                if importances[index] > 0
            }
    return {
        "classifier": classifier_importance,
        "operator_profit_coefficient": regression_importance,
    }


def run_feature_set_evaluation(
    feature_set_name: str,
    train_all: pd.DataFrame,
    test_all: pd.DataFrame,
    train_nodes: set[str],
    test_nodes: set[str],
    businesses: list[str],
    derived_nodes: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    config = FEATURE_SET_CONFIGS[feature_set_name]
    train = train_all[train_all["business"].isin(businesses)].copy()
    test = test_all[test_all["business"].isin(businesses)].copy()
    miner_baseline_by_node = best_business_by_node(train, "cum_cost_7d")
    operator_baseline_by_node = best_business_by_node(train, "cum_profit_7d")
    baseline_miner = str(miner_baseline_by_node.mode().iloc[0])
    baseline_operator = str(operator_baseline_by_node.mode().iloc[0])
    model = Recommender(
        config["categorical"],
        config["model_numeric"],
        config["interaction"],
        random_seed=seed,
    ).fit(train)
    health_bounds = build_health_bounds(train)
    train_metrics, train_rec, train_signals = evaluate(
        "train_in_sample", model,
        derived_nodes[derived_nodes["node_id"].isin(train_nodes)], train,
        businesses, health_bounds, baseline_miner, baseline_operator,
    )
    test_metrics, test_rec, test_signals = evaluate(
        "test_node_holdout", model,
        derived_nodes[derived_nodes["node_id"].isin(test_nodes)], test,
        businesses, health_bounds, baseline_miner, baseline_operator,
    )
    all_rec = pd.concat([
        train_rec.assign(split="train_in_sample"),
        test_rec.assign(split="test_node_holdout"),
    ], ignore_index=True)
    return {
        "feature_set": feature_set_name,
        "feature_set_label": config["label"],
        "model": model,
        "health_bounds": health_bounds,
        "train": train_metrics,
        "test": test_metrics,
        "train_signals": train_signals,
        "test_signals": test_signals,
        "recommendations": all_rec,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_importance": feature_importance_for_model(model),
    }


def compare_signals(
    enriched: dict[str, np.ndarray],
    core: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    enriched_frame = pd.DataFrame({
        "node_id": enriched["node_id"].astype(str),
        "miner_hit": enriched["miner_hit"],
        "operator_hit": enriched["operator_hit"],
    }).set_index("node_id")
    core_frame = pd.DataFrame({
        "node_id": core["node_id"].astype(str),
        "miner_hit": core["miner_hit"],
        "operator_hit": core["operator_hit"],
    }).set_index("node_id")
    common = sorted(set(enriched_frame.index) & set(core_frame.index))
    if not common:
        return {
            "n_nodes": 0,
            "miner_enriched_minus_core": bootstrap_difference(np.array([]), np.array([]), seed),
            "operator_enriched_minus_core": bootstrap_difference(np.array([]), np.array([]), seed),
        }
    return {
        "n_nodes": len(common),
        "miner_enriched_minus_core": bootstrap_difference(
            enriched_frame.loc[common, "miner_hit"].to_numpy(),
            core_frame.loc[common, "miner_hit"].to_numpy(),
            seed,
        ),
        "operator_enriched_minus_core": bootstrap_difference(
            enriched_frame.loc[common, "operator_hit"].to_numpy(),
            core_frame.loc[common, "operator_hit"].to_numpy(),
            seed,
        ),
    }


def summarize_multiseed(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_names = [
        "miner_hit_rate_at_1",
        "operator_hit_rate_at_1",
        "baseline_miner_hit_rate_health_aware",
        "baseline_operator_hit_rate_health_aware",
        "miner_realized_regret",
        "operator_realized_regret",
    ]
    for feature_set_name in FEATURE_SET_CONFIGS:
        summary[feature_set_name] = {}
        for split in ["train", "test"]:
            summary[feature_set_name][split] = {}
            for metric_name in metric_names:
                values = [
                    result["feature_sets"][feature_set_name][split].get(metric_name)
                    for result in seed_results
                ]
                values = [float(value) for value in values if value is not None]
                summary[feature_set_name][split][metric_name] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
                    "min": float(np.min(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                }
    for split in ["train", "test"]:
        enriched_values = [
            result["feature_comparison"][split]["miner_enriched_minus_core"]["difference"]
            for result in seed_results
        ]
        operator_values = [
            result["feature_comparison"][split]["operator_enriched_minus_core"]["difference"]
            for result in seed_results
        ]
        summary.setdefault("enriched_vs_core", {})[split] = {
            "miner_difference_mean": float(np.mean(enriched_values)),
            "miner_difference_std": float(np.std(enriched_values, ddof=1)) if len(enriched_values) > 1 else 0.0,
            "miner_positive_seed_count": int(sum(value > 0 for value in enriched_values)),
            "operator_difference_mean": float(np.mean(operator_values)),
            "operator_difference_std": float(np.std(operator_values, ddof=1)) if len(operator_values) > 1 else 0.0,
            "operator_positive_seed_count": int(sum(value > 0 for value in operator_values)),
        }
    return summary


def train_and_evaluate(pairs: pd.DataFrame, outcomes: pd.DataFrame, nodes: pd.DataFrame, prom_probe: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_nodes = nodes.copy()
    _, outcomes, model_df = prepare_frames(pairs, outcomes, raw_nodes)
    derived_nodes = add_derived_features(raw_nodes)
    export_nodes = derived_nodes.copy()
    for column in CAT_FEATURES + NUM_FEATURES:
        if column in raw_nodes:
            export_nodes[column] = raw_nodes[column].to_numpy()
    export_nodes.to_csv(OUTPUT_NODES, index=False)
    node_ids = sorted(model_df["node_id"].astype(str).unique())
    seeds = tuple(dict.fromkeys((RANDOM_SEED, *VALIDATION_SEEDS)))
    seed_results: list[dict[str, Any]] = []
    primary_results: dict[str, dict[str, Any]] = {}

    for seed in seeds:
        train_nodes, test_nodes = split_node_sets(node_ids, seed)
        train_all = model_df[model_df["node_id"].isin(train_nodes)].copy()
        test_all = model_df[model_df["node_id"].isin(test_nodes)].copy()
        candidate_counts = train_all["business"].value_counts()
        businesses = sorted(
            candidate_counts[candidate_counts >= MIN_BUSINESS_SUPPORT].index.astype(str).tolist()
        )
        if not businesses:
            raise RuntimeError(f"随机种子 {seed} 的训练集没有达到最低支持数的候选业务。")
        print(
            f"validation seed={seed}: train_nodes={len(train_nodes):,}, "
            f"test_nodes={len(test_nodes):,}, businesses={len(businesses):,}"
        )
        feature_sets: dict[str, dict[str, Any]] = {}
        for feature_set_name in FEATURE_SET_CONFIGS:
            print(f"fit feature set={feature_set_name}, seed={seed}")
            result = run_feature_set_evaluation(
                feature_set_name,
                train_all,
                test_all,
                train_nodes,
                test_nodes,
                businesses,
                derived_nodes,
                seed,
            )
            feature_sets[feature_set_name] = result
        seed_result = {
            "seed": seed,
            "train_nodes": len(train_nodes),
            "test_nodes": len(test_nodes),
            "train_rows_all": len(train_all),
            "test_rows_all": len(test_all),
            "candidate_businesses": len(businesses),
            "businesses": businesses,
            "feature_sets": {
                name: {
                    "train": result["train"],
                    "test": result["test"],
                }
                for name, result in feature_sets.items()
            },
            "feature_comparison": {
                "train": compare_signals(
                    feature_sets["enriched_pre_day"]["train_signals"],
                    feature_sets["core_pre_day"]["train_signals"],
                    seed,
                ),
                "test": compare_signals(
                    feature_sets["enriched_pre_day"]["test_signals"],
                    feature_sets["core_pre_day"]["test_signals"],
                    seed,
                ),
            },
        }
        seed_results.append(seed_result)
        if seed == RANDOM_SEED:
            primary_results = feature_sets
            primary_train_nodes = train_nodes
            primary_test_nodes = test_nodes

    primary_core = primary_results["core_pre_day"]
    primary_enriched = primary_results["enriched_pre_day"]
    primary_comparison = {
        "seed": RANDOM_SEED,
        "train": {
            "core": primary_core["train"],
            "enriched": primary_enriched["train"],
            **compare_signals(
                primary_enriched["train_signals"],
                primary_core["train_signals"],
                RANDOM_SEED,
            ),
        },
        "test": {
            "core": primary_core["test"],
            "enriched": primary_enriched["test"],
            **compare_signals(
                primary_enriched["test_signals"],
                primary_core["test_signals"],
                RANDOM_SEED,
            ),
        },
    }
    with open(OUTPUT_FEATURE_COMPARISON, "w", encoding="utf-8") as handle:
        json.dump(primary_comparison, handle, ensure_ascii=False, indent=2, default=str)

    multiseed = {
        "seeds": list(seeds),
        "primary_seed": RANDOM_SEED,
        "per_seed": seed_results,
        "summary": summarize_multiseed(seed_results),
    }
    with open(OUTPUT_MULTISEED, "w", encoding="utf-8") as handle:
        json.dump(multiseed, handle, ensure_ascii=False, indent=2, default=str)

    primary_enriched["recommendations"].to_csv(OUTPUT_RECOMMENDATIONS, index=False)
    primary_core["recommendations"].to_csv(OUTPUT_BASELINE_RECOMMENDATIONS, index=False)

    diagnostics = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "wide_table": "test.node_day_ops_wide_full",
        "database": "yzh-starrocks",
        "history_start": START_DAY.isoformat(),
        "history_end": END_DAY.isoformat(),
        "sample_suffixes": list(SUFFIXES),
        "node_buckets": list(NODE_BUCKETS),
        "sample_rate": SAMPLE_RATE,
        "eligibility": {
            "min_business_count": MIN_BUSINESS_COUNT,
            "min_distinct_business_first_days": MIN_BUSINESS_COUNT,
            "required_ever_state": "online",
            "business_filter": "customerId > 0",
            "last_full_window_business_day": (END_DAY - dt.timedelta(days=6)).isoformat(),
        },
        "outcome_window": "[business_online_day, business_online_day + 6]",
        "attribute_lookback_days": ATTRIBUTE_LOOKBACK_DAYS,
        "attribute_snapshot_mode": (
            f"online_day_minus_{ATTRIBUTE_LOOKBACK_DAYS}d"
            if ATTRIBUTE_LOOKBACK_DAYS > 0 else "online_day"
        ),
        "pair_rows": int(len(pairs)),
        "outcome_rows": int(len(outcomes)),
        "nodes_with_pairs": int(pairs["node_id"].nunique()),
        "model_nodes": int(len(node_ids)),
        "businesses_total": int(outcomes["business"].nunique()),
        "train_nodes": len(primary_train_nodes),
        "test_nodes": len(primary_test_nodes),
        "train_rows": primary_enriched["train_rows"],
        "test_rows": primary_enriched["test_rows"],
        "candidate_businesses": len(seed_results[0]["businesses"]),
        "candidate_business_support_min": MIN_BUSINESS_SUPPORT,
        "validation_seeds": list(seeds),
        "feature_sets": {
            name: {
                "label": config["label"],
                "features_categorical": config["categorical"],
                "features_numeric": config["numeric"],
                "model_features_numeric": config["model_numeric"],
                "interaction_features": config["interaction"],
            }
            for name, config in FEATURE_SET_CONFIGS.items()
        },
        # 这里的 features_* 表示实际允许进入收益模型的节点固有属性；
        # collected_* 才表示节点 CSV 中完整保留的属性和动态健康观测。
        "features_categorical": NODE_INTRINSIC_CAT_FEATURES,
        "features_numeric": NODE_INTRINSIC_NUM_FEATURES,
        "collected_attribute_categorical": CAT_FEATURES,
        "collected_attribute_numeric": NUM_FEATURES,
        "model_features_numeric": MODEL_NUM_FEATURES,
        "interaction_features": NODE_INTRINSIC_INTERACTION_FEATURES,
        "model_interaction_features": MODEL_INTERACTION_FEATURES,
        "vm_runtime_only_features": sorted(VM_RUNTIME_ONLY_FEATURES),
        "health_metrics_in_contract": HEALTH_METRICS,
        "node_intrinsic_feature_policy": (
            "模型只使用上线前可确定的节点配置、地域、设备、硬件容量和名义网络配置；"
            "上线后状态、利用率、质量、在线时长和业务流量只用于健康准入。"
        ),
        "health_manual_only_metrics": sorted(HEALTH_MANUAL_ONLY_METRICS),
        "attribute_snapshot_modes_observed": sorted(derived_nodes["attribute_snapshot_mode"].dropna().astype(str).unique()) if "attribute_snapshot_mode" in derived_nodes else [],
        "excluded_unreadable_node_analysis_partitions": [
            {"day": day, "hour": hour}
            for day, hour in sorted(UNREADABLE_NODE_ANALYSIS_PARTITIONS)
        ],
        "prometheus_probe": prom_probe,
        "attribute_missing_cells": int(derived_nodes[CAT_FEATURES + NUM_FEATURES].isna().sum().sum()),
        "attribute_coverage": {
            column: float(derived_nodes[column].notna().mean()) if column in derived_nodes else 0.0
            for column in [
                "cpuload1", "join_cpu_load1", "memtotal", "join_memtotal",
                "totaldisksize", "join_disks_total_size", "retrans",
                "join_net_tcpretransrate", "join_net_pinglossrate",
                "smart_available_spare", "smart_bad_block_count",
                "smart_case_temperature", "smart_critical_warning",
                "zfs_pool_online", "zfs_dataset_read", "zfs_dataset_write",
                "prom_retrans_ratio",
            ]
        },
        "zero_cost_rows": int((outcomes["cum_cost_7d"] == 0).sum()),
        "zero_revenue_rows": int((outcomes["cum_revenue_7d"] == 0).sum()),
        "invalid_negative_value_counts": {
            column: int((pd.to_numeric(nodes[column], errors="coerce") < 0).sum())
            for column in [
                "retrans", "v4pingloss", "v6pingloss", "quality_retransrate",
                "quality_pinglossrate", "join_net_tcpretransrate",
                "join_net_pinglossrate", "join_net_pingrtt",
            ]
            if column in raw_nodes
        },
        "nonpositive_capacity_counts": {
            column: int((pd.to_numeric(nodes[column], errors="coerce") <= 0).sum())
            for column in [
                "corenum", "memtotal", "totaldisksize", "hdddisksize",
                "ssddisksize", "systemdisksize",
            ]
            if column in raw_nodes
        },
        "outcome_distinct_days_distribution": {
            "min": int(outcomes["outcome_distinct_days"].min()) if "outcome_distinct_days" in outcomes else None,
            "median": float(outcomes["outcome_distinct_days"].median()) if "outcome_distinct_days" in outcomes else None,
            "max": int(outcomes["outcome_distinct_days"].max()) if "outcome_distinct_days" in outcomes else None,
            "less_than_7": int((outcomes["outcome_distinct_days"] < 7).sum()) if "outcome_distinct_days" in outcomes else None,
        },
        "baseline_business_miner": str(primary_enriched["train"]["baseline_business_miner"]),
        "baseline_business_operator": str(primary_enriched["train"]["baseline_business_operator"]),
        "health_rules": GLOBAL_HEALTH_RULES,
        "health_rule_note": "硬阈值为初始可审计规则；业务历史可行域来自训练集1%/99%分位数，不代表因果或显著性验证。SMART温度与ZFS数据集读写没有可审计阈值，缺失或有值均需人工审核，不能自动放行。",
        "model": "推荐模型：ExtraTrees 多分类器直接预测节点历史成本最佳业务和运营利润最佳业务；稀疏交互 Ridge 预测金额用于辅助诊断和 CSV 数值输出",
        "recommendation_model": {
            "algorithm": "ExtraTreesClassifier",
            "n_estimators": int(os.environ.get("MULTIBUSINESS_TREE_ESTIMATORS", "200")),
            "max_depth": int(os.environ.get("MULTIBUSINESS_TREE_MAX_DEPTH", "18")),
            "min_samples_leaf": int(os.environ.get("MULTIBUSINESS_TREE_MIN_SAMPLES_LEAF", "5")),
            "max_features": os.environ.get("MULTIBUSINESS_TREE_MAX_FEATURES", "sqrt"),
            "target_cost": "节点训练历史中 cum_cost_7d 最大的业务",
            "target_operator": "节点训练历史中 cum_profit_7d 最大的业务",
        },
        "auxiliary_regression_model": f"稀疏交互 Ridge(alpha={float(os.environ.get('MULTIBUSINESS_RIDGE_ALPHA', '10.0'))}, solver='lsqr', target_transform={primary_enriched['model'].target_transform})",
        "feature_importance_classifier": primary_enriched["feature_importance"]["classifier"],
        "feature_importance_operator_profit_coefficient": primary_enriched["feature_importance"]["operator_profit_coefficient"],
    }
    metrics = {
        "train": primary_enriched["train"],
        "test": primary_enriched["test"],
        "core_baseline": {
            "train": primary_core["train"],
            "test": primary_core["test"],
        },
        "feature_comparison": primary_comparison,
        "multi_seed_validation": multiseed,
        "diagnostics": diagnostics,
    }
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    return metrics, primary_enriched["recommendations"]


def build_report(metrics: dict[str, Any]) -> None:
    diagnostics = metrics["diagnostics"]
    train = metrics["train"]
    test = metrics["test"]
    core = metrics["core_baseline"]
    comparison = metrics["feature_comparison"]
    multiseed = metrics["multi_seed_validation"]

    def pct(value: Any) -> str:
        return "无" if value is None else f"{float(value):.2%}"

    def number(value: Any) -> str:
        return "无" if value is None else f"{float(value):.6g}"

    miner_effect = test["model_minus_baseline_miner"]
    operator_effect = test["model_minus_baseline_operator"]
    baseline_effects_positive = (
        miner_effect["difference"] > 0
        and operator_effect["difference"] > 0
    )
    baseline_effects_significant = (
        (
            miner_effect["ci95_low"] > 0
            or miner_effect["ci95_high"] < 0
        )
        and (
            operator_effect["ci95_low"] > 0
            or operator_effect["ci95_high"] < 0
        )
    )
    feature_miner_effect = comparison["test"]["miner_enriched_minus_core"]
    feature_operator_effect = comparison["test"]["operator_enriched_minus_core"]
    feature_gain_positive = (
        feature_miner_effect["difference"] > 0
        and feature_operator_effect["difference"] > 0
    )
    feature_gain_significant = (
        feature_miner_effect["ci95_low"] > 0
        and feature_operator_effect["ci95_low"] > 0
    )
    if feature_gain_positive and feature_gain_significant:
        feature_conclusion = "增强字段相对核心基线两项目标均有显著提升。"
    elif feature_gain_positive:
        feature_conclusion = "增强字段相对核心基线呈正向提升，但至少一项95% CI跨0，暂不能认定新增字段显著有效。"
    else:
        feature_conclusion = "增强字段相对核心基线至少一项目标没有正向提升，暂不能认定新增字段有效。"
    if baseline_effects_positive and baseline_effects_significant:
        baseline_conclusion = "增强版相对固定健康约束基线的两项提升均显著。"
    elif baseline_effects_positive:
        baseline_conclusion = "增强版相对固定健康约束基线呈正向趋势，但显著性证据不足。"
    else:
        baseline_conclusion = "增强版相对固定健康约束基线没有同时呈现正向提升。"

    seed_rows = []
    for result in multiseed["per_seed"]:
        core_test = result["feature_sets"]["core_pre_day"]["test"]
        enriched_test = result["feature_sets"]["enriched_pre_day"]["test"]
        seed_rows.append(
            "| {seed} | {core_miner} | {enriched_miner} | {core_op} | {enriched_op} | {miner_diff} | {op_diff} |".format(
                seed=result["seed"],
                core_miner=pct(core_test["miner_hit_rate_at_1"]),
                enriched_miner=pct(enriched_test["miner_hit_rate_at_1"]),
                core_op=pct(core_test["operator_hit_rate_at_1"]),
                enriched_op=pct(enriched_test["operator_hit_rate_at_1"]),
                miner_diff=pct(result["feature_comparison"]["test"]["miner_enriched_minus_core"]["difference"]),
                op_diff=pct(result["feature_comparison"]["test"]["operator_enriched_minus_core"]["difference"]),
            )
        )
    extra_categories = [
        column for column in diagnostics["features_categorical"]
        if column not in diagnostics["feature_sets"]["core_pre_day"]["features_categorical"]
    ]
    extra_numeric = [
        column for column in diagnostics["features_numeric"]
        if column not in diagnostics["feature_sets"]["core_pre_day"]["features_numeric"]
    ]
    lines = [
        "# 跨日期多业务节点最佳业务推荐重建报告",
        "",
        "## 结论",
        "",
        f"本次使用宽表符合资格节点 {diagnostics['nodes_with_pairs']:,} 个，其中可与属性和目标成功合并并进入模型的节点 {diagnostics['model_nodes']:,} 个；主随机种子 {comparison['seed']} 按节点切分训练 {diagnostics['train_nodes']:,} 个、测试 {diagnostics['test_nodes']:,} 个。",
        f"增强版测试集矿主最佳业务命中率@1为 {pct(test['miner_hit_rate_at_1'])}，运营最佳业务命中率@1为 {pct(test['operator_hit_rate_at_1'])}；同一健康约束下固定基线分别为 {pct(test['baseline_miner_hit_rate_health_aware'])}、{pct(test['baseline_operator_hit_rate_health_aware'])}。",
        f"相对健康约束基线，矿主差值 {pct(miner_effect['difference'])}，p={miner_effect['p_value']:.4g}，95% CI [{pct(miner_effect['ci95_low'])}, {pct(miner_effect['ci95_high'])}]；运营差值 {pct(operator_effect['difference'])}，p={operator_effect['p_value']:.4g}，95% CI [{pct(operator_effect['ci95_low'])}, {pct(operator_effect['ci95_high'])}]。{baseline_conclusion}",
        f"增强版相对核心字段基线，测试集矿主差值 {pct(feature_miner_effect['difference'])}，95% CI [{pct(feature_miner_effect['ci95_low'])}, {pct(feature_miner_effect['ci95_high'])}]；运营差值 {pct(feature_operator_effect['difference'])}，95% CI [{pct(feature_operator_effect['ci95_low'])}, {pct(feature_operator_effect['ci95_high'])}]。{feature_conclusion}",
        f"5个节点级随机种子中，增强版相对核心基线矿主正向的种子数为 {multiseed['summary']['enriched_vs_core']['test']['miner_positive_seed_count']}/{len(multiseed['seeds'])}，运营正向的种子数为 {multiseed['summary']['enriched_vs_core']['test']['operator_positive_seed_count']}/{len(multiseed['seeds'])}。上述是预测回测显著性，不是因果效果或线上 A/B 结论。",
        "",
        "## 数据口径",
        "",
        f"- 宽表：`{diagnostics['wide_table']}`，历史范围 `{diagnostics['history_start']}` 至 `{diagnostics['history_end']}`。",
        f"- 筛选：`customerId > 0`，至少 {diagnostics['eligibility']['min_business_count']} 个不同业务、业务首次出现日至少 {diagnostics['eligibility']['min_distinct_business_first_days']} 个不同日期，且至少一条记录 `state='online'`；业务首次出现日不晚于 `{diagnostics['eligibility']['last_full_window_business_day']}`，保证7天窗口可计算。",
        f"- 样本：节点 ID 后缀 `{','.join(diagnostics['sample_suffixes'])}`，约 {diagnostics['sample_rate']:.2%}；不是全量节点推断。",
        f"- 目标窗口：每个节点-业务从该业务首次出现日开始的 7 天，成本为 `cum_cost_7d`，运营目标为 `cum_revenue_7d - cum_cost_7d`。",
        f"- 窗口观测日数：最少 {diagnostics['outcome_distinct_days_distribution']['min']} 天，中位数 {diagnostics['outcome_distinct_days_distribution']['median']:.0f} 天，最多 {diagnostics['outcome_distinct_days_distribution']['max']} 天；其中 {diagnostics['outcome_distinct_days_distribution']['less_than_7']:,} 行少于 7 个观测日，缺失日按宽表没有记录处理为目标累计中的零贡献，不能解释为真实业务运行状态。",
        f"- 原始属性异常：无法读取的 `node_analysis_data` 分区为 {diagnostics['excluded_unreadable_node_analysis_partitions'] or '无'}；这些小时被排除，没有进行数值补填，节点若没有其他可用小时则对应属性保持缺失。",
        f"- 模型：{diagnostics['model'].rstrip('。')}。训练集为拟合内评估，测试集为节点级留出评估。节点属性严格取 `online_day - 1 day` 的最新可用小时，避免把业务启动后的状态泄漏进7天目标。",
        f"- 推荐分类目标：矿主为训练节点历史 `cum_cost_7d` 最大业务，运营为训练节点历史 `cum_profit_7d` 最大业务；`miner_model_probability`/`operator_model_probability` 为分类概率，`miner_pred_cost`/`operator_pred_profit` 为 Ridge 辅助金额预测。",
        f"- 金额辅助回归测试集 R²：成本 {number(test['regression']['cost_r2'])}，利润 {number(test['regression']['profit_r2'])}；两者接近或低于0，说明当前数据和 Ridge 不能证明金额预测有效，最终业务推荐以最佳业务分类命中率为主。",
        f"- 原始结果 {diagnostics['outcome_rows']:,} 行，业务数 {diagnostics['businesses_total']:,}；候选业务 {diagnostics['candidate_businesses']:,} 个，训练支持阈值为 {diagnostics['candidate_business_support_min']} 行。",
        f"- 特征边界：节点 CSV 共保留 {len(diagnostics['collected_attribute_categorical'])} 个类别属性和 {len(diagnostics['collected_attribute_numeric'])} 个数值/健康观测字段；实际收益模型只使用 {len(diagnostics['features_categorical'])} 个节点固有类别属性、{len(diagnostics['model_features_numeric'])} 个节点固有数值属性及其缺失标记。",
        "- 严格排除：历史在线时长、前一天在线率、带宽利用率、运行中 CPU/内存/磁盘使用率、RTT、丢包、重传、SMART、ZFS 和业务流量等上线后观测字段；它们只用于健康准入或人工审核。",
        "",
        "## 使用字段",
        "",
        "核心基线类别字段：" + "、".join(diagnostics["feature_sets"]["core_pre_day"]["features_categorical"]) + "。",
        "",
        "增强版新增类别字段：" + "、".join(extra_categories) + "。",
        "",
        "核心基线数值字段：" + "、".join(diagnostics["feature_sets"]["core_pre_day"]["features_numeric"]) + "。",
        "",
        "增强版新增数值字段：" + "、".join(extra_numeric) + "。",
        "",
        "增强版历史收益模型实际使用数值字段：" + "、".join(diagnostics["model_features_numeric"]) + "。",
        "",
        "完整健康契约：CPU负载/核、CPU使用分解、内存总量/使用量/使用率、总盘/HDD/SSD/系统盘容量和使用率、磁盘IO利用率、node_analysis 与 node_join 重传/丢包/RTT、SMART spare/坏块/温度/critical warning、ZFS池在线状态及读写、Prometheus 节点重传。完整字段名和每个字段的来源保存在 `multibusiness_metrics.json` 与 `multibusiness_nodes.csv`；健康契约不等于模型输入字段。",
        "",
        "SMART、ZFS、Prometheus重传以及所有上线后动态观测均未从采集契约中删除，但不进入历史收益模型。当前 VM 接口只能返回运行时快照，无法按每个节点上线日前一天重建历史值；这些字段只用于上线前健康准入。若硬阈值不合格，业务直接阻断；指标缺失则不判定为合格，转人工检查。SMART温度与ZFS数据集读写没有可审计阈值，缺失或有值均只允许人工审核，不能自动放行。",
        "",
        "## 训练与测试效果",
        "",
        "主随机种子下核心基线与增强版：",
        "",
        "| 集合 | 特征集 | 节点数 | 行数 | 矿主命中率@1 | 运营命中率@1 | 矿主覆盖率 | 运营覆盖率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| 训练集拟合内 | 核心基线 | {core['train']['n_nodes']:,} | {diagnostics['train_rows']:,} | {pct(core['train']['miner_hit_rate_at_1'])} | {pct(core['train']['operator_hit_rate_at_1'])} | {pct(core['train']['miner_realized_coverage'])} | {pct(core['train']['operator_realized_coverage'])} |",
        f"| 训练集拟合内 | 增强版 | {train['n_nodes']:,} | {diagnostics['train_rows']:,} | {pct(train['miner_hit_rate_at_1'])} | {pct(train['operator_hit_rate_at_1'])} | {pct(train['miner_realized_coverage'])} | {pct(train['operator_realized_coverage'])} |",
        f"| 测试集节点留出 | 核心基线 | {core['test']['n_nodes']:,} | {core['test']['test_rows'] if 'test_rows' in core['test'] else diagnostics['test_rows']:,} | {pct(core['test']['miner_hit_rate_at_1'])} | {pct(core['test']['operator_hit_rate_at_1'])} | {pct(core['test']['miner_realized_coverage'])} | {pct(core['test']['operator_realized_coverage'])} |",
        f"| 测试集节点留出 | 增强版 | {test['n_nodes']:,} | {diagnostics['test_rows']:,} | {pct(test['miner_hit_rate_at_1'])} | {pct(test['operator_hit_rate_at_1'])} | {pct(test['miner_realized_coverage'])} | {pct(test['operator_realized_coverage'])} |",
        "",
        "| 集合 | 成本MAE | 成本RMSE | 成本R² | 利润MAE | 利润RMSE | 利润R² | 自动通过节点 | 需人工检查节点 | 无可行业务节点 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 训练集拟合内 | 核心基线 | {core['train']['regression']['cost_mae']:.6g} | {core['train']['regression']['cost_rmse']:.6g} | {number(core['train']['regression']['cost_r2'])} | {core['train']['regression']['profit_mae']:.6g} | {core['train']['regression']['profit_rmse']:.6g} | {number(core['train']['regression']['profit_r2'])} | {core['train']['automatic_pass_node_count']:,} | {core['train']['manual_check_node_count']:,} | {core['train']['no_feasible_business_node_count']:,} |",
        f"| 训练集拟合内 | 增强版 | {train['regression']['cost_mae']:.6g} | {train['regression']['cost_rmse']:.6g} | {number(train['regression']['cost_r2'])} | {train['regression']['profit_mae']:.6g} | {train['regression']['profit_rmse']:.6g} | {number(train['regression']['profit_r2'])} | {train['automatic_pass_node_count']:,} | {train['manual_check_node_count']:,} | {train['no_feasible_business_node_count']:,} |",
        f"| 测试集节点留出 | 核心基线 | {core['test']['regression']['cost_mae']:.6g} | {core['test']['regression']['cost_rmse']:.6g} | {number(core['test']['regression']['cost_r2'])} | {core['test']['regression']['profit_mae']:.6g} | {core['test']['regression']['profit_rmse']:.6g} | {number(core['test']['regression']['profit_r2'])} | {core['test']['automatic_pass_node_count']:,} | {core['test']['manual_check_node_count']:,} | {core['test']['no_feasible_business_node_count']:,} |",
        f"| 测试集节点留出 | 增强版 | {test['regression']['cost_mae']:.6g} | {test['regression']['cost_rmse']:.6g} | {number(test['regression']['cost_r2'])} | {test['regression']['profit_mae']:.6g} | {test['regression']['profit_rmse']:.6g} | {number(test['regression']['profit_r2'])} | {test['automatic_pass_node_count']:,} | {test['manual_check_node_count']:,} | {test['no_feasible_business_node_count']:,} |",
        "",
        f"训练集矿主基线业务为 `{diagnostics['baseline_business_miner']}`，运营基线业务为 `{diagnostics['baseline_business_operator']}`。",
        "覆盖率只表示临时排序业务在该节点历史结果中实际出现过，遗憾和价值捕获率仅在可观测业务上计算；这不是对从未运行过业务的因果效果估计。",
        "",
        "## 多随机种子",
        "",
        "| seed | 核心矿主命中 | 增强矿主命中 | 核心运营命中 | 增强运营命中 | 矿主增强-核心 | 运营增强-核心 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *seed_rows,
        "",
        f"测试集跨种子平均差值：矿主 {pct(multiseed['summary']['enriched_vs_core']['test']['miner_difference_mean'])}，标准差 {pct(multiseed['summary']['enriched_vs_core']['test']['miner_difference_std'])}；运营 {pct(multiseed['summary']['enriched_vs_core']['test']['operator_difference_mean'])}，标准差 {pct(multiseed['summary']['enriched_vs_core']['test']['operator_difference_std'])}。",
        "",
        "## 健康约束",
        "",
        "CPU、内存、磁盘、SMART、ZFS、重传都纳入字段契约。硬阈值见 `multibusiness_metrics.json`；业务特定可行域由训练集该业务历史节点的1%/99%分位数生成。由于没有业务方提供的业务-资源要求映射，这些规则是可追溯的初始准入规则，不是已经因果验证的标准。硬性不合格会阻断对应业务；指标缺失不判定为合格，进入人工检查；SMART温度与ZFS数据集读写即使有值也不自动放行。",
        "",
        f"Prometheus 直接按 `node_id` 探查结果为 `mapping_usable={diagnostics['prometheus_probe']['mapping_usable']}`；运行时覆盖率：SMART spare {diagnostics['attribute_coverage'].get('smart_available_spare', 0):.2%}、坏块 {diagnostics['attribute_coverage'].get('smart_bad_block_count', 0):.2%}、温度 {diagnostics['attribute_coverage'].get('smart_case_temperature', 0):.2%}、critical warning {diagnostics['attribute_coverage'].get('smart_critical_warning', 0):.2%}、Prometheus重传 {diagnostics['attribute_coverage'].get('prom_retrans_ratio', 0):.2%}、ZFS池 {diagnostics['attribute_coverage'].get('zfs_pool_online', 0):.2%}。当前自动通过节点为训练 {train['automatic_pass_node_count']:,}、测试 {test['automatic_pass_node_count']:,}；`miner_best`/`operator_best` 只有自动通过时才填值，需人工检查的 provisional 推荐不能直接下发。",
        f"非正容量/核数异常计数：{diagnostics['nonpositive_capacity_counts']}；负值哨兵清洗后计数：{diagnostics['invalid_negative_value_counts']}。这些值在模型派生特征中按缺失处理；节点 CSV 保留对应原始字段列和缺失状态。",
        "",
        "## 文件",
        "",
        f"- `{OUTPUT_PAIRS.name}`：符合资格的节点-业务首次日期明细。",
        f"- `{OUTPUT_OUTCOMES.name}`：节点-业务 7 天成本、收入、利润结果。",
        f"- `{OUTPUT_NODES.name}`：node_analysis_data 与 node_join 属性快照及缺失值。",
        f"- `{OUTPUT_PROM.name}`：Prometheus 指标映射探查结果。",
        f"- `{OUTPUT_RECOMMENDATIONS.name}`：训练集和测试集推荐结果及健康阻断理由。",
        f"- `{OUTPUT_BASELINE_RECOMMENDATIONS.name}`：同一主随机种子下核心字段基线推荐结果。",
        f"- `{OUTPUT_FEATURE_COMPARISON.name}`：核心基线与增强版的主随机种子配对比较。",
        f"- `{OUTPUT_MULTISEED.name}`：多随机种子节点级留出验证结果。",
        f"- `{OUTPUT_METRICS.name}`：机器可读指标、字段和规则。",
    ]
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_REPORT}")


def main() -> None:
    started = time.time()
    pairs, outcomes = read_or_fetch_outcomes()
    node_ids = sorted(set(pairs["node_id"].astype(str)) & set(outcomes["node_id"].astype(str)))
    node_targets = (
        pairs[["node_id", "online_day"]]
        .assign(node_id=lambda frame: frame["node_id"].astype(str))
        .drop_duplicates("node_id")
    )
    cached_nodes = pd.read_csv(OUTPUT_NODES, low_memory=False) if OUTPUT_NODES.exists() else None
    expected_attribute_snapshot_mode = (
        f"online_day_minus_{ATTRIBUTE_LOOKBACK_DAYS}d"
        if ATTRIBUTE_LOOKBACK_DAYS > 0 else "online_day"
    )
    cache_is_attribute_aligned = (
        cached_nodes is not None
        and "attribute_snapshot_mode" in cached_nodes.columns
        and set(cached_nodes["attribute_snapshot_mode"].dropna().astype(str)) == {expected_attribute_snapshot_mode}
        and "attribute_day" in cached_nodes.columns
        and len(cached_nodes) == len(node_targets)
        and set(cached_nodes["node_id"].astype(str)) == set(node_targets["node_id"])
    )
    nodes = (
        cached_nodes
        if cache_is_attribute_aligned and os.environ.get("MULTIBUSINESS_REFRESH") != "1"
        else fetch_attributes(node_targets)
    )
    prom_probe = json.loads(OUTPUT_PROM.read_text(encoding="utf-8")) if OUTPUT_PROM.exists() and os.environ.get("MULTIBUSINESS_REFRESH") != "1" else probe_prometheus(node_ids)
    prom_columns = [
        "smart_available_spare", "smart_bad_block_count", "smart_case_temperature",
        "smart_critical_warning", "zfs_pool_online", "zfs_dataset_read",
        "zfs_dataset_write", "prom_retrans_ratio",
    ]
    prom_has_values = any(
        column in nodes.columns and nodes[column].notna().any()
        for column in prom_columns
    )
    if not prom_has_values or os.environ.get("MULTIBUSINESS_PROM_REFRESH") == "1":
        nodes = enrich_nodes_with_prometheus(nodes, node_ids)
    metrics, _ = train_and_evaluate(pairs, outcomes, nodes, prom_probe)
    metrics["diagnostics"]["elapsed_sec"] = round(time.time() - started, 1)
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=str)
    build_report(metrics)


if __name__ == "__main__":
    main()
