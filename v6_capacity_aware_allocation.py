#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build business-capacity facts and capacity-aware node allocation plans.

V5 estimates node/business fit. V6 keeps that score and adds a separate demand
capacity layer so a batch plan cannot send unlimited build bandwidth to one
business. The output is planning-only and never changes a live assignment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_v3_daily_business_training as v3
import v1_recommendation_pipeline as v1
import v2_ranking_model as v2


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = HERE / "recent_month_large_1d"
DEFAULT_OUTPUT_DIR = HERE / "recent_month_large_mainstream_v6_capacity"
DEFAULT_ALLOWLIST = HERE / "mainstream_business_allowlist.csv"
DEFAULT_BUSINESS_MAP = HERE / "v1_business_name_map_enriched.csv"
DEFAULT_RECOMMENDATIONS = (
    HERE / "recent_month_large_mainstream_v5_hybrid" / "v5_node_recommendations.csv"
)
OUTPUT_FACTS = "v6_node_day_capacity_facts.csv"
OUTPUT_DAILY = "v6_business_capacity_daily.csv"
OUTPUT_POOLS = "v6_capacity_pool_daily.csv"
OUTPUT_POOL_SUMMARY = "v6_capacity_pool_summary.csv"
OUTPUT_SUMMARY = "v6_business_capacity_summary.csv"
OUTPUT_CORRELATIONS = "v6_business_capacity_correlations.csv"
OUTPUT_ALLOCATIONS = "v6_capacity_aware_node_allocations.csv"
OUTPUT_NODE_REPORT = "v6_capacity_node_report.csv"
OUTPUT_REVIEW_NODES = "v6_capacity_review_nodes.csv"
OUTPUT_BLOCKED_NODES = "v6_capacity_blocked_nodes.csv"
OUTPUT_NODE_JSON = "v6_capacity_node_report.json"
OUTPUT_NODE_DATA = "v6_node_report_data.js"
OUTPUT_NODE_HTML = "v6_node_report.html"
OUTPUT_ALLOCATION_SUMMARY = "v6_capacity_allocation_summary.csv"
OUTPUT_JSON = "v6_capacity_summary.json"
OUTPUT_DATA = "v6_capacity_report_data.js"
OUTPUT_HTML = "v6_capacity_report.html"

POOL_LEVELS = {
    "business": [],
    "business_isp": ["daily_isp"],
    "business_isp_schedule": [
        "daily_isp", "network_schedule_type", "schedule_target_isp",
    ],
    "business_province_isp_schedule": [
        "daily_province", "daily_isp", "network_schedule_type",
        "schedule_target_isp",
    ],
}
POOL_LEVEL_ORDER = tuple(POOL_LEVELS)
POOL_LEVEL_LABELS = {
    "business": "业务整体",
    "business_isp": "业务 + 原运营商",
    "business_isp_schedule": "业务 + 原运营商 + 调度 + 目标运营商",
    "business_province_isp_schedule": "业务 + 省份 + 原运营商 + 调度 + 目标运营商",
}
POOL_NODE_FIELDS = {
    "business": [],
    "business_isp": ["isp"],
    "business_isp_schedule": [
        "isp", "network_schedule_type", "schedule_target_isp",
    ],
    "business_province_isp_schedule": [
        "province", "isp", "network_schedule_type", "schedule_target_isp",
    ],
}
CONFIDENCE_ORDER = {"high": 2, "medium": 1, "low": 0}


def clean_id(value: Any) -> str:
    text = v1.clean_cell(value)
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, set[str]], Path]:
    allowlist = v3.load_allowlist(args.allowlist)
    names = v3.load_business_names(args.business_map)
    binding_path = args.virtual_bindings or v3.latest_virtual_binding_path()
    bindings = v3.load_virtual_bindings(binding_path)
    if args.raw_input:
        raw_path = args.raw_input
        raw = pd.read_csv(
            raw_path, dtype={"nodeId": "string", "customerId": "string"}, low_memory=False,
        )
    else:
        metadata = json.loads(args.candidate_json.read_text(encoding="utf-8"))
        candidate_ids = [clean_id(value) for value in metadata["node_ids"]]
        raw_path = args.output_dir / f"node_day_business_capacity_raw_{args.start_day.replace('-', '')}_{args.end_day.replace('-', '')}.csv"
        raw = v3.fetch_raw_rows(
            candidate_ids, args.start_day, args.end_day, raw_path,
            args.chunk_size, args.workers, args.refresh,
        )
    return raw, allowlist, names, bindings, raw_path


def prepare_capacity_facts(
    raw: pd.DataFrame,
    allowlist: pd.DataFrame,
    names: dict[str, str],
    bindings: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    facts, audit, _ = v3.build_daily_facts(raw, allowlist, names, bindings)
    if facts.empty:
        return facts, audit
    output = facts.copy()
    output["business"] = output["business"].map(clean_id)
    output["buildBandwidth"] = pd.to_numeric(output["buildBandwidth"], errors="coerce")
    output["capacity_peak95_mbps"] = pd.to_numeric(
        output["capacity_peak95_mbps"], errors="coerce"
    )
    output["miner_income"] = pd.to_numeric(output["cum_cost_7d"], errors="coerce")
    output["platform_revenue"] = pd.to_numeric(output["cum_revenue_7d"], errors="coerce")
    output["platform_profit"] = output["platform_revenue"] - output["miner_income"]
    output["network_schedule_type"] = output.apply(
        lambda row: v2.infer_network_schedule_type({
            "isp": row.get("daily_isp"),
            "scheduleisps": row.get("daily_scheduleisps"),
            "analysis_transprovrate": row.get("daily_transprovrate"),
        }),
        axis=1,
    )
    output["schedule_target_isp"] = output.apply(
        lambda row: v2.primary_schedule_isp({
            "isp": row.get("daily_isp"),
            "scheduleisps": row.get("daily_scheduleisps"),
        }),
        axis=1,
    )
    output["capacity_observed"] = (
        output["capacity_peak95_mbps"].notna()
        & output["buildBandwidth"].gt(0)
        & ~output["capacity_peak95_conflict"].fillna(False).astype(bool)
    )
    output["node_utilization"] = (
        output["capacity_peak95_mbps"] / output["buildBandwidth"]
    ).where(output["capacity_observed"])
    return output, audit


def _pool_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return frame["business"].astype(str)
    return frame[["business", *columns]].fillna("").astype(str).agg("|".join, axis=1)


def aggregate_capacity_pools(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return pd.DataFrame()
    base = facts.copy()
    if "schedule_target_isp" not in base:
        base["schedule_target_isp"] = base.apply(
            lambda row: v2.primary_schedule_isp({
                "isp": row.get("daily_isp"),
                "scheduleisps": row.get("daily_scheduleisps"),
            }),
            axis=1,
        )
    base["_observed_bw"] = base["buildBandwidth"].where(base["capacity_observed"], 0.0)
    base["_observed_peak"] = base["capacity_peak95_mbps"].where(
        base["capacity_observed"], 0.0
    )
    base["_observed_node"] = base["capacity_observed"].astype(int)
    base["_low_utilization"] = (
        base["capacity_observed"] & base["node_utilization"].lt(0.50)
    ).astype(int)
    base["_ratio_source"] = (
        base["capacity_observed"]
        & base["capacity_peak95_source"].eq("peak95Ratio_reconstructed")
    ).astype(int)
    outputs: list[pd.DataFrame] = []
    for level, dimensions in POOL_LEVELS.items():
        group_columns = ["sample_day", "business", "business_name", *dimensions]
        daily = base.groupby(group_columns, dropna=False, sort=False).agg(
            node_count=("node_id", "nunique"),
            active_build_bandwidth_mbps=("buildBandwidth", "sum"),
            observed_node_count=("_observed_node", "sum"),
            observed_build_bandwidth_mbps=("_observed_bw", "sum"),
            observed_peak95_mbps=("_observed_peak", "sum"),
            low_utilization_node_count=("_low_utilization", "sum"),
            reconstructed_ratio_node_count=("_ratio_source", "sum"),
            miner_income=("miner_income", "sum"),
            platform_revenue=("platform_revenue", "sum"),
            platform_profit=("platform_profit", "sum"),
        ).reset_index()
        observed = base[base["capacity_observed"]]
        if not observed.empty:
            quantiles = (
                observed.groupby(group_columns, dropna=False)["node_utilization"]
                .quantile([0.25, 0.5, 0.75]).unstack()
                .rename(columns={0.25: "node_utilization_p25", 0.5: "node_utilization_p50", 0.75: "node_utilization_p75"})
                .reset_index()
            )
            daily = daily.merge(quantiles, on=group_columns, how="left", sort=False)
        else:
            for column in ["node_utilization_p25", "node_utilization_p50", "node_utilization_p75"]:
                daily[column] = np.nan
        daily["bandwidth_observation_coverage"] = (
            daily["observed_build_bandwidth_mbps"]
            / daily["active_build_bandwidth_mbps"].replace(0, np.nan)
        )
        daily["observed_utilization"] = (
            daily["observed_peak95_mbps"]
            / daily["observed_build_bandwidth_mbps"].replace(0, np.nan)
        )
        daily["estimated_total_peak95_mbps"] = (
            daily["observed_utilization"] * daily["active_build_bandwidth_mbps"]
        )
        daily["low_utilization_node_share"] = (
            daily["low_utilization_node_count"]
            / daily["observed_node_count"].replace(0, np.nan)
        )
        daily["ratio_reconstructed_node_share"] = (
            daily["reconstructed_ratio_node_count"]
            / daily["observed_node_count"].replace(0, np.nan)
        )
        daily["miner_income_per_build_mbps"] = (
            daily["miner_income"] / daily["active_build_bandwidth_mbps"].replace(0, np.nan)
        )
        daily["platform_profit_per_build_mbps"] = (
            daily["platform_profit"] / daily["active_build_bandwidth_mbps"].replace(0, np.nan)
        )
        daily["pool_level"] = level
        daily["pool_key"] = _pool_key(daily, dimensions)
        outputs.append(daily)
    return pd.concat(outputs, ignore_index=True, sort=False)


def safe_corr(left: pd.Series, right: pd.Series, rank: bool = False) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    if rank:
        pair = pair.rank(method="average")
    return float(pair["left"].corr(pair["right"]))


def build_capacity_pool_summary(
    pool_daily: pd.DataFrame,
    target_utilization: float,
    forecast_days: int,
    minimum_valid_days: int,
    minimum_bandwidth_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pool_daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = pool_daily.copy()
    if "pool_level" not in frame:
        frame["pool_level"] = "business"
    if "pool_key" not in frame:
        frame["pool_key"] = frame["business"].map(clean_id)
    dimension_columns = [
        "daily_province", "daily_isp", "network_schedule_type",
        "schedule_target_isp",
    ]
    for column in dimension_columns:
        if column not in frame:
            frame[column] = ""
    frame["sample_day"] = pd.to_datetime(frame["sample_day"], errors="coerce")
    as_of = frame["sample_day"].max()
    summaries: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    group_columns = ["pool_level", "pool_key", "business", "business_name"]
    for (level, key, business, name), group in frame.groupby(
        group_columns, dropna=False, sort=False,
    ):
        group = group.sort_values("sample_day")
        valid = group[
            group["bandwidth_observation_coverage"].ge(minimum_bandwidth_coverage)
            & group["observed_utilization"].between(0, v3.MAX_CAPACITY_UTILIZATION)
            & group["estimated_total_peak95_mbps"].notna()
        ].copy()
        recent = valid[valid["sample_day"].gt(as_of - pd.Timedelta(days=forecast_days))]
        forecast_source = recent if not recent.empty else valid.tail(forecast_days)
        valid_days = int(len(valid))
        recent_coverage = float(forecast_source["bandwidth_observation_coverage"].mean()) if not forecast_source.empty else 0.0
        evidence = valid_days >= minimum_valid_days and recent_coverage >= minimum_bandwidth_coverage
        forecast_peak = (
            float(forecast_source["estimated_total_peak95_mbps"].quantile(0.75))
            if evidence else float("nan")
        )
        raw_ceiling = forecast_peak / target_utilization if evidence else float("nan")
        as_of_rows = group[group["sample_day"].eq(as_of)]
        current_bw = float(as_of_rows["active_build_bandwidth_mbps"].sum())
        current_peak = float(as_of_rows["estimated_total_peak95_mbps"].sum(min_count=1)) if not as_of_rows.empty else float("nan")
        current_util = current_peak / current_bw if current_bw > 0 and math.isfinite(current_peak) else float("nan")
        headroom = max(raw_ceiling - current_bw, 0.0) if evidence else 0.0
        oversupply = max(current_bw - raw_ceiling, 0.0) if evidence else 0.0
        if not evidence:
            state = "证据不足"
            confidence = "low"
        elif oversupply > max(raw_ceiling * 0.05, 100.0):
            state = "建设带宽偏多"
            confidence = "high" if valid_days >= 21 and recent_coverage >= 0.95 else "medium"
        elif headroom > max(raw_ceiling * 0.10, 100.0):
            state = "仍有容量"
            confidence = "high" if valid_days >= 21 and recent_coverage >= 0.95 else "medium"
        else:
            state = "接近目标"
            confidence = "high" if valid_days >= 21 and recent_coverage >= 0.95 else "medium"
        dimensions = {
            column: v1.clean_cell(group.iloc[-1].get(column))
            for column in dimension_columns
        }
        summaries.append({
            "pool_level": level,
            "pool_key": key,
            "business": clean_id(business),
            "business_name": name,
            **dimensions,
            "as_of_day": as_of.strftime("%Y-%m-%d"),
            "history_days": int(len(group)),
            "valid_capacity_days": valid_days,
            "recent_forecast_days": int(len(forecast_source)),
            "recent_bandwidth_coverage": recent_coverage,
            "capacity_evidence_sufficient": evidence,
            "capacity_confidence": confidence,
            "capacity_state": state,
            "target_utilization": target_utilization,
            "forecast_peak95_mbps_p75": forecast_peak,
            "raw_capacity_ceiling_mbps": raw_ceiling,
            "operational_capacity_ceiling_mbps": max(current_bw, raw_ceiling) if evidence else current_bw,
            "current_active_build_bandwidth_mbps": current_bw,
            "current_estimated_peak95_mbps": current_peak,
            "current_utilization": current_util,
            "allocatable_headroom_mbps": headroom,
            "oversupplied_bandwidth_mbps": oversupply,
            "median_utilization": float(valid["observed_utilization"].median()) if not valid.empty else float("nan"),
            "low_utilization_day_share": float(valid["observed_utilization"].lt(0.50).mean()) if not valid.empty else float("nan"),
            "median_low_utilization_node_share": float(valid["low_utilization_node_share"].median()) if not valid.empty else float("nan"),
            "median_platform_profit_per_build_mbps": float(valid["platform_profit_per_build_mbps"].median()) if not valid.empty else float("nan"),
        })
        traffic_slope = float("nan")
        if len(valid) >= 3 and valid["active_build_bandwidth_mbps"].nunique() >= 2:
            traffic_slope = float(np.polyfit(
                valid["active_build_bandwidth_mbps"], valid["estimated_total_peak95_mbps"], 1
            )[0])
        correlations.append({
            "pool_level": level,
            "pool_key": key,
            "business": clean_id(business),
            "business_name": name,
            **dimensions,
            "valid_days": valid_days,
            "bandwidth_vs_utilization_pearson": safe_corr(valid["active_build_bandwidth_mbps"], valid["observed_utilization"]),
            "bandwidth_vs_utilization_spearman": safe_corr(valid["active_build_bandwidth_mbps"], valid["observed_utilization"], rank=True),
            "node_count_vs_utilization_pearson": safe_corr(valid["node_count"], valid["observed_utilization"]),
            "bandwidth_vs_traffic_pearson": safe_corr(valid["active_build_bandwidth_mbps"], valid["estimated_total_peak95_mbps"]),
            "marginal_peak95_mbps_per_build_mbps": traffic_slope,
            "interpretation": (
                "负相关表示建设带宽增加时利用率倾向下降；相关性描述历史共变，不证明因果。"
            ),
        })
    summary = pd.DataFrame(summaries)
    summary["pool_level_order"] = summary["pool_level"].map(
        {level: index for index, level in enumerate(POOL_LEVEL_ORDER)}
    )
    summary = summary.sort_values(
        ["pool_level_order", "capacity_evidence_sufficient", "current_active_build_bandwidth_mbps"],
        ascending=[True, False, False],
    ).drop(columns="pool_level_order")
    return summary.reset_index(drop=True), pd.DataFrame(correlations)


def build_business_capacity_summary(
    business_daily: pd.DataFrame,
    target_utilization: float,
    forecast_days: int,
    minimum_valid_days: int,
    minimum_bandwidth_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible business-level view of the generic pool summary."""
    summary, correlations = build_capacity_pool_summary(
        business_daily, target_utilization, forecast_days,
        minimum_valid_days, minimum_bandwidth_coverage,
    )
    if summary.empty:
        return summary, correlations
    return (
        summary[summary["pool_level"].eq("business")].reset_index(drop=True),
        correlations[correlations["pool_level"].eq("business")].reset_index(drop=True),
    )


def load_recommendations(path: Path) -> pd.DataFrame:
    id_columns = ["node_id", "current_business", "business_top1", "business_top2", "business_top3"]
    frame = pd.read_csv(path, dtype={column: "string" for column in id_columns}, low_memory=False)
    for column in id_columns:
        frame[column] = frame[column].map(clean_id)
    return frame


def attach_current_capacity_snapshot(
    recommendations: pd.DataFrame,
    facts: pd.DataFrame,
) -> pd.DataFrame:
    output = recommendations.copy()
    output["schedule_target_isp"] = output.apply(
        lambda row: v2.primary_schedule_isp({
            "isp": row.get("isp"),
            "scheduleisps": row.get("scheduleisps"),
        }),
        axis=1,
    )
    current_columns = {
        "daily_province": "capacity_current_province",
        "daily_isp": "capacity_current_isp",
        "network_schedule_type": "capacity_current_network_schedule_type",
        "schedule_target_isp": "capacity_current_schedule_target_isp",
        "buildBandwidth": "capacity_current_build_bandwidth_mbps",
    }
    current_columns = {
        source: target for source, target in current_columns.items()
        if source in facts
    }
    if not facts.empty and "current_business_day" in output:
        snapshots = facts[[
            "node_id", "sample_day", "business", *current_columns,
        ]].copy()
        snapshots["node_id"] = snapshots["node_id"].map(clean_id)
        snapshots["business"] = snapshots["business"].map(clean_id)
        snapshots["sample_day"] = snapshots["sample_day"].astype(str)
        snapshots = snapshots.rename(columns={
            "sample_day": "current_business_day",
            "business": "current_business",
            **current_columns,
        }).drop_duplicates(["node_id", "current_business_day", "current_business"])
        output["current_business_day"] = output["current_business_day"].astype(str)
        output = output.merge(
            snapshots,
            on=["node_id", "current_business_day", "current_business"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
    for column in [
        "province", "isp", "network_schedule_type", "schedule_target_isp",
    ]:
        current_column = f"capacity_current_{column}"
        if current_column in output:
            output[f"capacity_target_{column}"] = output.apply(
                lambda row, source=current_column, fallback=column: (
                    v1.clean_cell(row.get(source))
                    or v1.clean_cell(row.get(fallback))
                ),
                axis=1,
            )
        else:
            output[f"capacity_target_{column}"] = output[column]
    return output


def _candidate_options(row: pd.Series, minimum_score_gain: float) -> list[dict[str, Any]]:
    current = clean_id(row.get("current_business"))
    current_score = finite(row.get("current_predicted_combined_score"), float("nan"))
    options: list[dict[str, Any]] = [{
        "business": current,
        "name": v1.clean_cell(row.get("current_business_name")),
        "rank": 0,
        "score": current_score if math.isfinite(current_score) else 0.0,
        "gain": 0.0,
        "confidence": "current",
    }]
    if row.get("recommendation_action") == "暂不推荐执行" or not math.isfinite(current_score):
        return options
    seen = {current}
    for rank in (1, 2, 3):
        business = clean_id(row.get(f"business_top{rank}"))
        score = finite(row.get(f"combined_score_top{rank}"), float("nan"))
        confidence = v1.clean_cell(row.get(f"confidence_top{rank}"))
        platform_profit = finite(row.get(f"platform_unit_profit_top{rank}"), float("nan"))
        gain = score - current_score
        if (
            not business or business in seen or not math.isfinite(score)
            or CONFIDENCE_ORDER.get(confidence, 0) < CONFIDENCE_ORDER["medium"]
            or not math.isfinite(platform_profit) or platform_profit <= 0
            or gain < minimum_score_gain
        ):
            continue
        seen.add(business)
        options.append({
            "business": business,
            "name": v1.clean_cell(row.get(f"business_name_top{rank}")),
            "rank": rank,
            "score": score,
            "gain": gain,
            "confidence": confidence,
        })
    return options


def truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def pool_constraint_id(level: str, key: str) -> str:
    return f"{level}::{key}"


def node_pool_key(
    business: str,
    row: pd.Series,
    level: str,
    current_route: bool = False,
) -> str:
    values = [clean_id(business)]
    for column in POOL_NODE_FIELDS[level]:
        value = ""
        if current_route:
            value = v1.clean_cell(row.get(f"capacity_current_{column}"))
        else:
            value = v1.clean_cell(row.get(f"capacity_target_{column}"))
        value = value or v1.clean_cell(row.get(column))
        if column == "schedule_target_isp" and not value:
            value = v2.primary_schedule_isp(row)
        values.append(value)
    return "|".join(values)


def normalize_capacity_summary(capacity_summary: pd.DataFrame) -> pd.DataFrame:
    capacities = capacity_summary.copy()
    capacities["business"] = capacities["business"].map(clean_id)
    if "pool_level" not in capacities:
        capacities["pool_level"] = "business"
    if "pool_key" not in capacities:
        capacities["pool_key"] = capacities["business"]
    capacities["pool_level"] = capacities["pool_level"].fillna("business").astype(str)
    capacities["pool_key"] = capacities["pool_key"].fillna("").astype(str)
    capacities["pool_constraint_id"] = capacities.apply(
        lambda row: pool_constraint_id(row["pool_level"], row["pool_key"]), axis=1,
    )
    return capacities


def option_pool_constraints(
    business: str,
    row: pd.Series,
    pools: dict[str, dict[str, Any]],
    evidence_only: bool,
    current_route: bool = False,
) -> tuple[list[str], list[str]]:
    constraint_ids: list[str] = []
    levels: list[str] = []
    for level in POOL_LEVEL_ORDER:
        key = node_pool_key(business, row, level, current_route=current_route)
        constraint_id = pool_constraint_id(level, key)
        capacity = pools.get(constraint_id)
        if capacity is None:
            continue
        if evidence_only and not truthy(capacity.get("capacity_evidence_sufficient")):
            continue
        constraint_ids.append(constraint_id)
        levels.append(level)
    return constraint_ids, levels


def optimize_allocations(
    recommendations: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    minimum_score_gain: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    capacities = normalize_capacity_summary(capacity_summary)
    pools = {
        row["pool_constraint_id"]: row.to_dict()
        for _, row in capacities.iterrows()
    }
    base_load = dict(zip(
        capacities["pool_constraint_id"],
        capacities["current_active_build_bandwidth_mbps"].map(finite),
    ))
    cap_limit = dict(zip(
        capacities["pool_constraint_id"],
        capacities["operational_capacity_ceiling_mbps"].map(finite),
    ))
    business_capacities = capacities[capacities["pool_level"].eq("business")]
    business_evidence = dict(zip(
        business_capacities["business"],
        business_capacities["capacity_evidence_sufficient"].map(truthy),
    ))
    names = dict(zip(business_capacities["business"], business_capacities["business_name"]))
    required_candidate_levels = ["business"]
    if capacities["pool_level"].eq("business_isp").any():
        required_candidate_levels.append("business_isp")

    eligible_indices: list[int] = []
    option_sets: list[list[dict[str, Any]]] = []
    options_by_index: dict[int, list[dict[str, Any]]] = {}
    for index, row in recommendations.iterrows():
        if row.get("current_business_status") != "current_not_top1":
            continue
        options: list[dict[str, Any]] = []
        for option in _candidate_options(row, minimum_score_gain):
            constraint_ids, levels = option_pool_constraints(
                option["business"], row, pools,
                evidence_only=option["rank"] > 0,
                current_route=option["rank"] == 0,
            )
            if option["rank"] > 0 and not all(
                level in levels for level in required_candidate_levels
            ):
                continue
            option = {
                **option,
                "pool_constraints": constraint_ids,
                "pool_levels": levels,
                "most_specific_level": levels[-1] if levels else "",
            }
            options.append(option)
        if len(options) > 1:
            eligible_indices.append(index)
            option_sets.append(options)
            options_by_index[index] = options

    eligible_current_bw: dict[str, float] = {}
    for node_position, index in enumerate(eligible_indices):
        bandwidth = finite(recommendations.loc[index].get("build_bandwidth_mbps"))
        current_option = option_sets[node_position][0]
        for constraint_id in current_option["pool_constraints"]:
            eligible_current_bw[constraint_id] = (
                eligible_current_bw.get(constraint_id, 0.0) + bandwidth
            )
    fixed_load = {
        constraint_id: max(
            base_load.get(constraint_id, 0.0)
            - eligible_current_bw.get(constraint_id, 0.0),
            0.0,
        )
        for constraint_id in cap_limit
    }
    effective_cap_limit = {
        constraint_id: max(
            cap_limit[constraint_id],
            fixed_load[constraint_id] + eligible_current_bw.get(constraint_id, 0.0),
        )
        for constraint_id in cap_limit
    }

    selected_by_index: dict[int, dict[str, Any]] = {}
    solver_message = "no eligible nodes"
    solver_name = "none"
    if eligible_indices and importlib.util.find_spec("scipy") is not None:
        solver_name = "scipy.optimize.milp"
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix

        variables: list[tuple[int, dict[str, Any]]] = []
        for node_position, options in enumerate(option_sets):
            variables.extend((node_position, option) for option in options)
        node_count = len(option_sets)
        constraint_ids = sorted(cap_limit)
        constraint_position = {
            constraint_id: index for index, constraint_id in enumerate(constraint_ids)
        }
        node_constraints = lil_matrix((node_count, len(variables)))
        capacity_constraints = lil_matrix((len(constraint_ids), len(variables)))
        objective = np.zeros(len(variables))
        for variable, (node_position, option) in enumerate(variables):
            row = recommendations.loc[eligible_indices[node_position]]
            bandwidth = finite(row.get("build_bandwidth_mbps"))
            node_constraints[node_position, variable] = 1.0
            for constraint_id in option["pool_constraints"]:
                capacity_constraints[constraint_position[constraint_id], variable] = bandwidth
            # Maximize score gain, with stable tiny preferences for higher confidence/rank.
            objective[variable] = -(
                option["gain"]
                + 1e-6 * CONFIDENCE_ORDER.get(option["confidence"], 0)
                - 1e-8 * option["rank"]
            )
        upper = np.array([
            max(effective_cap_limit[constraint_id] - fixed_load[constraint_id], 0.0)
            for constraint_id in constraint_ids
        ])
        constraints = [
            LinearConstraint(node_constraints.tocsr(), np.ones(node_count), np.ones(node_count)),
            LinearConstraint(
                capacity_constraints.tocsr(),
                np.full(len(constraint_ids), -np.inf), upper,
            ),
        ]
        result = milp(
            c=objective,
            integrality=np.ones(len(variables)),
            bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
            constraints=constraints,
            options={"time_limit": 120},
        )
        solver_message = str(result.message)
        if result.x is None:
            raise RuntimeError(f"capacity allocation solver failed: {result.message}")
        for variable, chosen in enumerate(result.x):
            if chosen < 0.5:
                continue
            node_position, option = variables[variable]
            selected_by_index[eligible_indices[node_position]] = option
    elif eligible_indices:
        solver_name = "deterministic_capacity_greedy"
        solver_message = "SciPy unavailable; used deterministic score-gain greedy fallback"
        simulated_load = {
            constraint_id: fixed_load[constraint_id]
            + eligible_current_bw.get(constraint_id, 0.0)
            for constraint_id in cap_limit
        }
        order = sorted(
            range(len(eligible_indices)),
            key=lambda position: (
                -max(option["gain"] for option in option_sets[position]),
                clean_id(recommendations.loc[eligible_indices[position]].get("node_id")),
            ),
        )
        for position in order:
            index = eligible_indices[position]
            row = recommendations.loc[index]
            bandwidth = finite(row.get("build_bandwidth_mbps"))
            current_option = option_sets[position][0]
            candidates = sorted(
                (option for option in option_sets[position] if option["rank"] > 0),
                key=lambda option: (-option["gain"], option["rank"], option["business"]),
            )
            for option in candidates:
                if any(
                    simulated_load.get(constraint_id, 0.0) + bandwidth
                    > effective_cap_limit[constraint_id] + 1e-9
                    for constraint_id in option["pool_constraints"]
                ):
                    continue
                selected_by_index[index] = option
                for constraint_id in current_option["pool_constraints"]:
                    simulated_load[constraint_id] = max(
                        simulated_load.get(constraint_id, 0.0) - bandwidth, 0.0,
                    )
                for constraint_id in option["pool_constraints"]:
                    simulated_load[constraint_id] = (
                        simulated_load.get(constraint_id, 0.0) + bandwidth
                    )
                break

    rows: list[dict[str, Any]] = []
    eligible_index_set = set(eligible_indices)
    for index, row in recommendations.iterrows():
        current = clean_id(row.get("current_business"))
        selected = selected_by_index.get(index)
        if selected is None:
            constraint_ids, levels = option_pool_constraints(
                current, row, pools, evidence_only=False, current_route=True,
            )
            selected = {
                "business": current, "name": v1.clean_cell(row.get("current_business_name")),
                "rank": 0, "score": finite(row.get("current_predicted_combined_score"), float("nan")),
                "gain": 0.0, "confidence": "current",
                "pool_constraints": constraint_ids,
                "pool_levels": levels,
                "most_specific_level": levels[-1] if levels else "",
            }
        changed = bool(current and selected["rank"] > 0 and selected["business"] != current)
        candidates = [
            option for option in options_by_index.get(index, []) if option["rank"] > 0
        ]
        best_candidate = min(
            candidates,
            key=lambda option: (-option["gain"], option["rank"], option["business"]),
        ) if candidates else None
        if changed:
            decision = "容量约束下建议人工复核切换"
            reason = (
                f"V5 Top{selected['rank']}适配分提升{selected['gain']:.3f}，"
                f"且{len(selected['pool_levels'])}层有效容量池均可承接"
            )
        elif row.get("current_business_status") == "already_top1":
            decision = "保持当前业务"
            reason = "当前业务已是V5 Top1"
        elif row.get("current_business_status") == "current_business_unknown":
            decision = "暂不分配"
            reason = "当前业务未知，无法安全释放原业务容量"
        elif row.get("current_business_status") == "current_business_not_in_candidate_set":
            decision = "暂不分配"
            reason = "当前业务不在主流候选集，先人工核实"
        elif row.get("recommendation_action") == "暂不推荐执行":
            decision = "保持当前业务"
            reason = "V5节点适配证据不足"
        elif index in eligible_indices:
            decision = "保持当前业务"
            reason = "更优候选受分层容量约束，或全局方案保留当前业务收益更高"
        else:
            decision = "保持当前业务"
            reason = "没有同时满足分数增益、正平台利润和中高置信度的候选"
        rows.append({
            "node_id": clean_id(row.get("node_id")),
            "province": row.get("province"),
            "isp": row.get("isp"),
            "network_schedule_type": row.get("network_schedule_type"),
            "schedule_target_isp": v1.clean_cell(
                row.get("capacity_target_schedule_target_isp")
            ) or v1.clean_cell(row.get("schedule_target_isp")),
            "capacity_target_network_schedule_type": row.get(
                "capacity_target_network_schedule_type"
            ),
            "capacity_current_province": row.get("capacity_current_province"),
            "capacity_current_isp": row.get("capacity_current_isp"),
            "capacity_current_network_schedule_type": row.get(
                "capacity_current_network_schedule_type"
            ),
            "capacity_current_schedule_target_isp": row.get(
                "capacity_current_schedule_target_isp"
            ),
            "build_bandwidth_mbps": finite(row.get("build_bandwidth_mbps")),
            "current_business": current,
            "current_business_name": row.get("current_business_name"),
            "current_business_status": row.get("current_business_status"),
            "v5_recommendation_action": row.get("recommendation_action"),
            "v5_recommendation_confidence": row.get("recommendation_confidence"),
            "model_top1_business": clean_id(row.get("business_top1")),
            "model_top1_business_name": row.get("business_name_top1"),
            "model_top1_score": finite(row.get("combined_score_top1"), float("nan")),
            "planned_business": selected["business"],
            "planned_business_name": selected["name"] or names.get(selected["business"], ""),
            "capacity_aware_rank": selected["rank"],
            "planned_score": selected["score"],
            "planned_score_gain_vs_current": selected["gain"],
            "planned_confidence": selected["confidence"],
            "capacity_evidence_sufficient": business_evidence.get(selected["business"], False),
            "capacity_constraint_levels": " > ".join(selected["pool_levels"]),
            "capacity_most_specific_level": selected["most_specific_level"],
            "capacity_pool_keys": " || ".join(selected["pool_constraints"]),
            "capacity_optimizer_eligible": index in eligible_index_set,
            "best_capacity_candidate_business": (
                best_candidate["business"] if best_candidate else ""
            ),
            "best_capacity_candidate_business_name": (
                best_candidate["name"] if best_candidate else ""
            ),
            "best_capacity_candidate_rank": (
                best_candidate["rank"] if best_candidate else 0
            ),
            "best_capacity_candidate_score_gain": (
                best_candidate["gain"] if best_candidate else float("nan")
            ),
            "best_capacity_candidate_constraint_levels": (
                " > ".join(best_candidate["pool_levels"]) if best_candidate else ""
            ),
            "best_capacity_candidate_most_specific_level": (
                best_candidate["most_specific_level"] if best_candidate else ""
            ),
            "best_capacity_candidate_pool_keys": (
                " || ".join(best_candidate["pool_constraints"])
                if best_candidate else ""
            ),
            "capacity_decision": decision,
            "capacity_decision_reason": reason,
            "planned_change": changed,
            "planning_only": True,
        })
    output = pd.DataFrame(rows)
    diagnostics = {
        "solver": solver_name,
        "solver_message": solver_message,
        "eligible_nodes": len(eligible_indices),
        "planned_changes": int(output["planned_change"].sum()),
        "capacity_constraints": len(cap_limit),
        "required_candidate_levels": required_candidate_levels,
    }
    return output, diagnostics


def allocation_business_summary(
    recommendations: pd.DataFrame,
    allocations: pd.DataFrame,
    capacity_summary: pd.DataFrame,
) -> pd.DataFrame:
    output = capacity_summary.copy()
    output["business"] = output["business"].map(clean_id)
    base = dict(zip(output["business"], output["current_active_build_bandwidth_mbps"].map(finite)))
    unconstrained = base.copy()
    planned = base.copy()
    switch_in_count = {business: 0 for business in base}
    switch_out_count = {business: 0 for business in base}
    for (_, rec), (_, allocation) in zip(recommendations.iterrows(), allocations.iterrows()):
        bandwidth = finite(rec.get("build_bandwidth_mbps"))
        current = clean_id(rec.get("current_business"))
        top1 = clean_id(rec.get("business_top1"))
        destination = clean_id(allocation.get("planned_business"))
        if current and current in unconstrained and top1 in unconstrained and current != top1:
            unconstrained[current] -= bandwidth
            unconstrained[top1] += bandwidth
        if bool(allocation.get("planned_change")) and current in planned and destination in planned:
            planned[current] -= bandwidth
            planned[destination] += bandwidth
            switch_out_count[current] += 1
            switch_in_count[destination] += 1
    output["unconstrained_top1_load_mbps"] = output["business"].map(unconstrained).clip(lower=0)
    output["unconstrained_top1_overflow_mbps"] = (
        output["unconstrained_top1_load_mbps"] - output["operational_capacity_ceiling_mbps"]
    ).clip(lower=0)
    output["planned_load_mbps"] = output["business"].map(planned).clip(lower=0)
    output["planned_headroom_mbps"] = (
        output["operational_capacity_ceiling_mbps"] - output["planned_load_mbps"]
    ).clip(lower=0)
    output["planned_load_ratio"] = (
        output["planned_load_mbps"]
        / output["operational_capacity_ceiling_mbps"].replace(0, np.nan)
    )
    output["planned_switch_in_nodes"] = output["business"].map(switch_in_count).fillna(0).astype(int)
    output["planned_switch_out_nodes"] = output["business"].map(switch_out_count).fillna(0).astype(int)
    return output.sort_values(
        ["unconstrained_top1_overflow_mbps", "current_active_build_bandwidth_mbps"],
        ascending=False,
    ).reset_index(drop=True)


def allocation_pool_summary(
    recommendations: pd.DataFrame,
    allocations: pd.DataFrame,
    capacity_summary: pd.DataFrame,
) -> pd.DataFrame:
    output = normalize_capacity_summary(capacity_summary)
    pools = {
        row["pool_constraint_id"]: row.to_dict()
        for _, row in output.iterrows()
    }
    base = dict(zip(
        output["pool_constraint_id"],
        output["current_active_build_bandwidth_mbps"].map(finite),
    ))
    unconstrained = base.copy()
    planned = base.copy()
    switch_in_count = {constraint_id: 0 for constraint_id in base}
    switch_out_count = {constraint_id: 0 for constraint_id in base}

    def move_load(
        loads: dict[str, float], row: pd.Series, source: str, destination: str,
        bandwidth: float, count_switches: bool,
    ) -> None:
        source_ids, _ = option_pool_constraints(
            source, row, pools, evidence_only=False, current_route=True,
        )
        destination_ids, _ = option_pool_constraints(
            destination, row, pools, evidence_only=False,
        )
        for constraint_id in source_ids:
            loads[constraint_id] = max(loads.get(constraint_id, 0.0) - bandwidth, 0.0)
            if count_switches:
                switch_out_count[constraint_id] += 1
        for constraint_id in destination_ids:
            loads[constraint_id] = loads.get(constraint_id, 0.0) + bandwidth
            if count_switches:
                switch_in_count[constraint_id] += 1

    for (_, rec), (_, allocation) in zip(recommendations.iterrows(), allocations.iterrows()):
        bandwidth = finite(rec.get("build_bandwidth_mbps"))
        current = clean_id(rec.get("current_business"))
        top1 = clean_id(rec.get("business_top1"))
        destination = clean_id(allocation.get("planned_business"))
        if current and top1 and current != top1:
            move_load(unconstrained, rec, current, top1, bandwidth, False)
        if bool(allocation.get("planned_change")) and current and destination:
            move_load(planned, rec, current, destination, bandwidth, True)

    output["unconstrained_top1_load_mbps"] = (
        output["pool_constraint_id"].map(unconstrained).clip(lower=0)
    )
    output["unconstrained_top1_overflow_mbps"] = (
        output["unconstrained_top1_load_mbps"]
        - output["operational_capacity_ceiling_mbps"]
    ).clip(lower=0)
    output["planned_load_mbps"] = output["pool_constraint_id"].map(planned).clip(lower=0)
    output["planned_headroom_mbps"] = (
        output["operational_capacity_ceiling_mbps"] - output["planned_load_mbps"]
    ).clip(lower=0)
    output["planned_load_ratio"] = (
        output["planned_load_mbps"]
        / output["operational_capacity_ceiling_mbps"].replace(0, np.nan)
    )
    output["planned_overflow_mbps"] = (
        output["planned_load_mbps"] - output["operational_capacity_ceiling_mbps"]
    ).clip(lower=0)
    output["constraint_active"] = output["capacity_evidence_sufficient"].map(truthy)
    output["planned_switch_in_nodes"] = (
        output["pool_constraint_id"].map(switch_in_count).fillna(0).astype(int)
    )
    output["planned_switch_out_nodes"] = (
        output["pool_constraint_id"].map(switch_out_count).fillna(0).astype(int)
    )
    output["pool_level_order"] = output["pool_level"].map(
        {level: index for index, level in enumerate(POOL_LEVEL_ORDER)}
    )
    return output.sort_values(
        ["pool_level_order", "planned_overflow_mbps", "current_active_build_bandwidth_mbps"],
        ascending=[True, False, False],
    ).drop(columns="pool_level_order").reset_index(drop=True)


def split_pool_constraints(value: Any) -> list[str]:
    text = v1.clean_cell(value)
    return [item for item in text.split(" || ") if item] if text else []


def build_node_capacity_report(
    allocations: pd.DataFrame,
    pool_summary: pd.DataFrame,
) -> pd.DataFrame:
    pools = {
        row["pool_constraint_id"]: row.to_dict()
        for _, row in pool_summary.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, allocation in allocations.iterrows():
        changed = truthy(allocation.get("planned_change"))
        eligible = truthy(allocation.get("capacity_optimizer_eligible"))
        if changed:
            category = (
                "建议切换-Top1"
                if int(finite(allocation.get("capacity_aware_rank"))) == 1
                else f"建议切换-Top{int(finite(allocation.get('capacity_aware_rank')))}"
            )
            evaluated_business = clean_id(allocation.get("planned_business"))
            evaluated_name = v1.clean_cell(allocation.get("planned_business_name"))
            route_ids = split_pool_constraints(allocation.get("capacity_pool_keys"))
            most_specific = v1.clean_cell(
                allocation.get("capacity_most_specific_level")
            )
        elif eligible:
            category = "容量约束保留"
            evaluated_business = clean_id(
                allocation.get("best_capacity_candidate_business")
            )
            evaluated_name = v1.clean_cell(
                allocation.get("best_capacity_candidate_business_name")
            )
            route_ids = split_pool_constraints(
                allocation.get("best_capacity_candidate_pool_keys")
            )
            most_specific = v1.clean_cell(
                allocation.get("best_capacity_candidate_most_specific_level")
            )
        elif allocation.get("capacity_decision") == "暂不分配":
            category = "暂不分配"
            evaluated_business = clean_id(allocation.get("planned_business"))
            evaluated_name = v1.clean_cell(allocation.get("planned_business_name"))
            route_ids = split_pool_constraints(allocation.get("capacity_pool_keys"))
            most_specific = v1.clean_cell(
                allocation.get("capacity_most_specific_level")
            )
        elif allocation.get("current_business_status") == "already_top1":
            category = "当前已是Top1"
            evaluated_business = clean_id(allocation.get("planned_business"))
            evaluated_name = v1.clean_cell(allocation.get("planned_business_name"))
            route_ids = split_pool_constraints(allocation.get("capacity_pool_keys"))
            most_specific = v1.clean_cell(
                allocation.get("capacity_most_specific_level")
            )
        else:
            category = "保持当前"
            evaluated_business = clean_id(allocation.get("planned_business"))
            evaluated_name = v1.clean_cell(allocation.get("planned_business_name"))
            route_ids = split_pool_constraints(allocation.get("capacity_pool_keys"))
            most_specific = v1.clean_cell(
                allocation.get("capacity_most_specific_level")
            )

        route_pools = [pools[constraint_id] for constraint_id in route_ids if constraint_id in pools]
        active_route_pools = [
            pool for pool in route_pools if truthy(pool.get("constraint_active"))
        ]
        bandwidth = finite(allocation.get("build_bandwidth_mbps"))
        blocking_pools = []
        if eligible and not changed:
            blocking_pools = [
                pool for pool in active_route_pools
                if finite(pool.get("planned_headroom_mbps")) + 1e-5 < bandwidth
            ]
        if blocking_pools:
            bottleneck = min(
                blocking_pools,
                key=lambda pool: finite(pool.get("planned_headroom_mbps")) - bandwidth,
            )
        elif active_route_pools:
            bottleneck = min(
                active_route_pools,
                key=lambda pool: finite(pool.get("planned_headroom_mbps")),
            )
        elif route_pools:
            bottleneck = min(
                route_pools,
                key=lambda pool: finite(pool.get("planned_headroom_mbps")),
            )
        else:
            bottleneck = {}

        bottleneck_level = v1.clean_cell(bottleneck.get("pool_level"))
        bottleneck_headroom = finite(
            bottleneck.get("planned_headroom_mbps"), float("nan")
        )
        if blocking_pools:
            diagnostic = (
                f"{POOL_LEVEL_LABELS.get(bottleneck_level, bottleneck_level)}剩余"
                f"{bottleneck_headroom / 1000:.2f} Gbps，小于节点"
                f"{bandwidth / 1000:.2f} Gbps"
            )
        elif changed and bottleneck:
            diagnostic = (
                f"最紧容量池为{POOL_LEVEL_LABELS.get(bottleneck_level, bottleneck_level)}，"
                f"规划后剩余{bottleneck_headroom / 1000:.2f} Gbps"
            )
        elif eligible:
            diagnostic = "候选受全局容量组合约束，保留当前业务"
        else:
            diagnostic = v1.clean_cell(allocation.get("capacity_decision_reason"))

        rows.append({
            **allocation.to_dict(),
            "capacity_report_category": category,
            "capacity_evaluated_business": evaluated_business,
            "capacity_evaluated_business_name": evaluated_name,
            "capacity_active_constraint_count": len(active_route_pools),
            "capacity_blocking_pool_count": len(blocking_pools),
            "capacity_fallback_used": bool(
                most_specific
                and most_specific != "business_province_isp_schedule"
            ),
            "capacity_bottleneck_level": bottleneck_level,
            "capacity_bottleneck_level_name": POOL_LEVEL_LABELS.get(
                bottleneck_level, bottleneck_level,
            ),
            "capacity_bottleneck_pool_key": v1.clean_cell(
                bottleneck.get("pool_key")
            ),
            "capacity_bottleneck_business_name": v1.clean_cell(
                bottleneck.get("business_name")
            ),
            "capacity_bottleneck_province": v1.clean_cell(
                bottleneck.get("daily_province")
            ),
            "capacity_bottleneck_isp": v1.clean_cell(bottleneck.get("daily_isp")),
            "capacity_bottleneck_schedule_type": v1.clean_cell(
                bottleneck.get("network_schedule_type")
            ),
            "capacity_bottleneck_schedule_target_isp": v1.clean_cell(
                bottleneck.get("schedule_target_isp")
            ),
            "capacity_bottleneck_state": v1.clean_cell(
                bottleneck.get("capacity_state")
            ),
            "capacity_bottleneck_current_load_mbps": finite(
                bottleneck.get("current_active_build_bandwidth_mbps"), float("nan")
            ),
            "capacity_bottleneck_ceiling_mbps": finite(
                bottleneck.get("operational_capacity_ceiling_mbps"), float("nan")
            ),
            "capacity_bottleneck_planned_load_mbps": finite(
                bottleneck.get("planned_load_mbps"), float("nan")
            ),
            "capacity_bottleneck_headroom_mbps": bottleneck_headroom,
            "capacity_bottleneck_load_ratio": finite(
                bottleneck.get("planned_load_ratio"), float("nan")
            ),
            "capacity_diagnostic": diagnostic,
        })
    return pd.DataFrame(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_report_data(
    path: Path,
    summary: dict[str, Any],
    allocation_summary: pd.DataFrame,
    pool_summary: pd.DataFrame,
    business_daily: pd.DataFrame,
    allocations: pd.DataFrame,
) -> None:
    daily_columns = [
        "sample_day", "business", "business_name", "active_build_bandwidth_mbps",
        "estimated_total_peak95_mbps", "observed_utilization", "bandwidth_observation_coverage",
    ]
    payload = {
        "summary": summary,
        "businesses": allocation_summary.to_dict("records"),
        "pools": pool_summary.to_dict("records"),
        "daily": business_daily[daily_columns].to_dict("records"),
        "allocations": allocations.to_dict("records"),
    }
    path.write_text(
        "window.V6_CAPACITY_DATA = " + json.dumps(json_ready(payload), ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


def write_node_report_data(
    path: Path,
    summary: dict[str, Any],
    node_report: pd.DataFrame,
) -> None:
    payload = {
        "summary": summary,
        "nodes": node_report.to_dict("records"),
    }
    path.write_text(
        "window.V6_NODE_REPORT_DATA = "
        + json.dumps(json_ready(payload), ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )


def build_summary(
    args: argparse.Namespace,
    raw_path: Path,
    facts: pd.DataFrame,
    audit: pd.DataFrame,
    capacity: pd.DataFrame,
    pool_capacity: pd.DataFrame,
    allocations: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    evidence_count = int(capacity["capacity_evidence_sufficient"].map(truthy).sum())
    unconstrained_overflow = float(capacity["unconstrained_top1_overflow_mbps"].sum())
    pool_counts = {}
    for level in POOL_LEVEL_ORDER:
        level_rows = pool_capacity[pool_capacity["pool_level"].eq(level)]
        pool_counts[level] = {
            "pools": int(len(level_rows)),
            "evidence_sufficient": int(level_rows["capacity_evidence_sufficient"].map(truthy).sum()),
            "current_pools": int(level_rows["current_active_build_bandwidth_mbps"].gt(0).sum()),
            "planned_overflow_pools": int(
                (level_rows["constraint_active"].map(truthy)
                 & level_rows["planned_overflow_mbps"].gt(1e-5)).sum()
            ),
        }
    return {
        "version": "v6.2_schedule_target_capacity_allocation",
        "date_window": {"start": args.start_day, "end": args.end_day},
        "capacity_definition": {
            "traffic_priority": "peak95 -> analyzePeak95 -> buildBandwidth * peak95Ratio / 100",
            "virtual_duplicate_policy": "同节点同日同一逻辑业务取一个最大有效值，不累加虚拟业务重复行",
            "outlier_policy": f"单节点利用率超过{v3.MAX_CAPACITY_UTILIZATION:.0%}的测量值无效并尝试下一数据源",
            "forecast": f"最近{args.forecast_days}天有效总流量的P75",
            "capacity_ceiling": f"预测流量 / 目标利用率{args.target_utilization:.0%}",
            "capacity_levels": list(POOL_LEVEL_ORDER),
            "schedule_target_policy": "scheduleISPs为空取节点原运营商；非空按原始顺序只取第一个运营商",
            "hierarchical_policy": "调度容量按调度类型和目标运营商拆分；同时约束所有证据充分父子容量池，细层样本不足时回退",
            "insufficient_evidence": "业务整体或业务+运营商证据不足时不允许新增；更细层证据不足时向父层回退",
        },
        "allocation_definition": {
            "objective": "在四层容量池上限内最大化V5节点-业务综合适配分增益",
            "candidate_rule": f"V5 Top3、置信度至少medium、平台利润点估计>0、分数提升>={args.minimum_score_gain}",
            "solver": diagnostics,
            "planning_only": True,
            "automated_switching_allowed": False,
        },
        "raw_path": str(raw_path),
        "raw_rows": int(len(audit) and audit["raw_rows"].sum()),
        "clean_node_days": int(len(facts)),
        "capacity_observed_node_days": int(facts["capacity_observed"].sum()),
        "capacity_observation_rate": float(facts["capacity_observed"].mean()) if len(facts) else 0.0,
        "businesses": int(len(capacity)),
        "businesses_with_sufficient_capacity_evidence": evidence_count,
        "capacity_pool_counts": pool_counts,
        "capacity_pools": int(len(pool_capacity)),
        "capacity_pools_with_sufficient_evidence": int(
            pool_capacity["capacity_evidence_sufficient"].map(truthy).sum()
        ),
        "planned_active_pool_overflow_mbps": float(
            pool_capacity.loc[
                pool_capacity["constraint_active"].map(truthy), "planned_overflow_mbps"
            ].sum()
        ),
        "report_nodes": int(len(allocations)),
        "planned_change_nodes": int(allocations["planned_change"].sum()),
        "capacity_blocked_nodes": int(
            allocations.get("capacity_report_category", pd.Series(dtype=str))
            .eq("容量约束保留").sum()
        ),
        "planned_change_rank_counts": {
            f"top{rank}": int(
                (allocations["planned_change"].map(truthy)
                 & allocations["capacity_aware_rank"].eq(rank)).sum()
            )
            for rank in (1, 2, 3)
        },
        "unconstrained_top1_overflow_mbps": unconstrained_overflow,
        "unconstrained_top1_overflow_businesses": int(capacity["unconstrained_top1_overflow_mbps"].gt(0).sum()),
        "warnings": [
            "容量是历史需求支撑能力估计，不等于客户承诺的固定需求上限。",
            "相关性不代表因果，容量分配结果只用于人工规划。",
            "当前模型整体可信度仍偏低，V6不会绕过V5的置信度与平台利润门槛。",
        ],
    }


def write_html(path: Path) -> None:
    path.write_text(REPORT_HTML, encoding="utf-8")


def write_node_html(path: Path) -> None:
    path.write_text(NODE_REPORT_HTML, encoding="utf-8")


def build(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, allowlist, names, bindings, raw_path = load_inputs(args)
    facts, audit = prepare_capacity_facts(raw, allowlist, names, bindings)
    facts.to_csv(args.output_dir / OUTPUT_FACTS, index=False)
    audit.to_csv(args.output_dir / "v6_node_day_capacity_audit.csv", index=False)
    pools = aggregate_capacity_pools(facts)
    pools.to_csv(args.output_dir / OUTPUT_POOLS, index=False)
    business_daily = pools[pools["pool_level"].eq("business")].copy()
    business_daily.to_csv(args.output_dir / OUTPUT_DAILY, index=False)
    pool_capacity_summary, correlations = build_capacity_pool_summary(
        pools, args.target_utilization, args.forecast_days,
        args.minimum_valid_days, args.minimum_bandwidth_coverage,
    )
    correlations.to_csv(args.output_dir / OUTPUT_CORRELATIONS, index=False)
    capacity_summary = pool_capacity_summary[
        pool_capacity_summary["pool_level"].eq("business")
    ].reset_index(drop=True)
    recommendations = attach_current_capacity_snapshot(
        load_recommendations(args.recommendations), facts,
    )
    allocations, diagnostics = optimize_allocations(
        recommendations, pool_capacity_summary, args.minimum_score_gain,
    )
    allocation_summary = allocation_business_summary(
        recommendations, allocations, capacity_summary,
    )
    pool_allocation_summary = allocation_pool_summary(
        recommendations, allocations, pool_capacity_summary,
    )
    node_report = build_node_capacity_report(allocations, pool_allocation_summary)
    allocations.to_csv(args.output_dir / OUTPUT_ALLOCATIONS, index=False)
    node_report.to_csv(args.output_dir / OUTPUT_NODE_REPORT, index=False)
    review_nodes = node_report[node_report["planned_change"].map(truthy)].sort_values(
        ["capacity_aware_rank", "planned_score_gain_vs_current"],
        ascending=[True, False],
    )
    review_nodes.to_csv(args.output_dir / OUTPUT_REVIEW_NODES, index=False)
    blocked_nodes = node_report[
        node_report["capacity_report_category"].eq("容量约束保留")
    ].sort_values("best_capacity_candidate_score_gain", ascending=False)
    blocked_nodes.to_csv(args.output_dir / OUTPUT_BLOCKED_NODES, index=False)
    pool_allocation_summary.to_csv(args.output_dir / OUTPUT_POOL_SUMMARY, index=False)
    allocation_summary.to_csv(args.output_dir / OUTPUT_ALLOCATION_SUMMARY, index=False)
    capacity_summary.to_csv(args.output_dir / OUTPUT_SUMMARY, index=False)
    summary = build_summary(
        args, raw_path, facts, audit, allocation_summary, pool_allocation_summary,
        node_report, diagnostics,
    )
    (args.output_dir / OUTPUT_JSON).write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / OUTPUT_NODE_JSON).write_text(
        json.dumps(
            json_ready({"summary": summary, "nodes": node_report.to_dict("records")}),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    write_report_data(
        args.output_dir / OUTPUT_DATA, summary, allocation_summary,
        pool_allocation_summary, business_daily, node_report,
    )
    write_node_report_data(args.output_dir / OUTPUT_NODE_DATA, summary, node_report)
    write_html(args.output_dir / OUTPUT_HTML)
    write_node_html(args.output_dir / OUTPUT_NODE_HTML)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


def check(args: argparse.Namespace) -> int:
    required = [
        OUTPUT_FACTS, OUTPUT_DAILY, OUTPUT_SUMMARY, OUTPUT_POOL_SUMMARY,
        OUTPUT_ALLOCATIONS, OUTPUT_ALLOCATION_SUMMARY, OUTPUT_JSON,
        OUTPUT_NODE_REPORT, OUTPUT_REVIEW_NODES, OUTPUT_BLOCKED_NODES,
        OUTPUT_NODE_JSON, OUTPUT_NODE_DATA, OUTPUT_NODE_HTML,
        OUTPUT_DATA, OUTPUT_HTML,
    ]
    missing = [name for name in required if not (args.output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing V6 outputs: {missing}")
    summary = json.loads((args.output_dir / OUTPUT_JSON).read_text(encoding="utf-8"))
    allocations = pd.read_csv(args.output_dir / OUTPUT_ALLOCATIONS, low_memory=False)
    businesses = pd.read_csv(args.output_dir / OUTPUT_ALLOCATION_SUMMARY, low_memory=False)
    pools = pd.read_csv(args.output_dir / OUTPUT_POOL_SUMMARY, low_memory=False)
    node_report = pd.read_csv(args.output_dir / OUTPUT_NODE_REPORT, low_memory=False)
    review_nodes = pd.read_csv(args.output_dir / OUTPUT_REVIEW_NODES, low_memory=False)
    blocked_nodes = pd.read_csv(args.output_dir / OUTPUT_BLOCKED_NODES, low_memory=False)
    if len(allocations) != summary["report_nodes"]:
        raise RuntimeError("allocation row count does not match summary")
    if len(node_report) != summary["report_nodes"]:
        raise RuntimeError("node report row count does not match summary")
    if len(review_nodes) != summary["planned_change_nodes"]:
        raise RuntimeError("review node count does not match summary")
    if len(blocked_nodes) != summary["capacity_blocked_nodes"]:
        raise RuntimeError("capacity-blocked node count does not match summary")
    if not review_nodes["planned_change"].map(truthy).all():
        raise RuntimeError("review node report contains a non-change row")
    if not blocked_nodes["capacity_blocking_pool_count"].gt(0).all():
        raise RuntimeError("a capacity-blocked node has no blocking pool")
    if blocked_nodes["capacity_bottleneck_pool_key"].isna().any():
        raise RuntimeError("a capacity-blocked node has no named bottleneck pool")
    node_json = json.loads(
        (args.output_dir / OUTPUT_NODE_JSON).read_text(encoding="utf-8")
    )
    if len(node_json.get("nodes", [])) != summary["report_nodes"]:
        raise RuntimeError("node JSON row count does not match summary")
    if not allocations["planning_only"].map(truthy).all():
        raise RuntimeError("V6 must remain planning-only")
    sufficient = businesses["capacity_evidence_sufficient"].map(truthy)
    overflow = businesses.loc[sufficient, "planned_load_mbps"] - businesses.loc[sufficient, "operational_capacity_ceiling_mbps"]
    if not overflow.empty and overflow.max() > 1e-5:
        raise RuntimeError(f"planned allocation exceeds capacity by {overflow.max():.3f} Mbps")
    active_pools = pools["constraint_active"].map(truthy)
    pool_overflow = (
        pools.loc[active_pools, "planned_load_mbps"]
        - pools.loc[active_pools, "operational_capacity_ceiling_mbps"]
    )
    if not pool_overflow.empty and pool_overflow.max() > 1e-5:
        raise RuntimeError(
            f"planned allocation exceeds a hierarchical pool by {pool_overflow.max():.3f} Mbps"
        )
    result = {
        "status": "ok", "report_nodes": len(allocations),
        "planned_changes": int(allocations["planned_change"].map(truthy).sum()),
        "businesses": len(businesses),
        "capacity_pools": len(pools),
        "active_capacity_pools": int(active_pools.sum()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    build_parser.add_argument("--business-map", type=Path, default=DEFAULT_BUSINESS_MAP)
    build_parser.add_argument("--virtual-bindings", type=Path)
    build_parser.add_argument("--recommendations", type=Path, default=DEFAULT_RECOMMENDATIONS)
    build_parser.add_argument("--candidate-json", type=Path, default=DEFAULT_SOURCE_DIR / "large_candidate_node_ids_recent_1m.json")
    build_parser.add_argument("--raw-input", type=Path)
    build_parser.add_argument("--start-day", default="2026-08-02")
    build_parser.add_argument("--end-day", default="2026-09-01")
    build_parser.add_argument("--target-utilization", type=float, default=0.70)
    build_parser.add_argument("--forecast-days", type=int, default=14)
    build_parser.add_argument("--minimum-valid-days", type=int, default=7)
    build_parser.add_argument("--minimum-bandwidth-coverage", type=float, default=0.80)
    build_parser.add_argument("--minimum-score-gain", type=float, default=0.03)
    build_parser.add_argument("--chunk-size", type=int, default=700)
    build_parser.add_argument("--workers", type=int, default=6)
    build_parser.add_argument("--refresh", action="store_true")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.command == "build":
        if not 0 < args.target_utilization <= 1:
            parser.error("--target-utilization must be in (0, 1]")
        if not 0 < args.minimum_bandwidth_coverage <= 1:
            parser.error("--minimum-bandwidth-coverage must be in (0, 1]")
    return args


def main() -> int:
    args = parse_args()
    return build(args) if args.command == "build" else check(args)


REPORT_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>业务容量感知推荐报告 V6</title><style>
:root{--bg:#f4f6f8;--panel:#fff;--line:#d9e0e7;--text:#18212b;--muted:#607080;--blue:#1769aa;--green:#16835f;--amber:#a86600;--red:#c9362b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;letter-spacing:0}header{background:#132a3a;color:#fff;padding:20px 28px;border-bottom:4px solid #25a57a}header h1{font-size:24px;margin:0 0 5px}header p{margin:0;color:#c9d6df}.wrap{padding:18px 24px;max-width:1680px;margin:auto}.notice{background:#fff8e5;border:1px solid #e9c56b;padding:11px 14px;margin-bottom:14px}.kpis{display:grid;grid-template-columns:repeat(6,minmax(140px,1fr));gap:10px;margin-bottom:14px}.kpi,.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px}.kpi{padding:12px}.kpi b{display:block;font-size:22px;margin-top:3px}.muted{color:var(--muted)}.panel{margin-bottom:14px}.panel h2{font-size:16px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}.tools{display:flex;gap:8px;padding:10px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap}input,select{border:1px solid #aeb9c4;border-radius:4px;padding:7px 9px;background:#fff}input{min-width:260px}.table-wrap{overflow:auto;max-height:560px}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #e6ebef}th{position:sticky;top:0;background:#edf2f5;z-index:1;font-size:12px}tr:hover td{background:#f7fafc}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.tag{display:inline-block;padding:2px 6px;border-radius:3px;background:#e8eef3;font-size:12px}.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}canvas{width:100%;height:260px;border:1px solid var(--line);background:#fff}@media(max-width:1000px){.kpis{grid-template-columns:repeat(2,1fr)}.chart-grid{grid-template-columns:1fr}.wrap{padding:12px}.table-wrap{max-height:none}}
</style></head><body><header><h1>业务容量感知推荐报告 V6.1</h1><p>节点适配由 V5 负责，V6.1 同时约束业务、运营商、调度类型和省份四层容量池</p></header><main class="wrap"><div class="notice">本报告仅用于人工规划，不会自动切换业务。细粒度容量证据不足时回退到最近的有效父层级；容量估计不代表客户承诺的固定需求上限。</div><section class="kpis" id="kpis"></section><section class="panel"><h2>业务整体容量与 Top1 压力</h2><div class="tools"><input id="bizSearch" placeholder="搜索业务名称或 ID"><select id="stateFilter"><option value="">全部容量状态</option><option>建设带宽偏多</option><option>接近目标</option><option>仍有容量</option><option>证据不足</option></select></div><div class="table-wrap"><table><thead><tr><th>业务</th><th>状态</th><th>有效天</th><th>覆盖率</th><th>当前带宽（Gbps）</th><th>当前利用率</th><th>容量上限（Gbps）</th><th>可新增（Gbps）</th><th>假设全切 Top1 负载（Gbps）</th><th>假设溢出（Gbps）</th><th>约束后负载（Gbps）</th><th>切入/切出</th></tr></thead><tbody id="bizRows"></tbody></table></div></section><section class="panel"><h2>分层容量池</h2><div class="tools"><input id="poolSearch" placeholder="搜索业务、省份、运营商或调度类型"><select id="poolLevel"><option value="">全部层级</option><option value="business">业务整体</option><option value="business_isp">业务 + 运营商</option><option value="business_isp_schedule">业务 + 运营商 + 调度</option><option value="business_province_isp_schedule">业务 + 省份 + 运营商 + 调度</option></select></div><div class="table-wrap"><table><thead><tr><th>层级</th><th>业务</th><th>省份</th><th>运营商</th><th>调度</th><th>状态</th><th>有效天</th><th>覆盖率</th><th>当前（Gbps）</th><th>容量上限（Gbps）</th><th>约束后（Gbps）</th><th>余量（Gbps）</th><th>约束生效</th><th>切入/切出</th></tr></thead><tbody id="poolRows"></tbody></table></div></section><section class="panel"><h2>重点业务历史趋势</h2><div class="tools"><select id="chartBusiness"></select></div><div class="chart-grid"><canvas id="utilChart" width="760" height="260"></canvas><canvas id="trafficChart" width="760" height="260"></canvas></div></section><section class="panel"><h2>节点容量约束方案</h2><div class="tools"><input id="nodeSearch" placeholder="搜索节点 ID、当前业务或规划业务"><select id="decisionFilter"><option value="">全部决策</option><option>容量约束下建议人工复核切换</option><option>保持当前业务</option><option>暂不分配</option></select></div><div class="table-wrap"><table><thead><tr><th>节点</th><th>省份/运营商</th><th>调度</th><th>带宽（Gbps）</th><th>当前业务</th><th>V5 Top1</th><th>容量后方案</th><th>采用排名</th><th>最细约束层级</th><th>分数增益</th><th>决策</th><th>原因</th></tr></thead><tbody id="nodeRows"></tbody></table></div></section></main><script src="v6_capacity_report_data.js"></script><script>
const D=window.V6_CAPACITY_DATA,F=n=>n==null?'--':Number(n).toLocaleString('zh-CN',{maximumFractionDigits:2}),G=n=>n==null?'--':F(Number(n)/1000),P=n=>n==null?'--':(Number(n)*100).toFixed(1)+'%';
const S=D.summary,bizSearch=document.getElementById('bizSearch'),stateFilter=document.getElementById('stateFilter'),bizRows=document.getElementById('bizRows'),poolSearch=document.getElementById('poolSearch'),poolLevel=document.getElementById('poolLevel'),poolRows=document.getElementById('poolRows'),nodeSearch=document.getElementById('nodeSearch'),decisionFilter=document.getElementById('decisionFilter'),nodeRows=document.getElementById('nodeRows'),chartBusiness=document.getElementById('chartBusiness');document.getElementById('kpis').innerHTML=[["干净节点日",F(S.clean_node_days)],["流量观测率",P(S.capacity_observation_rate)],["业务数",F(S.businesses)],["有效容量池",`${F(S.capacity_pools_with_sufficient_evidence)}/${F(S.capacity_pools)}`],["规划切换节点",F(S.planned_change_nodes)],["假设全切 Top1 溢出",G(S.unconstrained_top1_overflow_mbps)+' Gbps']].map(x=>`<div class="kpi"><span class="muted">${x[0]}</span><b>${x[1]}</b></div>`).join('');
function stateClass(s){return s==='仍有容量'?'good':s==='建设带宽偏多'?'bad':s==='证据不足'?'warn':''}function renderBiz(){let q=bizSearch.value.toLowerCase(),st=stateFilter.value;bizRows.innerHTML=D.businesses.filter(x=>(!st||x.capacity_state===st)&&(`${x.business}${x.business_name}`.toLowerCase().includes(q))).map(x=>`<tr><td><b>${x.business_name}</b><br><span class="muted">${x.business}</span></td><td class="${stateClass(x.capacity_state)}">${x.capacity_state}</td><td>${x.valid_capacity_days}</td><td>${P(x.recent_bandwidth_coverage)}</td><td>${G(x.current_active_build_bandwidth_mbps)}</td><td>${P(x.current_utilization)}</td><td>${G(x.raw_capacity_ceiling_mbps)}</td><td>${G(x.allocatable_headroom_mbps)}</td><td>${G(x.unconstrained_top1_load_mbps)}</td><td class="${x.unconstrained_top1_overflow_mbps>0?'bad':''}">${G(x.unconstrained_top1_overflow_mbps)}</td><td>${G(x.planned_load_mbps)}</td><td>${x.planned_switch_in_nodes}/${x.planned_switch_out_nodes}</td></tr>`).join('')}
const LEVEL={business:'业务整体',business_isp:'业务 + 原运营商',business_isp_schedule:'业务 + 原运营商 + 调度 + 目标运营商',business_province_isp_schedule:'业务 + 省份 + 原运营商 + 调度 + 目标运营商'};function renderPools(){let q=poolSearch.value.toLowerCase(),lv=poolLevel.value;poolRows.innerHTML=D.pools.filter(x=>(!lv||x.pool_level===lv)&&(`${x.business_name}${x.business}${x.daily_province||''}${x.daily_isp||''}${x.network_schedule_type||''}${x.schedule_target_isp||''}`.toLowerCase().includes(q))).slice(0,2500).map(x=>`<tr><td>${LEVEL[x.pool_level]||x.pool_level}</td><td><b>${x.business_name}</b><br><span class="muted">${x.business}</span></td><td>${x.daily_province||'--'}</td><td>${x.daily_isp||'--'}</td><td>${x.network_schedule_type||'--'}<br><span class="muted">目标：${x.schedule_target_isp||'--'}</span></td><td class="${stateClass(x.capacity_state)}">${x.capacity_state}</td><td>${x.valid_capacity_days}</td><td>${P(x.recent_bandwidth_coverage)}</td><td>${G(x.current_active_build_bandwidth_mbps)}</td><td>${G(x.raw_capacity_ceiling_mbps)}</td><td class="${x.planned_overflow_mbps>0?'bad':''}">${G(x.planned_load_mbps)}</td><td>${G(x.planned_headroom_mbps)}</td><td>${x.constraint_active?'是':'否，回退父层'}</td><td>${x.planned_switch_in_nodes}/${x.planned_switch_out_nodes}</td></tr>`).join('')}
function renderNodes(){let q=nodeSearch.value.toLowerCase(),st=decisionFilter.value;nodeRows.innerHTML=D.allocations.filter(x=>(!st||x.capacity_decision===st)&&(`${x.node_id}${x.current_business_name}${x.planned_business_name}${x.schedule_target_isp||''}`.toLowerCase().includes(q))).slice(0,1500).map(x=>`<tr><td>${x.node_id}</td><td>${x.province||''}/${x.isp||''}</td><td>${x.capacity_target_network_schedule_type||x.network_schedule_type||''}<br><span class="muted">目标：${x.schedule_target_isp||'--'}</span></td><td>${G(x.build_bandwidth_mbps)}</td><td>${x.current_business_name||'未知'}</td><td>${x.model_top1_business_name||''}</td><td><b>${x.planned_business_name||'--'}</b></td><td>${x.capacity_aware_rank?`Top${x.capacity_aware_rank}`:'当前'}</td><td>${LEVEL[x.capacity_most_specific_level]||'--'}</td><td>${x.planned_score_gain_vs_current==null?'--':Number(x.planned_score_gain_vs_current).toFixed(3)}</td><td class="${x.planned_change?'good':''}">${x.capacity_decision}</td><td>${x.capacity_decision_reason}</td></tr>`).join('')}
function chart(canvas,rows,key,title,color,scale=1,format=F){const c=document.getElementById(canvas),x=c.getContext('2d'),w=c.width,h=c.height,p=42;x.clearRect(0,0,w,h);x.fillStyle='#18212b';x.font='13px sans-serif';x.fillText(title,12,20);const vals=rows.map(r=>Number(r[key])/scale).filter(Number.isFinite);if(vals.length<2)return;let min=Math.min(...vals,0),max=Math.max(...vals);if(max===min)max=min+1;x.strokeStyle='#d9e0e7';for(let i=0;i<5;i++){let y=p+(h-2*p)*i/4;x.beginPath();x.moveTo(p,y);x.lineTo(w-p,y);x.stroke()}x.strokeStyle=color;x.lineWidth=2;x.beginPath();let started=false;rows.forEach((r,i)=>{let v=Number(r[key])/scale;if(!Number.isFinite(v))return;let px=p+(w-2*p)*i/Math.max(rows.length-1,1),py=h-p-(v-min)/(max-min)*(h-2*p);started?x.lineTo(px,py):x.moveTo(px,py);started=true});x.stroke();x.fillStyle='#607080';x.fillText(format(max),4,p);x.fillText(format(min),4,h-p)}
function renderCharts(){let b=chartBusiness.value,rows=D.daily.filter(x=>x.business===b).sort((a,c)=>a.sample_day.localeCompare(c.sample_day));chart('utilChart',rows,'observed_utilization','历史整体利用率','#1769aa',1,P);chart('trafficChart',rows,'estimated_total_peak95_mbps','估算 95 峰值流量（Gbps）','#16835f',1000,F)}
[bizSearch,stateFilter].forEach(x=>x.oninput=renderBiz);[poolSearch,poolLevel].forEach(x=>x.oninput=renderPools);[nodeSearch,decisionFilter].forEach(x=>x.oninput=renderNodes);chartBusiness.innerHTML=D.businesses.map(x=>`<option value="${x.business}">${x.business_name}</option>`).join('');chartBusiness.onchange=renderCharts;renderBiz();renderPools();renderNodes();renderCharts();
</script></body></html>'''

REPORT_HTML = (
    REPORT_HTML
    .replace("业务容量感知推荐报告 V6.1", "业务容量感知推荐报告 V6.2")
    .replace(
        "<title>业务容量感知推荐报告 V6</title>",
        "<title>业务容量感知推荐报告 V6.2</title>",
    )
    .replace(
        "V6.1 同时约束业务、运营商、调度类型和省份四层容量池",
        "V6.2 按节点原运营商、调度类型和调度目标运营商细分四层容量池",
    )
    .replace("搜索业务、省份、运营商或调度类型", "搜索业务、省份、原运营商、调度类型或目标运营商")
    .replace("业务 + 运营商</option>", "业务 + 原运营商</option>")
    .replace("业务 + 运营商 + 调度</option>", "业务 + 原运营商 + 调度 + 目标运营商</option>")
    .replace("业务 + 省份 + 运营商 + 调度</option>", "业务 + 省份 + 原运营商 + 调度 + 目标运营商</option>")
    .replace("<th>省份</th><th>运营商</th><th>调度</th>", "<th>省份</th><th>原运营商</th><th>调度</th>")
    .replace("<th>节点</th><th>省份/运营商</th>", "<th>节点</th><th>省份/原运营商</th>")
)


NODE_REPORT_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>节点容量规划报告 V6.2</title><style>
:root{--bg:#f4f6f8;--panel:#fff;--line:#d7dfe6;--text:#18212b;--muted:#607080;--green:#13795b;--amber:#9a6200;--red:#c9362b;--nav:#173143}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;letter-spacing:0}header{display:flex;align-items:center;justify-content:space-between;gap:20px;background:var(--nav);color:#fff;padding:18px 26px;border-bottom:4px solid #26a47b}h1{font-size:22px;margin:0}header p{margin:3px 0 0;color:#cbd6dd}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#fff;text-decoration:none;border:1px solid #6f8796;border-radius:4px;padding:6px 9px}.wrap{padding:16px 20px;max-width:1920px;margin:auto}.kpis{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:9px;margin-bottom:12px}.kpi,.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px}.kpi{padding:11px}.kpi b{display:block;font-size:21px;margin-top:2px}.muted{color:var(--muted)}.tools{display:flex;align-items:center;gap:8px;padding:10px;border-bottom:1px solid var(--line);flex-wrap:wrap}input,select,button{height:34px;border:1px solid #abb8c2;border-radius:4px;background:#fff;color:var(--text);padding:0 9px}input{min-width:270px}button{width:34px;padding:0;font-size:18px;cursor:pointer}button:disabled{opacity:.35;cursor:default}.count{margin-left:auto;color:var(--muted)}.table-wrap{overflow:auto;max-height:calc(100vh - 235px)}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:left;padding:8px 9px;border-bottom:1px solid #e5eaee;vertical-align:top}th{position:sticky;top:0;background:#edf2f5;z-index:1;font-size:12px}tr:hover td{background:#f8fafb}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.wrap-text{white-space:normal;min-width:220px;max-width:360px}.pager{display:flex;align-items:center;justify-content:flex-end;gap:8px;padding:9px 10px}@media(max-width:1000px){header{align-items:flex-start;flex-direction:column}.kpis{grid-template-columns:repeat(2,1fr)}.wrap{padding:10px}.table-wrap{max-height:none}.count{width:100%;margin-left:0}}
</style></head><body><header><div><h1>节点容量规划报告 V6.2</h1><p>按调度目标运营商细分的四层容量约束节点清单</p></div><nav class="nav"><a href="v6_capacity_report.html">容量总览</a><a href="v6_capacity_review_nodes.csv">切换清单 CSV</a><a href="v6_capacity_blocked_nodes.csv">容量拦截 CSV</a></nav></header><main class="wrap"><section class="kpis" id="kpis"></section><section class="panel"><div class="tools"><input id="search" placeholder="搜索节点或业务"><select id="category"><option value="">全部节点状态</option><option>建议切换-Top1</option><option>建议切换-Top2</option><option>建议切换-Top3</option><option>容量约束保留</option><option>当前已是Top1</option><option>保持当前</option><option>暂不分配</option></select><select id="province"><option value="">全部省份</option></select><select id="isp"><option value="">全部原运营商</option></select><select id="schedule"><option value="">全部调度类型</option></select><select id="targetIsp"><option value="">全部调度运营商</option></select><span class="count" id="count"></span></div><div class="table-wrap"><table><thead><tr><th>节点 ID</th><th>节点画像与调度</th><th>建设带宽（Gbps）</th><th>当前业务</th><th>V5 Top1</th><th>容量规划业务</th><th>采用排名</th><th>分数增益</th><th>节点状态</th><th>最紧容量层级</th><th>最紧容量池</th><th>规划后余量（Gbps）</th><th>容量诊断</th></tr></thead><tbody id="rows"></tbody></table></div><div class="pager"><button id="prev" title="上一页" aria-label="上一页">&#8592;</button><span id="page"></span><button id="next" title="下一页" aria-label="下一页">&#8594;</button></div></section></main><script src="v6_node_report_data.js"></script><script>
const D=window.V6_NODE_REPORT_DATA,S=D.summary,N=D.nodes,F=n=>n==null||!Number.isFinite(Number(n))?'--':Number(n).toLocaleString('zh-CN',{maximumFractionDigits:2}),G=n=>n==null||!Number.isFinite(Number(n))?'--':F(Number(n)/1000),E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const search=document.getElementById('search'),category=document.getElementById('category'),province=document.getElementById('province'),isp=document.getElementById('isp'),schedule=document.getElementById('schedule'),targetIsp=document.getElementById('targetIsp'),tbody=document.getElementById('rows'),count=document.getElementById('count'),pageText=document.getElementById('page'),prev=document.getElementById('prev'),next=document.getElementById('next');let page=1;const size=100;
document.getElementById('kpis').innerHTML=[["报告节点",F(S.report_nodes)],["进入容量优化",F(S.allocation_definition.solver.eligible_nodes)],["建议切换",F(S.planned_change_nodes)],["Top2/Top3",F((S.planned_change_rank_counts.top2||0)+(S.planned_change_rank_counts.top3||0))],["容量约束保留",F(S.capacity_blocked_nodes)],["生效容量池",F(S.capacity_pools_with_sufficient_evidence)]].map(x=>`<div class="kpi"><span class="muted">${x[0]}</span><b>${x[1]}</b></div>`).join('');
function options(el,key){const vals=[...new Set(N.map(x=>x[key]).filter(Boolean))].sort((a,b)=>String(a).localeCompare(String(b),'zh-CN'));el.innerHTML+=vals.map(x=>`<option>${E(x)}</option>`).join('')}options(province,'province');options(isp,'isp');options(schedule,'capacity_target_network_schedule_type');options(targetIsp,'schedule_target_isp');
function cls(x){return x.startsWith('建议切换')?'good':x==='容量约束保留'?'bad':x==='暂不分配'?'warn':''}function filtered(){const q=search.value.trim().toLowerCase();return N.filter(x=>(!category.value||x.capacity_report_category===category.value)&&(!province.value||x.province===province.value)&&(!isp.value||x.isp===isp.value)&&(!schedule.value||x.capacity_target_network_schedule_type===schedule.value)&&(!targetIsp.value||x.schedule_target_isp===targetIsp.value)&&(!q||`${x.node_id}${x.current_business_name||''}${x.model_top1_business_name||''}${x.planned_business_name||''}${x.capacity_evaluated_business_name||''}${x.schedule_target_isp||''}`.toLowerCase().includes(q)))}
function render(){const data=filtered(),pages=Math.max(1,Math.ceil(data.length/size));page=Math.min(page,pages);const start=(page-1)*size;tbody.innerHTML=data.slice(start,start+size).map(x=>`<tr><td>${E(x.node_id)}</td><td>${E(x.province||'--')} / 原${E(x.isp||'--')}<br><span class="muted">${E(x.capacity_target_network_schedule_type||x.network_schedule_type||'--')} / 调度${E(x.schedule_target_isp||'--')}</span></td><td>${G(x.build_bandwidth_mbps)}</td><td>${E(x.current_business_name||'未知')}<br><span class="muted">${E(x.current_business||'')}</span></td><td>${E(x.model_top1_business_name||'--')}</td><td><b>${E(x.planned_business_name||'--')}</b><br><span class="muted">${E(x.planned_business||'')}</span></td><td>${x.capacity_aware_rank?`Top${F(x.capacity_aware_rank)}`:'当前'}</td><td>${x.planned_change?F(x.planned_score_gain_vs_current):F(x.best_capacity_candidate_score_gain)}</td><td class="${cls(x.capacity_report_category)}">${E(x.capacity_report_category)}</td><td>${E(x.capacity_bottleneck_level_name||'--')}</td><td>${E(x.capacity_bottleneck_business_name||x.capacity_evaluated_business_name||'--')}<br><span class="muted">${E([x.capacity_bottleneck_province,x.capacity_bottleneck_isp,x.capacity_bottleneck_schedule_type,x.capacity_bottleneck_schedule_target_isp].filter(Boolean).join(' / ')||'--')}</span></td><td>${G(x.capacity_bottleneck_headroom_mbps)}</td><td class="wrap-text">${E(x.capacity_diagnostic||x.capacity_decision_reason||'')}</td></tr>`).join('');count.textContent=`${F(data.length)} 个节点`;pageText.textContent=`${page} / ${pages}`;prev.disabled=page<=1;next.disabled=page>=pages}
[search,category,province,isp,schedule,targetIsp].forEach(x=>x.addEventListener('input',()=>{page=1;render()}));prev.onclick=()=>{page--;render()};next.onclick=()=>{page++;render()};render();
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
