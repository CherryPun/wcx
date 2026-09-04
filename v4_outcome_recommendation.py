#!/usr/bin/env python3
"""V4 per-business miner/platform outcome recommendation pipeline.

The model is dependency-light and auditable: every business gets separate
weighted hierarchical regressions for miner unit income and platform unit
profit. Validation is strictly time ordered, and prediction intervals are
calibrated on dates after the training window.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v1_recommendation_pipeline as v1
import v2_ranking_model as v2


HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = (
    HERE
    / "recent_month_large_mainstream_v3_daily_weighted"
    / "v1_large_mainstream_outputs"
    / "v1_training_pairs.csv"
)
DEFAULT_HISTORICAL_PROFILES = HERE / "recent_month_large_1d" / "multibusiness_nodes_large_recent_1m.csv"
DEFAULT_LATEST_PRESSURE_PROFILES = HERE / "latest_node_pressure_profiles.csv"
DEFAULT_CURRENT_NODES = (
    HERE
    / "recent_month_large_mainstream_v3_daily_weighted"
    / "current_online_inservice_non_idc_large_nodes_v3.csv"
)
DEFAULT_CURRENT_BUSINESS = (
    HERE
    / "current_non_idc_large_scan_v3_daily"
    / "current_business_from_wide_20260901_20260901.csv"
)
DEFAULT_BUSINESS_MAP = HERE / "v1_business_name_map_enriched.csv"
DEFAULT_SOURCE_SUMMARY = (
    HERE
    / "recent_month_large_mainstream_v3_daily_weighted"
    / "v3_daily_training_summary.json"
)
DEFAULT_OUTPUT_DIR = HERE / "recent_month_large_mainstream_v4_outcome_latest_pressure"
DEFAULT_REPORT_TEMPLATE = HERE / "v4_frontend_report.html"

OUTPUT_MODEL = "v4_outcome_model.json"
OUTPUT_RECOMMENDATIONS = "v4_node_recommendations.csv"
OUTPUT_SWITCH_CANDIDATES = "v4_review_switch_candidates.csv"
OUTPUT_VALIDATION = "v4_temporal_validation_predictions.csv"
OUTPUT_BUSINESS_METRICS = "v4_business_training_metrics.csv"
OUTPUT_CONCENTRATION = "v4_recommendation_concentration.csv"
OUTPUT_SUMMARY = "v4_training_summary.json"
OUTPUT_REPORT_DATA = "v4_frontend_report_data.js"
OUTPUT_REPORT_HTML = "v4_business_recommendation_report.html"

UNKNOWN = "__UNKNOWN__"
KEY_SEPARATOR = "\x1f"
TARGET_MINER = "miner_unit_income"
TARGET_PLATFORM = "platform_unit_profit"

PROFILE_FIELDS = [
    "province",
    "isp",
    "nattype",
    "scheduleisps",
    "analysis_transprovrate",
    "analysis_issupportipv6",
    "bw",
    "corenum",
    "memtotal",
    "totaldisksize",
]

LATEST_PRESSURE_FIELDS = [
    "overall_packet_loss_benchmark_satisfaction_pct",
    "pressure_snapshot_day",
    "pressure_snapshot_hour",
    "pressure_last_report_time",
]

MODEL_LEVELS = [
    (
        "province_isp_network_bw_loss",
        [
            "province", "isp", "network_schedule_type", "bw_bucket",
            "packet_loss_satisfaction_bucket",
        ],
        2.00,
    ),
    ("province_isp_network", ["province", "isp", "network_schedule_type"], 1.60),
    (
        "isp_network_bw_loss",
        ["isp", "network_schedule_type", "bw_bucket", "packet_loss_satisfaction_bucket"],
        1.45,
    ),
    ("network_nat_bw", ["network_schedule_type", "nattype", "bw_bucket"], 1.10),
    ("hardware_capacity", ["corenum_bucket", "memtotal_bucket", "totaldisksize_bucket"], 0.95),
    (
        "nat_ipv6_loss",
        ["nattype", "ipv6_capability", "packet_loss_satisfaction_bucket"],
        0.90,
    ),
    ("ipv6_network", ["isp", "ipv6_capability", "network_schedule_type"], 0.80),
    ("province_isp", ["province", "isp"], 0.75),
    ("isp_network", ["isp", "network_schedule_type"], 0.70),
    ("nat_bw", ["nattype", "bw_bucket"], 0.55),
    ("isp", ["isp"], 0.35),
    ("network_schedule", ["network_schedule_type"], 0.35),
    ("bw", ["bw_bucket"], 0.35),
    ("packet_loss", ["packet_loss_satisfaction_bucket"], 0.35),
]

PROPENSITY_LEVELS = [
    (
        "province_isp_network_bw_loss",
        [
            "province", "isp", "network_schedule_type", "bw_bucket",
            "packet_loss_satisfaction_bucket",
        ],
    ),
    ("province_isp_network", ["province", "isp", "network_schedule_type"]),
    ("isp_network_bw", ["isp", "network_schedule_type", "bw_bucket"]),
    ("province_isp", ["province", "isp"]),
    ("isp_network", ["isp", "network_schedule_type"]),
    ("isp", ["isp"]),
]

NUMERIC_RANGE_FIELDS = [
    "bw",
    "corenum",
    "memtotal",
    "totaldisksize",
    "overall_packet_loss_benchmark_satisfaction_pct",
]


def clean(value: Any) -> str:
    text = v1.clean_cell(value)
    return text if text else UNKNOWN


def finite_number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def segment_keys(frame: pd.DataFrame, fields: list[str]) -> pd.Series:
    if not fields:
        return pd.Series("", index=frame.index, dtype="string")
    key = frame[fields[0]].fillna(UNKNOWN).astype(str)
    for field in fields[1:]:
        key = key + KEY_SEPARATOR + frame[field].fillna(UNKNOWN).astype(str)
    return key


def segment_key_from_row(row: dict[str, Any], fields: list[str]) -> str:
    return KEY_SEPARATOR.join(clean(row.get(field)) for field in fields)


def load_business_names(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {
        v1.clean_cell(row.business): v1.clean_cell(row.business_name)
        for row in frame.itertuples(index=False)
        if v1.clean_cell(row.business)
    }


def load_source_data_quality(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    source = json.loads(path.read_text(encoding="utf-8"))
    raw_node_days = int(source.get("raw_node_days", 0))
    clean_node_days = int(source.get("clean_node_days", 0))
    return {
        "source_version": source.get("version", ""),
        "candidate_large_nodes": int(source.get("candidate_large_nodes", 0)),
        "raw_rows": int(source.get("raw_rows", 0)),
        "raw_node_days": raw_node_days,
        "clean_node_days": clean_node_days,
        "excluded_node_days": max(raw_node_days - clean_node_days, 0),
        "clean_node_day_rate": clean_node_days / raw_node_days if raw_node_days else 0.0,
        "sampling_status_counts": source.get("sampling_status_counts", {}),
        "business_rules": source.get("business_rules", {}),
    }


def load_current_business(path: Path | None, business_names: dict[str, str]) -> pd.DataFrame:
    columns = [
        "node_id",
        "current_business",
        "current_business_name",
        "current_business_day",
        "current_business_source",
        "current_business_confidence",
        "current_business_cost_sum",
        "current_business_profit_sum",
    ]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype={"node_id": "string", "current_business": "string"}, low_memory=False)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[columns].copy()
    frame["node_id"] = frame["node_id"].map(v1.clean_cell)
    frame["current_business"] = frame["current_business"].map(v1.clean_cell)
    frame["current_business_name"] = frame["current_business_name"].fillna("").map(v1.clean_cell)
    missing_name = frame["current_business_name"].eq("")
    frame.loc[missing_name, "current_business_name"] = (
        frame.loc[missing_name, "current_business"].map(business_names).fillna("")
    )
    frame["_day"] = pd.to_datetime(frame["current_business_day"], errors="coerce")
    return (
        frame.sort_values(["node_id", "_day"])
        .drop_duplicates("node_id", keep="last")
        .drop(columns=["_day"])
    )


def enrich_profiles(frame: pd.DataFrame, profile_path: Path | None) -> pd.DataFrame:
    output = frame.copy()
    if profile_path is None or not profile_path.exists():
        return output
    header = pd.read_csv(profile_path, nrows=0).columns
    profile_columns = ["node_id"] + [field for field in PROFILE_FIELDS if field in header]
    profiles = pd.read_csv(
        profile_path,
        usecols=profile_columns,
        dtype={"node_id": "string"},
        low_memory=False,
    ).drop_duplicates("node_id")
    profiles = profiles.rename(columns={
        field: f"{field}__profile" for field in profile_columns if field != "node_id"
    })
    output = output.merge(profiles, on="node_id", how="left", sort=False)
    for field in PROFILE_FIELDS:
        profile_field = f"{field}__profile"
        if profile_field not in output.columns:
            continue
        if field not in output.columns:
            output[field] = output[profile_field]
        elif field in NUMERIC_RANGE_FIELDS:
            current = pd.to_numeric(output[field], errors="coerce")
            output[field] = current.fillna(pd.to_numeric(output[profile_field], errors="coerce"))
        else:
            current = output[field].fillna("").map(v1.clean_cell)
            output[field] = current.where(current.ne(""), output[profile_field])
        output = output.drop(columns=[profile_field])
    return output


def enrich_latest_pressure(frame: pd.DataFrame, pressure_path: Path | None) -> pd.DataFrame:
    """Override pressure quality with one latest measured value per node."""
    output = frame.copy()
    if pressure_path is None or not pressure_path.exists():
        return output
    header = pd.read_csv(pressure_path, nrows=0).columns
    columns = ["node_id"] + [field for field in LATEST_PRESSURE_FIELDS if field in header]
    if "overall_packet_loss_benchmark_satisfaction_pct" not in columns:
        raise RuntimeError(
            "latest pressure profile lacks overall_packet_loss_benchmark_satisfaction_pct"
        )
    pressure = pd.read_csv(
        pressure_path,
        usecols=columns,
        dtype={"node_id": "string"},
        low_memory=False,
    )
    pressure["node_id"] = pressure["node_id"].map(v1.clean_cell)
    sort_fields = [
        field for field in [
            "pressure_snapshot_day", "pressure_snapshot_hour", "pressure_last_report_time"
        ]
        if field in pressure.columns
    ]
    if sort_fields:
        pressure = pressure.sort_values(["node_id", *sort_fields])
    pressure = pressure.drop_duplicates("node_id", keep="last")
    pressure = pressure.rename(columns={
        field: f"{field}__latest_pressure"
        for field in columns
        if field != "node_id"
    })
    output = output.merge(pressure, on="node_id", how="left", sort=False)
    for field in LATEST_PRESSURE_FIELDS:
        latest = f"{field}__latest_pressure"
        if latest not in output.columns:
            continue
        # The explicitly fetched latest snapshot is authoritative, including 0%.
        output[field] = output[latest].where(output[latest].notna(), output.get(field))
        output = output.drop(columns=[latest])
    return output


def normalize_ipv6_capability(value: Any) -> str:
    text = v1.clean_cell(value).lower()
    if text in {"true", "1", "yes", "y", "支持", "enabled"}:
        return "支持"
    if text in {"false", "0", "no", "n", "不支持", "disabled"}:
        return "不支持"
    return UNKNOWN


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "node_id" not in output.columns:
        output["node_id"] = ""
    output["node_id"] = output["node_id"].map(v1.clean_cell)
    for field in PROFILE_FIELDS:
        if field not in output.columns:
            output[field] = np.nan if field in NUMERIC_RANGE_FIELDS else UNKNOWN
    for field in NUMERIC_RANGE_FIELDS:
        if field not in output.columns:
            output[field] = np.nan
    for field in NUMERIC_RANGE_FIELDS + ["analysis_transprovrate"]:
        output[field] = pd.to_numeric(output[field], errors="coerce")
    output = v2.add_buckets(output)
    output["ipv6_capability"] = output["analysis_issupportipv6"].map(
        normalize_ipv6_capability
    )
    output["packet_loss_satisfaction_bucket"] = output[
        "overall_packet_loss_benchmark_satisfaction_pct"
    ].map(lambda value: v2.bucket_value(value, [50, 70, 80, 90, 95]))
    model_fields = sorted({field for _, fields, _ in MODEL_LEVELS for field in fields})
    model_fields += [
        field for _, fields in PROPENSITY_LEVELS for field in fields if field not in model_fields
    ]
    for field in model_fields:
        if field not in output.columns:
            output[field] = UNKNOWN
        output[field] = output[field].fillna(UNKNOWN).map(clean)
    return output


def load_training_pairs(
    path: Path,
    profile_path: Path | None,
    latest_pressure_path: Path | None = None,
) -> pd.DataFrame:
    pairs = pd.read_csv(path, dtype={"node_id": "string", "business": "string"}, low_memory=False)
    pairs["node_id"] = pairs["node_id"].map(v1.clean_cell)
    pairs["business"] = pairs["business"].map(v1.clean_cell)
    pairs["sample_day"] = pd.to_datetime(pairs["sample_day"], errors="coerce")
    pairs["sample_weight"] = v1.sample_weights(pairs)
    bandwidth = v1.construction_bandwidth_mbps(pairs)
    pairs[TARGET_MINER] = pd.to_numeric(
        pairs.get("cost_per_bandwidth_mbps", pairs["cum_cost_7d"] / bandwidth),
        errors="coerce",
    )
    pairs[TARGET_PLATFORM] = pd.to_numeric(
        pairs.get(
            "profit_per_bandwidth_mbps",
            (pairs["cum_revenue_7d"] - pairs["cum_cost_7d"]) / bandwidth,
        ),
        errors="coerce",
    )
    pairs = pairs[
        pairs["sample_day"].notna()
        & pairs[TARGET_MINER].notna()
        & pairs[TARGET_PLATFORM].notna()
        & pairs["business"].ne("")
    ].copy()
    enriched = enrich_profiles(pairs, profile_path)
    enriched = enrich_latest_pressure(enriched, latest_pressure_path)
    return prepare_features(enriched)


def temporal_split(
    pairs: pd.DataFrame,
    validation_days: int = 7,
    calibration_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    dates = sorted(pd.Series(pairs["sample_day"].dropna().unique()).tolist())
    if len(dates) < validation_days + 2:
        raise RuntimeError("not enough unique dates for strict temporal validation")
    validation_dates = dates[-validation_days:]
    calibration_days = max(1, min(calibration_days, len(validation_dates) - 1))
    calibration_dates = set(validation_dates[:calibration_days])
    test_dates = set(validation_dates[calibration_days:])
    validation_set = set(validation_dates)
    train = pairs[~pairs["sample_day"].isin(validation_set)].copy()
    calibration = pairs[pairs["sample_day"].isin(calibration_dates)].copy()
    test = pairs[pairs["sample_day"].isin(test_dates)].copy()
    windows = {
        "train_start": pd.Timestamp(train["sample_day"].min()).strftime("%Y-%m-%d"),
        "train_end": pd.Timestamp(train["sample_day"].max()).strftime("%Y-%m-%d"),
        "calibration_start": pd.Timestamp(calibration["sample_day"].min()).strftime("%Y-%m-%d"),
        "calibration_end": pd.Timestamp(calibration["sample_day"].max()).strftime("%Y-%m-%d"),
        "test_start": pd.Timestamp(test["sample_day"].min()).strftime("%Y-%m-%d"),
        "test_end": pd.Timestamp(test["sample_day"].max()).strftime("%Y-%m-%d"),
    }
    return train, calibration, test, windows


def candidate_businesses(
    train: pd.DataFrame,
    min_effective_support: float,
    min_nodes: int,
) -> list[str]:
    support = (
        train.groupby("business", dropna=False)
        .agg(
            effective_support=("sample_weight", "sum"),
            nodes=("node_id", "nunique"),
        )
    )
    eligible = support[
        support["effective_support"].ge(min_effective_support)
        & support["nodes"].ge(min_nodes)
    ]
    return sorted(eligible.index.astype(str).tolist())


def weighted_stats(
    frame: pd.DataFrame,
    group_fields: list[str],
    targets: list[str],
) -> pd.DataFrame:
    work = frame.copy()
    work["_weight"] = v1.sample_weights(work)
    aggregations: dict[str, tuple[str, str]] = {
        "raw_support": ("_weight", "size"),
        "effective_support": ("_weight", "sum"),
        "node_support": ("node_id", "nunique"),
    }
    for target in targets:
        work[f"_{target}_sum"] = work[target] * work["_weight"]
        work[f"_{target}_sq_sum"] = work[target].pow(2) * work["_weight"]
        aggregations[f"_{target}_sum"] = (f"_{target}_sum", "sum")
        aggregations[f"_{target}_sq_sum"] = (f"_{target}_sq_sum", "sum")
    output = work.groupby(group_fields, dropna=False).agg(**aggregations).reset_index()
    for target in targets:
        output[f"{target}_mean"] = output[f"_{target}_sum"] / output["effective_support"]
        variance = (
            output[f"_{target}_sq_sum"] / output["effective_support"]
            - output[f"{target}_mean"].pow(2)
        ).clip(lower=0.0)
        output[f"{target}_std"] = np.sqrt(variance)
        output = output.drop(columns=[f"_{target}_sum", f"_{target}_sq_sum"])
    return output


def target_bounds(frame: pd.DataFrame, businesses: list[str]) -> dict[str, dict[str, list[float]]]:
    output: dict[str, dict[str, list[float]]] = {}
    for business, group in frame[frame["business"].isin(businesses)].groupby("business"):
        weights = v1.sample_weights(group)
        output[str(business)] = {}
        for target in [TARGET_MINER, TARGET_PLATFORM]:
            low = v1.weighted_quantile(group[target], weights, 0.01)
            high = v1.weighted_quantile(group[target], weights, 0.99)
            if not math.isfinite(low) or not math.isfinite(high) or high <= low:
                low = float(pd.to_numeric(group[target], errors="coerce").min())
                high = float(pd.to_numeric(group[target], errors="coerce").max())
            output[str(business)][target] = [float(low), float(high)]
    return output


def build_propensity_model(
    frame: pd.DataFrame,
    businesses: list[str],
    alpha: float = 20.0,
) -> dict[str, Any]:
    work = frame[frame["business"].isin(businesses)].copy()
    total_support = float(work["sample_weight"].sum())
    global_support = work.groupby("business")["sample_weight"].sum().to_dict()
    payload: dict[str, Any] = {
        "alpha": alpha,
        "global_rates": {
            business: float(global_support.get(business, 0.0)) / max(total_support, 1e-12)
            for business in businesses
        },
        "levels": [],
    }
    for name, fields in PROPENSITY_LEVELS:
        level_work = work[["business", "sample_weight", *fields]].copy()
        level_work["_key"] = segment_keys(level_work, fields)
        totals = level_work.groupby("_key")["sample_weight"].sum().to_dict()
        supports = level_work.groupby(["_key", "business"])["sample_weight"].sum()
        segments: dict[str, dict[str, Any]] = {}
        for (key, business), support in supports.items():
            item = segments.setdefault(str(key), {
                "total_support": float(totals.get(key, 0.0)),
                "business_support": {},
            })
            item["business_support"][str(business)] = float(support)
        payload["levels"].append({"name": name, "fields": fields, "segments": segments})
    return payload


def fit_model(
    frame: pd.DataFrame,
    businesses: list[str],
    business_names: dict[str, str],
    smoothing_alpha: float = 20.0,
    min_segment_support: float = 2.0,
    learning_rate: float = 0.70,
) -> dict[str, Any]:
    work = frame[frame["business"].isin(businesses)].copy()
    bounds = target_bounds(work, businesses)
    for target in [TARGET_MINER, TARGET_PLATFORM]:
        lows = work["business"].map({key: value[target][0] for key, value in bounds.items()})
        highs = work["business"].map({key: value[target][1] for key, value in bounds.items()})
        work[target] = work[target].clip(lower=lows, upper=highs)

    global_rows = weighted_stats(work, ["business"], [TARGET_MINER, TARGET_PLATFORM])
    business_models: dict[str, dict[str, Any]] = {}
    for row in global_rows.to_dict(orient="records"):
        business = str(row["business"])
        business_models[business] = {
            "business_name": business_names.get(business, ""),
            "raw_support": int(row["raw_support"]),
            "effective_support": float(row["effective_support"]),
            "node_support": int(row["node_support"]),
            "target_bounds": bounds[business],
            "global": {
                TARGET_MINER: float(row[f"{TARGET_MINER}_mean"]),
                TARGET_PLATFORM: float(row[f"{TARGET_PLATFORM}_mean"]),
                f"{TARGET_MINER}_std": float(row[f"{TARGET_MINER}_std"]),
                f"{TARGET_PLATFORM}_std": float(row[f"{TARGET_PLATFORM}_std"]),
            },
            "segments": {},
        }

    work["_miner_prediction"] = work["business"].map({
        business: info["global"][TARGET_MINER] for business, info in business_models.items()
    })
    work["_platform_prediction"] = work["business"].map({
        business: info["global"][TARGET_PLATFORM] for business, info in business_models.items()
    })

    level_metadata: list[dict[str, Any]] = []
    for name, fields, importance in MODEL_LEVELS:
        work["_segment_key"] = segment_keys(work, fields)
        work["_miner_residual"] = work[TARGET_MINER] - work["_miner_prediction"]
        work["_platform_residual"] = work[TARGET_PLATFORM] - work["_platform_prediction"]
        work["_weighted_miner_residual"] = work["_miner_residual"] * work["sample_weight"]
        work["_weighted_platform_residual"] = work["_platform_residual"] * work["sample_weight"]
        grouped = (
            work.groupby(["business", "_segment_key"], dropna=False)
            .agg(
                raw_support=("sample_weight", "size"),
                effective_support=("sample_weight", "sum"),
                node_support=("node_id", "nunique"),
                miner_residual_sum=("_weighted_miner_residual", "sum"),
                platform_residual_sum=("_weighted_platform_residual", "sum"),
            )
            .reset_index()
        )
        grouped["miner_adjustment"] = (
            learning_rate * grouped["miner_residual_sum"]
            / (grouped["effective_support"] + smoothing_alpha)
        )
        grouped["platform_adjustment"] = (
            learning_rate * grouped["platform_residual_sum"]
            / (grouped["effective_support"] + smoothing_alpha)
        )
        eligible = grouped["effective_support"].ge(min_segment_support)
        grouped = grouped[eligible].copy()
        miner_lookup: dict[str, float] = {}
        platform_lookup: dict[str, float] = {}
        for row in grouped.to_dict(orient="records"):
            business = str(row["business"])
            key = str(row["_segment_key"])
            combined_key = business + KEY_SEPARATOR + key
            miner_lookup[combined_key] = float(row["miner_adjustment"])
            platform_lookup[combined_key] = float(row["platform_adjustment"])
            business_models[business]["segments"].setdefault(name, {})[key] = {
                "effective_support": float(row["effective_support"]),
                "raw_support": int(row["raw_support"]),
                "node_support": int(row["node_support"]),
                "miner_adjustment": float(row["miner_adjustment"]),
                "platform_adjustment": float(row["platform_adjustment"]),
            }
        row_keys = work["business"] + KEY_SEPARATOR + work["_segment_key"]
        work["_miner_prediction"] += row_keys.map(miner_lookup).fillna(0.0)
        work["_platform_prediction"] += row_keys.map(platform_lookup).fillna(0.0)
        level_metadata.append({"name": name, "fields": fields, "importance": importance})

    known_fields = sorted({field for _, fields, _ in MODEL_LEVELS for field in fields})
    known_values = {
        field: sorted(work[field].dropna().astype(str).unique().tolist())
        for field in known_fields
    }
    numeric_ranges: dict[str, dict[str, float]] = {}
    for field in NUMERIC_RANGE_FIELDS:
        values = pd.to_numeric(work[field], errors="coerce").dropna()
        if not values.empty:
            numeric_ranges[field] = {
                "p01": float(values.quantile(0.01)),
                "p99": float(values.quantile(0.99)),
            }

    return {
        "version": "v4",
        "model_type": "per_business_hierarchical_outcome_regressor",
        "objective": (
            "rank by 0.5 * normalized predicted miner unit income + "
            "0.5 * normalized predicted platform unit profit"
        ),
        "candidate_businesses": businesses,
        "businesses": business_models,
        "segment_levels": level_metadata,
        "smoothing_alpha": smoothing_alpha,
        "min_segment_support": min_segment_support,
        "learning_rate": learning_rate,
        "sample_weight_policy": "1 / consecutive valid days in the same node-business run",
        "known_values": known_values,
        "numeric_ranges": numeric_ranges,
        "propensity": build_propensity_model(work, businesses),
        "calibration": {},
        "validation_by_business": {},
    }


def propensity_for(row: dict[str, Any], business: str, model: dict[str, Any]) -> tuple[float, str, float]:
    payload = model["propensity"]
    prior = float(payload["global_rates"].get(business, 0.0))
    alpha = float(payload.get("alpha", 20.0))
    for level in payload.get("levels", []):
        key = segment_key_from_row(row, level["fields"])
        segment = level["segments"].get(key)
        if not segment:
            continue
        total = float(segment.get("total_support", 0.0))
        if total < 10:
            continue
        support = float(segment.get("business_support", {}).get(business, 0.0))
        probability = (support + alpha * prior) / (total + alpha)
        return probability, level["name"], total
    return prior, "global", float(sum(
        info.get("effective_support", 0.0) for info in model["businesses"].values()
    ))


def ood_reasons(row: dict[str, Any], model: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field, known in model.get("known_values", {}).items():
        value = clean(row.get(field))
        if value != UNKNOWN and value not in set(known):
            reasons.append(f"{field}={value}未在训练集出现")
    for field, limits in model.get("numeric_ranges", {}).items():
        value = finite_number(row.get(field), float("nan"))
        if math.isfinite(value) and (value < limits["p01"] or value > limits["p99"]):
            reasons.append(f"{field}超出训练P01-P99")
    return reasons


def predict_business(
    row: dict[str, Any],
    business: str,
    model: dict[str, Any],
    row_ood_reasons: list[str] | None = None,
) -> dict[str, Any]:
    info = model["businesses"][business]
    miner = float(info["global"][TARGET_MINER])
    platform = float(info["global"][TARGET_PLATFORM])
    matched: list[tuple[str, float, int, float]] = []
    for level in model.get("segment_levels", []):
        key = segment_key_from_row(row, level["fields"])
        segment = info.get("segments", {}).get(level["name"], {}).get(key)
        if not segment:
            continue
        miner += float(segment.get("miner_adjustment", 0.0))
        platform += float(segment.get("platform_adjustment", 0.0))
        matched.append((
            level["name"],
            float(segment.get("effective_support", 0.0)),
            int(segment.get("node_support", 0)),
            float(level.get("importance", 1.0)),
        ))
    miner_bounds = info["target_bounds"][TARGET_MINER]
    platform_bounds = info["target_bounds"][TARGET_PLATFORM]
    miner = float(np.clip(miner, miner_bounds[0], miner_bounds[1]))
    platform = float(np.clip(platform, platform_bounds[0], platform_bounds[1]))
    matched_sorted = sorted(matched, key=lambda item: (-item[3], -item[1]))
    coverage_level = matched_sorted[0][0] if matched_sorted else "global"
    local_support = matched_sorted[0][1] if matched_sorted else 0.0
    local_nodes = matched_sorted[0][2] if matched_sorted else 0
    propensity, propensity_source, propensity_total = propensity_for(row, business, model)
    calibration = model.get("calibration", {}).get(business, {})
    support_scale = math.sqrt(1.0 + 5.0 / max(local_support + 5.0, 5.0))
    miner_radius80 = float(calibration.get("miner_radius80", info["global"][f"{TARGET_MINER}_std"])) * support_scale
    miner_radius90 = float(calibration.get("miner_radius90", info["global"][f"{TARGET_MINER}_std"])) * support_scale
    platform_radius80 = float(calibration.get("platform_radius80", info["global"][f"{TARGET_PLATFORM}_std"])) * support_scale
    platform_radius90 = float(calibration.get("platform_radius90", info["global"][f"{TARGET_PLATFORM}_std"])) * support_scale
    ood = row_ood_reasons if row_ood_reasons is not None else ood_reasons(row, model)
    validation = model.get("validation_by_business", {}).get(business, {})
    miner_span = max(miner_bounds[1] - miner_bounds[0], 1e-9)
    platform_span = max(platform_bounds[1] - platform_bounds[0], 1e-9)
    uncertainty_ratio = 0.5 * (
        miner_radius90 / miner_span + platform_radius90 / platform_span
    )
    validation_support = float(validation.get("test_effective_support", 0.0))
    coverage90 = float(validation.get("joint_average_coverage90", 0.0))
    if (
        local_support >= 30
        and info["effective_support"] >= 100
        and propensity >= 0.03
        and validation_support >= 20
        and 0.78 <= coverage90 <= 0.99
        and uncertainty_ratio <= 0.75
        and not ood
    ):
        confidence = "high"
    elif (
        local_support >= 10
        and info["effective_support"] >= 30
        and propensity >= 0.01
        and validation_support >= 5
        and uncertainty_ratio <= 1.5
        and len(ood) <= 1
    ):
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "business": business,
        "business_name": info.get("business_name", ""),
        TARGET_MINER: miner,
        TARGET_PLATFORM: platform,
        "miner_low80": miner - miner_radius80,
        "miner_high80": miner + miner_radius80,
        "miner_low90": miner - miner_radius90,
        "miner_high90": miner + miner_radius90,
        "platform_low80": platform - platform_radius80,
        "platform_high80": platform + platform_radius80,
        "platform_low90": platform - platform_radius90,
        "platform_high90": platform + platform_radius90,
        "global_effective_support": float(info["effective_support"]),
        "global_node_support": int(info["node_support"]),
        "local_effective_support": local_support,
        "local_node_support": local_nodes,
        "coverage_level": coverage_level,
        "matched_level_count": len(matched),
        "propensity": propensity,
        "propensity_source": propensity_source,
        "propensity_segment_support": propensity_total,
        "ood_reasons": ood,
        "uncertainty_ratio": uncertainty_ratio,
        "confidence": confidence,
    }


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if not math.isfinite(high - low) or high - low < 1e-12:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def score_candidates(row: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
    row_ood = ood_reasons(row, model)
    predictions = [
        predict_business(row, business, model, row_ood_reasons=row_ood)
        for business in model["candidate_businesses"]
    ]
    miner_scores = minmax([item[TARGET_MINER] for item in predictions])
    platform_scores = minmax([item[TARGET_PLATFORM] for item in predictions])
    miner_low_scores = minmax([item["miner_low90"] for item in predictions])
    miner_high_scores = minmax([item["miner_high90"] for item in predictions])
    platform_low_scores = minmax([item["platform_low90"] for item in predictions])
    platform_high_scores = minmax([item["platform_high90"] for item in predictions])
    for item, miner_score, platform_score, miner_low, miner_high, platform_low, platform_high in zip(
        predictions,
        miner_scores,
        platform_scores,
        miner_low_scores,
        miner_high_scores,
        platform_low_scores,
        platform_high_scores,
    ):
        item["miner_score"] = miner_score
        item["platform_score"] = platform_score
        item["combined_score"] = 0.5 * miner_score + 0.5 * platform_score
        item["combined_low90"] = 0.5 * miner_low + 0.5 * platform_low
        item["combined_high90"] = 0.5 * miner_high + 0.5 * platform_high
    return sorted(predictions, key=lambda item: (-item["combined_score"], item["business"]))


def predict_observed(frame: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        business = str(row["business"])
        if business not in model["businesses"]:
            continue
        prediction = predict_business(row, business, model)
        rows.append({
            "node_id": row["node_id"],
            "sample_day": pd.Timestamp(row["sample_day"]).strftime("%Y-%m-%d"),
            "business": business,
            "business_name": prediction["business_name"],
            "sample_weight": float(row["sample_weight"]),
            "actual_miner_unit_income": float(row[TARGET_MINER]),
            "actual_platform_unit_profit": float(row[TARGET_PLATFORM]),
            "predicted_miner_unit_income": prediction[TARGET_MINER],
            "predicted_platform_unit_profit": prediction[TARGET_PLATFORM],
            "local_effective_support": prediction["local_effective_support"],
            "propensity": prediction["propensity"],
            "confidence": prediction["confidence"],
            "miner_low90": prediction["miner_low90"],
            "miner_high90": prediction["miner_high90"],
            "platform_low90": prediction["platform_low90"],
            "platform_high90": prediction["platform_high90"],
        })
    return pd.DataFrame(rows)


def attach_calibration(
    model: dict[str, Any],
    calibration_rows: pd.DataFrame,
    minimum_support: float = 8.0,
) -> dict[str, Any]:
    observed = predict_observed(calibration_rows, model)
    if observed.empty:
        return model
    observed["miner_scaled_residual"] = (
        observed["actual_miner_unit_income"] - observed["predicted_miner_unit_income"]
    ).abs() / np.sqrt(1.0 + 5.0 / (observed["local_effective_support"] + 5.0).clip(lower=5.0))
    observed["platform_scaled_residual"] = (
        observed["actual_platform_unit_profit"] - observed["predicted_platform_unit_profit"]
    ).abs() / np.sqrt(1.0 + 5.0 / (observed["local_effective_support"] + 5.0).clip(lower=5.0))
    weights = v1.sample_weights(observed)
    global_values = {
        "miner_radius80": v1.weighted_quantile(observed["miner_scaled_residual"], weights, 0.80),
        "miner_radius90": v1.weighted_quantile(observed["miner_scaled_residual"], weights, 0.90),
        "platform_radius80": v1.weighted_quantile(observed["platform_scaled_residual"], weights, 0.80),
        "platform_radius90": v1.weighted_quantile(observed["platform_scaled_residual"], weights, 0.90),
    }
    payload: dict[str, Any] = {}
    for business in model["candidate_businesses"]:
        group = observed[observed["business"].eq(business)]
        effective_support = float(group["sample_weight"].sum()) if not group.empty else 0.0
        item = dict(global_values)
        item["source"] = "global_calibration"
        item["effective_support"] = effective_support
        if effective_support >= minimum_support:
            group_weights = v1.sample_weights(group)
            item.update({
                "miner_radius80": v1.weighted_quantile(group["miner_scaled_residual"], group_weights, 0.80),
                "miner_radius90": v1.weighted_quantile(group["miner_scaled_residual"], group_weights, 0.90),
                "platform_radius80": v1.weighted_quantile(group["platform_scaled_residual"], group_weights, 0.80),
                "platform_radius90": v1.weighted_quantile(group["platform_scaled_residual"], group_weights, 0.90),
                "source": "business_calibration",
            })
        payload[business] = {key: float(value) if isinstance(value, (np.floating, float)) else value for key, value in item.items()}
    model["calibration"] = payload
    return model


def weighted_metrics(actual: pd.Series, predicted: pd.Series, weights: pd.Series) -> dict[str, float]:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    weight_values = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values) & (weight_values > 0)
    if not valid.any():
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "bias": 0.0}
    actual_values = actual_values[valid]
    predicted_values = predicted_values[valid]
    weight_values = weight_values[valid]
    residual = predicted_values - actual_values
    mean_actual = float(np.average(actual_values, weights=weight_values))
    ss_res = float(np.sum(weight_values * residual ** 2))
    ss_total = float(np.sum(weight_values * (actual_values - mean_actual) ** 2))
    return {
        "mae": float(np.average(np.abs(residual), weights=weight_values)),
        "rmse": math.sqrt(ss_res / float(weight_values.sum())),
        "r2": 1.0 - ss_res / ss_total if ss_total > 1e-12 else 0.0,
        "bias": float(np.average(residual, weights=weight_values)),
    }


def evaluate_temporal(
    test: pd.DataFrame,
    model: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    observed = predict_observed(test, model)
    if observed.empty:
        raise RuntimeError("no temporal test rows match candidate businesses")
    weights = v1.sample_weights(observed)
    miner_metrics = weighted_metrics(
        observed["actual_miner_unit_income"],
        observed["predicted_miner_unit_income"],
        weights,
    )
    platform_metrics = weighted_metrics(
        observed["actual_platform_unit_profit"],
        observed["predicted_platform_unit_profit"],
        weights,
    )
    miner_covered = observed["actual_miner_unit_income"].between(
        observed["miner_low90"], observed["miner_high90"]
    )
    platform_covered = observed["actual_platform_unit_profit"].between(
        observed["platform_low90"], observed["platform_high90"]
    )
    metrics: dict[str, Any] = {
        "raw_rows": int(len(observed)),
        "effective_support": float(weights.sum()),
        "unique_nodes": int(observed["node_id"].nunique()),
        "miner": miner_metrics,
        "platform": platform_metrics,
        "miner_interval_coverage90": float(np.average(miner_covered, weights=weights)),
        "platform_interval_coverage90": float(np.average(platform_covered, weights=weights)),
    }

    business_rows: list[dict[str, Any]] = []
    validation_lookup: dict[str, dict[str, Any]] = {}
    for business, group in observed.groupby("business"):
        group_weights = v1.sample_weights(group)
        miner = weighted_metrics(
            group["actual_miner_unit_income"], group["predicted_miner_unit_income"], group_weights
        )
        platform = weighted_metrics(
            group["actual_platform_unit_profit"], group["predicted_platform_unit_profit"], group_weights
        )
        miner_coverage = float(np.average(
            group["actual_miner_unit_income"].between(group["miner_low90"], group["miner_high90"]),
            weights=group_weights,
        ))
        platform_coverage = float(np.average(
            group["actual_platform_unit_profit"].between(group["platform_low90"], group["platform_high90"]),
            weights=group_weights,
        ))
        item = {
            "business": str(business),
            "business_name": group["business_name"].iloc[0],
            "test_rows": int(len(group)),
            "test_effective_support": float(group_weights.sum()),
            "test_nodes": int(group["node_id"].nunique()),
            "miner_mae": miner["mae"],
            "miner_rmse": miner["rmse"],
            "miner_r2": miner["r2"],
            "platform_mae": platform["mae"],
            "platform_rmse": platform["rmse"],
            "platform_r2": platform["r2"],
            "miner_coverage90": miner_coverage,
            "platform_coverage90": platform_coverage,
            "joint_average_coverage90": 0.5 * (miner_coverage + platform_coverage),
        }
        business_rows.append(item)
        validation_lookup[str(business)] = item
    model["validation_by_business"] = validation_lookup
    return observed, metrics, business_rows


def evaluate_ranking_and_dr(
    test: pd.DataFrame,
    model: dict[str, Any],
) -> dict[str, Any]:
    work = test[test["business"].isin(model["candidate_businesses"])].copy()
    cache: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    signature_fields = sorted({field for _, fields, _ in MODEL_LEVELS for field in fields})
    hit1: list[bool] = []
    hit3: list[bool] = []
    weights: list[float] = []
    top1_distribution: dict[str, float] = {}
    dm_miner: list[float] = []
    dm_platform: list[float] = []
    dr_miner: list[float] = []
    dr_platform: list[float] = []
    observed_miner: list[float] = []
    observed_platform: list[float] = []
    correction_weights: list[float] = []
    matched_weight = 0.0
    for row in work.to_dict(orient="records"):
        signature = tuple(clean(row.get(field)) for field in signature_fields)
        ranked = cache.get(signature)
        if ranked is None:
            ranked = score_candidates(row, model)
            cache[signature] = ranked
        policy = ranked[0]
        actual_business = str(row["business"])
        weight = float(row["sample_weight"])
        top_ids = [item["business"] for item in ranked[:3]]
        hit1.append(top_ids[0] == actual_business)
        hit3.append(actual_business in set(top_ids))
        weights.append(weight)
        top1_distribution[top_ids[0]] = top1_distribution.get(top_ids[0], 0.0) + weight
        observed_prediction = predict_business(row, actual_business, model)
        probability = max(float(observed_prediction["propensity"]), 0.05)
        match = top_ids[0] == actual_business
        correction = (1.0 / probability) if match else 0.0
        actual_miner = float(row[TARGET_MINER])
        actual_platform = float(row[TARGET_PLATFORM])
        dm_miner.append(policy[TARGET_MINER])
        dm_platform.append(policy[TARGET_PLATFORM])
        dr_miner.append(
            policy[TARGET_MINER]
            + correction * (actual_miner - observed_prediction[TARGET_MINER])
        )
        dr_platform.append(
            policy[TARGET_PLATFORM]
            + correction * (actual_platform - observed_prediction[TARGET_PLATFORM])
        )
        observed_miner.append(actual_miner)
        observed_platform.append(actual_platform)
        correction_weights.append(weight * correction)
        if match:
            matched_weight += weight
    weight_array = np.asarray(weights, dtype=float)
    correction_array = np.asarray(correction_weights, dtype=float)
    correction_ess = (
        float(correction_array.sum() ** 2 / np.sum(correction_array ** 2))
        if np.sum(correction_array ** 2) > 0 else 0.0
    )
    total = float(weight_array.sum())
    distribution = [
        {"business": business, "effective_top1": support, "share": support / max(total, 1e-12)}
        for business, support in sorted(top1_distribution.items(), key=lambda item: -item[1])
    ]
    return {
        "evaluated_rows": int(len(work)),
        "effective_support": total,
        "unique_profile_signatures": len(cache),
        "observed_business_hit_at_1": float(np.average(hit1, weights=weight_array)),
        "observed_business_hit_at_3": float(np.average(hit3, weights=weight_array)),
        "top1_distribution": distribution,
        "direct_method": {
            "miner_unit_income": float(np.average(dm_miner, weights=weight_array)),
            "platform_unit_profit": float(np.average(dm_platform, weights=weight_array)),
        },
        "doubly_robust": {
            "miner_unit_income": float(np.average(dr_miner, weights=weight_array)),
            "platform_unit_profit": float(np.average(dr_platform, weights=weight_array)),
            "policy_observed_match_rate": matched_weight / max(total, 1e-12),
            "correction_effective_sample_size": correction_ess,
            "propensity_floor": 0.05,
        },
        "observed_policy": {
            "miner_unit_income": float(np.average(observed_miner, weights=weight_array)),
            "platform_unit_profit": float(np.average(observed_platform, weights=weight_array)),
        },
        "warning": (
            "DR is an observational bias-correction diagnostic, not causal proof; it assumes all important "
            "historical allocation factors are present in the feature set."
        ),
    }


def confidence_reason(item: dict[str, Any]) -> str:
    reasons = [
        f"本地有效支持={item['local_effective_support']:.1f}",
        f"历史分配概率={item['propensity']:.1%}",
        f"匹配层级={item['coverage_level']}",
    ]
    if item["ood_reasons"]:
        reasons.append("超范围=" + "、".join(item["ood_reasons"][:2]))
    if item[TARGET_PLATFORM] < 0:
        reasons.append("预计平台单位利润为负")
    elif item["platform_low90"] < 0:
        reasons.append("平台利润90%区间包含负值")
    return "；".join(reasons)


def recommendation_action(top_item: dict[str, Any], score_gap: float) -> str:
    if (
        top_item["confidence"] == "low"
        or score_gap < 0.03
        or float(top_item[TARGET_PLATFORM]) <= 0
    ):
        return "暂不推荐执行"
    if float(top_item["platform_low90"]) < 0:
        return "谨慎人工复核"
    if top_item["confidence"] == "high" and score_gap >= 0.08:
        return "优先人工复核"
    return "谨慎人工复核"


def recommend_nodes(nodes: pd.DataFrame, model: dict[str, Any], top_k: int = 3) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in nodes.drop_duplicates("node_id").to_dict(orient="records"):
        ranked = score_candidates(row, model)
        selected = ranked[:top_k]
        if not selected:
            continue
        score_gap = (
            selected[0]["combined_score"] - selected[1]["combined_score"]
            if len(selected) > 1 else 1.0
        )
        top_confidence = selected[0]["confidence"]
        action = recommendation_action(selected[0], score_gap)
        bandwidth = max(finite_number(row.get("bw"), 0.0), 0.0)
        current_business = v1.clean_cell(row.get("current_business"))
        current_business_name = v1.clean_cell(row.get("current_business_name"))
        candidate_lookup = {item["business"]: item for item in ranked}
        current_prediction = candidate_lookup.get(current_business)
        if not current_business:
            current_status = "current_business_unknown"
        elif current_prediction is None:
            current_status = "current_business_not_in_candidate_set"
        elif current_business == selected[0]["business"]:
            current_status = "already_top1"
        else:
            current_status = "current_not_top1"
        model_current_not_top1 = current_status == "current_not_top1"
        review_switch_candidate = model_current_not_top1 and action != "暂不推荐执行"
        current_cost = finite_number(row.get("current_business_cost_sum"), float("nan"))
        current_profit = finite_number(row.get("current_business_profit_sum"), float("nan"))
        output: dict[str, Any] = {
            "node_id": row["node_id"],
            "province": clean(row.get("province")),
            "city": clean(row.get("city")),
            "isp": clean(row.get("isp")),
            "resourcetype": clean(row.get("resourcetype")),
            "deliverytype": clean(row.get("deliverytype")),
            "nattype": clean(row.get("nattype")),
            "scheduleisps": clean(row.get("scheduleisps")),
            "network_schedule_type": clean(row.get("network_schedule_type")),
            "cpu_bucket": clean(row.get("corenum_bucket")),
            "memory_bucket": clean(row.get("memtotal_bucket")),
            "disk_bucket": clean(row.get("totaldisksize_bucket")),
            "ipv6_capability": clean(row.get("ipv6_capability")),
            "bandwidth_bucket": clean(row.get("bw_bucket")),
            "packet_loss_satisfaction_pct": finite_number(
                row.get("overall_packet_loss_benchmark_satisfaction_pct"), float("nan")
            ),
            "packet_loss_satisfaction_bucket": clean(
                row.get("packet_loss_satisfaction_bucket")
            ),
            "build_bandwidth_mbps": bandwidth,
            "recommendation_action": action,
            "recommendation_confidence": top_confidence,
            "top1_score_gap": score_gap,
            "recommendation_reason": confidence_reason(selected[0]),
            "current_business": current_business,
            "current_business_name": current_business_name,
            "current_business_day": v1.clean_cell(row.get("current_business_day")),
            "current_business_source": v1.clean_cell(row.get("current_business_source")),
            "current_business_confidence": v1.clean_cell(row.get("current_business_confidence")),
            "current_business_status": current_status,
            "current_business_in_model": current_prediction is not None,
            "model_current_not_top1": model_current_not_top1,
            "review_switch_candidate": review_switch_candidate,
            "current_actual_miner_income_1d": current_cost,
            "current_actual_platform_profit_1d": current_profit,
            "current_actual_miner_unit_income": current_cost / bandwidth if bandwidth > 0 else float("nan"),
            "current_actual_platform_unit_profit": current_profit / bandwidth if bandwidth > 0 else float("nan"),
            "current_predicted_combined_score": (
                current_prediction["combined_score"] if current_prediction else float("nan")
            ),
            "current_predicted_miner_unit_income": (
                current_prediction[TARGET_MINER] if current_prediction else float("nan")
            ),
            "current_predicted_platform_unit_profit": (
                current_prediction[TARGET_PLATFORM] if current_prediction else float("nan")
            ),
            "top1_vs_current_predicted_miner_unit_delta": (
                selected[0][TARGET_MINER] - current_prediction[TARGET_MINER]
                if current_prediction else float("nan")
            ),
            "top1_vs_current_predicted_platform_unit_delta": (
                selected[0][TARGET_PLATFORM] - current_prediction[TARGET_PLATFORM]
                if current_prediction else float("nan")
            ),
        }
        details: list[dict[str, Any]] = []
        for index, item in enumerate(selected, 1):
            prefix = f"top{index}"
            output.update({
                f"business_{prefix}": item["business"],
                f"business_name_{prefix}": item["business_name"],
                f"combined_score_{prefix}": item["combined_score"],
                f"miner_unit_income_{prefix}": item[TARGET_MINER],
                f"platform_unit_profit_{prefix}": item[TARGET_PLATFORM],
                f"estimated_miner_income_1d_{prefix}": item[TARGET_MINER] * bandwidth,
                f"estimated_platform_profit_1d_{prefix}": item[TARGET_PLATFORM] * bandwidth,
                f"miner_unit_low90_{prefix}": item["miner_low90"],
                f"miner_unit_high90_{prefix}": item["miner_high90"],
                f"platform_unit_low90_{prefix}": item["platform_low90"],
                f"platform_unit_high90_{prefix}": item["platform_high90"],
                f"confidence_{prefix}": item["confidence"],
                f"local_effective_support_{prefix}": item["local_effective_support"],
                f"global_effective_support_{prefix}": item["global_effective_support"],
                f"historical_assignment_probability_{prefix}": item["propensity"],
                f"coverage_level_{prefix}": item["coverage_level"],
                f"reason_{prefix}": confidence_reason(item),
            })
            details.append({key: value for key, value in item.items() if key != "ood_reasons"})
        output["top3_detail"] = json.dumps(details, ensure_ascii=False, separators=(",", ":"))
        rows.append(output)
    return pd.DataFrame(rows)


def concentration_diagnostics(recommendations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = (
        recommendations.groupby(["business_top1", "business_name_top1"], dropna=False)
        .size()
        .rename("node_count")
        .reset_index()
        .sort_values(["node_count", "business_top1"], ascending=[False, True])
    )
    total = max(int(counts["node_count"].sum()), 1)
    counts["share"] = counts["node_count"] / total
    shares = counts["share"].to_numpy(dtype=float)
    hhi = float(np.sum(shares ** 2))
    entropy = float(-np.sum(shares[shares > 0] * np.log(shares[shares > 0])))
    normalized_entropy = entropy / math.log(len(shares)) if len(shares) > 1 else 0.0
    maximum_share = float(shares.max()) if len(shares) else 0.0
    if maximum_share > 0.70 or hhi > 0.50:
        level = "high"
    elif maximum_share > 0.50 or hhi > 0.30:
        level = "medium"
    else:
        level = "low"
    return counts, {
        "top1_businesses_used": int(len(counts)),
        "maximum_top1_share": maximum_share,
        "hhi": hhi,
        "effective_business_count": 1.0 / hhi if hhi > 0 else 0.0,
        "normalized_entropy": normalized_entropy,
        "concentration_risk": level,
    }


def training_business_summary(
    pairs: pd.DataFrame,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    candidates: list[str],
    business_names: dict[str, str],
    validation_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    validation = {row["business"]: row for row in validation_rows}
    rows: list[dict[str, Any]] = []
    for business, group in pairs.groupby("business"):
        business = str(business)
        weights = v1.sample_weights(group)
        item: dict[str, Any] = {
            "business": business,
            "business_name": business_names.get(business, ""),
            "is_candidate": business in candidates,
            "raw_node_days": int(len(group)),
            "effective_run_support": float(weights.sum()),
            "nodes": int(group["node_id"].nunique()),
            "sample_day_min": pd.Timestamp(group["sample_day"].min()).strftime("%Y-%m-%d"),
            "sample_day_max": pd.Timestamp(group["sample_day"].max()).strftime("%Y-%m-%d"),
            "weighted_miner_unit_income": float(np.average(group[TARGET_MINER], weights=weights)),
            "weighted_platform_unit_profit": float(np.average(group[TARGET_PLATFORM], weights=weights)),
        }
        for label, split in [("train", train), ("calibration", calibration), ("test", test)]:
            selected = split[split["business"].eq(business)]
            item[f"{label}_rows"] = int(len(selected))
            item[f"{label}_effective_support"] = float(selected["sample_weight"].sum())
        item.update(validation.get(business, {}))
        rows.append(item)
    return pd.DataFrame(rows).sort_values(
        ["is_candidate", "effective_run_support", "business"],
        ascending=[False, False, True],
    )


def credibility_summary(
    temporal: dict[str, Any],
    ranking_dr: dict[str, Any],
    concentration: dict[str, Any],
    recommendations: pd.DataFrame,
    business_metrics: pd.DataFrame,
) -> dict[str, Any]:
    coverage_ok = (
        0.80 <= temporal["miner_interval_coverage90"] <= 0.98
        and 0.80 <= temporal["platform_interval_coverage90"] <= 0.98
    )
    r2_ok = temporal["miner"]["r2"] > 0 and temporal["platform"]["r2"] > 0
    concentration_ok = concentration["concentration_risk"] == "low"
    top3_ok = ranking_dr["observed_business_hit_at_3"] >= 0.20
    if coverage_ok and r2_ok and concentration_ok and top3_ok:
        level = "medium"
    else:
        level = "low"
    validated = business_metrics[
        business_metrics["is_candidate"].astype(bool)
        & business_metrics["test_effective_support"].fillna(0).gt(0)
    ]
    both_positive = validated[
        validated["miner_r2"].fillna(float("-inf")).gt(0)
        & validated["platform_r2"].fillna(float("-inf")).gt(0)
    ]
    platform_point_negative = recommendations["platform_unit_profit_top1"].le(0)
    platform_interval_crosses_zero = recommendations["platform_unit_low90_top1"].lt(0)
    return {
        "level": level,
        "automated_switching_allowed": False,
        "coverage_calibrated": coverage_ok,
        "positive_out_of_time_r2_for_both_targets": r2_ok,
        "recommendation_concentration_acceptable": concentration_ok,
        "observed_business_top3_at_least_20pct": top3_ok,
        "validated_candidate_businesses": int(len(validated)),
        "validated_businesses_positive_r2_for_both_targets": int(len(both_positive)),
        "top1_platform_point_nonpositive_nodes": int(platform_point_negative.sum()),
        "top1_platform_interval_crosses_zero_nodes": int(platform_interval_crosses_zero.sum()),
        "confidence_distribution": recommendations["recommendation_confidence"].value_counts().to_dict(),
        "action_distribution": recommendations["recommendation_action"].value_counts().to_dict(),
        "interpretation": (
            "Intervals are validated only for historically observed businesses. Unobserved business outcomes "
            "remain counterfactual estimates and require human review or prospective validation."
        ),
    }


def current_business_comparison_summary(recommendations: pd.DataFrame) -> dict[str, Any]:
    status_counts = recommendations["current_business_status"].value_counts().to_dict()
    return {
        "snapshot_nodes": int(recommendations["current_business"].fillna("").ne("").sum()),
        "snapshot_day_counts": recommendations.loc[
            recommendations["current_business_day"].fillna("").ne(""), "current_business_day"
        ].value_counts().to_dict(),
        "status_counts": status_counts,
        "current_business_in_model_nodes": int(recommendations["current_business_in_model"].sum()),
        "model_current_not_top1_nodes": int(recommendations["model_current_not_top1"].sum()),
        "review_switch_candidate_nodes": int(recommendations["review_switch_candidate"].sum()),
    }


def write_report_data(
    output_path: Path,
    summary: dict[str, Any],
    recommendations: pd.DataFrame,
    business_metrics: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    report_columns = [
        "node_id", "province", "city", "isp", "resourcetype", "deliverytype", "nattype",
        "scheduleisps", "network_schedule_type", "build_bandwidth_mbps",
        "cpu_bucket", "memory_bucket", "disk_bucket", "ipv6_capability", "bandwidth_bucket",
        "packet_loss_satisfaction_pct", "packet_loss_satisfaction_bucket",
        "recommendation_action", "recommendation_confidence", "top1_score_gap",
        "recommendation_reason",
        "current_business", "current_business_name", "current_business_day",
        "current_business_source", "current_business_confidence", "current_business_status",
        "current_business_in_model", "model_current_not_top1", "interval_dominates_current",
        "review_switch_candidate",
        "current_actual_miner_income_1d", "current_actual_platform_profit_1d",
        "current_actual_miner_unit_income", "current_actual_platform_unit_profit",
        "current_predicted_combined_score", "current_predicted_miner_unit_income",
        "current_predicted_platform_unit_profit", "top1_vs_current_predicted_miner_unit_delta",
        "top1_vs_current_predicted_platform_unit_delta",
    ]
    for index in range(1, 4):
        report_columns.extend([
            f"business_top{index}", f"business_name_top{index}", f"combined_score_top{index}",
            f"miner_unit_income_top{index}", f"platform_unit_profit_top{index}",
            f"estimated_miner_income_1d_top{index}", f"estimated_platform_profit_1d_top{index}",
            f"miner_unit_low90_top{index}", f"miner_unit_high90_top{index}",
            f"platform_unit_low90_top{index}", f"platform_unit_high90_top{index}",
            f"confidence_top{index}", f"local_effective_support_top{index}",
            f"global_effective_support_top{index}", f"historical_assignment_probability_top{index}",
            f"coverage_level_top{index}", f"reason_top{index}",
        ])
    payload = {
        "summary": summary,
        "recommendations": recommendations[[column for column in report_columns if column in recommendations]].replace({np.nan: None}).to_dict(orient="records"),
        "business_metrics": business_metrics.replace({np.nan: None}).to_dict(orient="records"),
        "concentration": concentration.replace({np.nan: None}).to_dict(orient="records"),
    }
    output_path.write_text(
        "window.V4_REPORT_DATA=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str) + ";\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    business_names = load_business_names(args.business_map)
    pairs = load_training_pairs(
        args.pairs,
        args.historical_profiles,
        args.latest_pressure_profiles,
    )
    train, calibration, test, windows = temporal_split(
        pairs, args.validation_days, args.calibration_days
    )
    candidates = candidate_businesses(
        train, args.min_business_support, args.min_business_nodes
    )
    if len(candidates) < 3:
        raise RuntimeError("fewer than three businesses meet the V4 support thresholds")

    evaluation_model = fit_model(
        train,
        candidates,
        business_names,
        smoothing_alpha=args.smoothing_alpha,
        min_segment_support=args.min_segment_support,
        learning_rate=args.learning_rate,
    )
    attach_calibration(evaluation_model, calibration)
    validation_predictions, temporal_metrics, business_validation = evaluate_temporal(
        test, evaluation_model
    )
    ranking_dr = evaluate_ranking_and_dr(test, evaluation_model)

    final_model = fit_model(
        pairs,
        candidates,
        business_names,
        smoothing_alpha=args.smoothing_alpha,
        min_segment_support=args.min_segment_support,
        learning_rate=args.learning_rate,
    )
    final_model["calibration"] = evaluation_model["calibration"]
    final_model["validation_by_business"] = evaluation_model["validation_by_business"]
    final_model["temporal_windows"] = windows

    current_nodes = pd.read_csv(args.current_nodes, dtype={"node_id": "string"}, low_memory=False)
    current_business = load_current_business(args.current_business, business_names)
    if not current_business.empty:
        current_nodes = current_nodes.merge(current_business, on="node_id", how="left", sort=False)
    current_nodes = enrich_latest_pressure(current_nodes, args.latest_pressure_profiles)
    current_nodes = prepare_features(current_nodes)
    recommendations = recommend_nodes(current_nodes, final_model, top_k=3)
    concentration_frame, concentration = concentration_diagnostics(recommendations)
    business_metrics = training_business_summary(
        pairs,
        train,
        calibration,
        test,
        candidates,
        business_names,
        business_validation,
    )
    credibility = credibility_summary(
        temporal_metrics, ranking_dr, concentration, recommendations, business_metrics
    )
    data_quality = load_source_data_quality(args.source_summary)
    summary = {
        "version": "v4",
        "model_type": final_model["model_type"],
        "date_window": {
            "start": pd.Timestamp(pairs["sample_day"].min()).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(pairs["sample_day"].max()).strftime("%Y-%m-%d"),
        },
        "temporal_windows": windows,
        "training_data": {
            "raw_node_days": int(len(pairs)),
            "effective_run_support": float(pairs["sample_weight"].sum()),
            "nodes": int(pairs["node_id"].nunique()),
            "observed_businesses": int(pairs["business"].nunique()),
            "candidate_businesses": len(candidates),
            "candidate_business_ids": candidates,
            "feature_levels": MODEL_LEVELS,
            "feature_whitelist": [
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
            ],
            "latest_pressure_policy": "one latest non-null Superset netbench result per node",
            "latest_pressure_time_alignment": (
                "static latest snapshot reused for every historical sample day"
            ),
            "latest_pressure_validation_caveat": (
                "outcome dates remain time-split, but latest pressure is not reconstructed "
                "as-of each historical day"
            ),
            "latest_pressure_training_coverage": float(
                pairs["overall_packet_loss_benchmark_satisfaction_pct"].notna().mean()
            ),
            "runtime_metrics_used": False,
            "delivery_type_hard_gate": False,
        },
        "data_quality": data_quality,
        "targets": {
            "miner": "cost_finalAmount / buildBandwidth",
            "platform": "(revenue_finalAmount - cost_finalAmount) / buildBandwidth",
            "ranking": "50% normalized miner prediction + 50% normalized platform prediction",
            "outlier_policy": "per-business weighted P01-P99 winsorization for point prediction",
        },
        "temporal_validation": temporal_metrics,
        "ranking_and_dr_diagnostics": ranking_dr,
        "recommendation_concentration": concentration,
        "current_business_comparison": current_business_comparison_summary(recommendations),
        "credibility": credibility,
        "report_nodes": int(len(recommendations)),
        "artifacts": {},
    }

    paths = {
        "model": args.output_dir / OUTPUT_MODEL,
        "recommendations": args.output_dir / OUTPUT_RECOMMENDATIONS,
        "review_switch_candidates": args.output_dir / OUTPUT_SWITCH_CANDIDATES,
        "validation_predictions": args.output_dir / OUTPUT_VALIDATION,
        "business_metrics": args.output_dir / OUTPUT_BUSINESS_METRICS,
        "concentration": args.output_dir / OUTPUT_CONCENTRATION,
        "summary": args.output_dir / OUTPUT_SUMMARY,
        "report_data": args.output_dir / OUTPUT_REPORT_DATA,
        "report_html": args.output_dir / OUTPUT_REPORT_HTML,
    }
    paths["model"].write_text(
        json.dumps(final_model, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    recommendations.to_csv(paths["recommendations"], index=False)
    recommendations[recommendations["review_switch_candidate"]].to_csv(
        paths["review_switch_candidates"], index=False
    )
    validation_predictions.to_csv(paths["validation_predictions"], index=False)
    business_metrics.to_csv(paths["business_metrics"], index=False)
    concentration_frame.to_csv(paths["concentration"], index=False)
    summary["artifacts"] = {key: str(path) for key, path in paths.items()}
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_report_data(
        paths["report_data"], summary, recommendations, business_metrics, concentration_frame
    )
    if not args.report_template.exists():
        raise RuntimeError(f"report template not found: {args.report_template}")
    html = args.report_template.read_text(encoding="utf-8").replace(
        "v4_frontend_report_data.js", paths["report_data"].name
    )
    paths["report_html"].write_text(html, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def check(args: argparse.Namespace) -> int:
    required = [
        OUTPUT_MODEL,
        OUTPUT_RECOMMENDATIONS,
        OUTPUT_SWITCH_CANDIDATES,
        OUTPUT_VALIDATION,
        OUTPUT_BUSINESS_METRICS,
        OUTPUT_CONCENTRATION,
        OUTPUT_SUMMARY,
        OUTPUT_REPORT_DATA,
        OUTPUT_REPORT_HTML,
    ]
    missing = [name for name in required if not (args.output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing V4 artifacts: {missing}")
    model = json.loads((args.output_dir / OUTPUT_MODEL).read_text(encoding="utf-8"))
    summary = json.loads((args.output_dir / OUTPUT_SUMMARY).read_text(encoding="utf-8"))
    recommendations = pd.read_csv(args.output_dir / OUTPUT_RECOMMENDATIONS, low_memory=False)
    switch_candidates = pd.read_csv(args.output_dir / OUTPUT_SWITCH_CANDIDATES, low_memory=False)
    if model.get("version") != "v4" or len(model.get("candidate_businesses", [])) < 3:
        raise RuntimeError("invalid V4 model")
    if recommendations.empty or not {"business_top1", "recommendation_confidence"}.issubset(recommendations.columns):
        raise RuntimeError("invalid V4 recommendations")
    if summary.get("report_nodes") != len(recommendations):
        raise RuntimeError("V4 report node count mismatch")
    candidate_ids = set(model["candidate_businesses"])
    recommended_ids = set()
    for index in range(1, 4):
        column = f"business_top{index}"
        recommended_ids.update(recommendations[column].dropna().astype(str))
    if recommended_ids - candidate_ids:
        raise RuntimeError("V4 recommendations contain a business outside the candidate set")
    duplicate_top3 = (
        recommendations[["business_top1", "business_top2", "business_top3"]]
        .nunique(axis=1, dropna=True)
        .lt(3)
    )
    if duplicate_top3.any():
        raise RuntimeError("V4 recommendations contain duplicate Top3 businesses")
    expected_switches = int(
        summary.get("current_business_comparison", {}).get("review_switch_candidate_nodes", 0)
    )
    if len(switch_candidates) != expected_switches:
        raise RuntimeError("V4 review switch candidate count mismatch")
    if not switch_candidates.empty and (
        ~switch_candidates["review_switch_candidate"].astype(bool)
        | ~switch_candidates["model_current_not_top1"].astype(bool)
        | switch_candidates["recommendation_action"].eq("暂不推荐执行")
    ).any():
        raise RuntimeError("V4 review switch candidate eligibility check failed")
    print(json.dumps({
        "status": "V4 artifacts check passed",
        "candidate_businesses": len(model["candidate_businesses"]),
        "report_nodes": len(recommendations),
        "review_switch_candidates": len(switch_candidates),
        "credibility": summary["credibility"]["level"],
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    build_parser.add_argument("--historical-profiles", type=Path, default=DEFAULT_HISTORICAL_PROFILES)
    build_parser.add_argument(
        "--latest-pressure-profiles",
        type=Path,
        default=DEFAULT_LATEST_PRESSURE_PROFILES,
    )
    build_parser.add_argument("--current-nodes", type=Path, default=DEFAULT_CURRENT_NODES)
    build_parser.add_argument("--current-business", type=Path, default=DEFAULT_CURRENT_BUSINESS)
    build_parser.add_argument("--business-map", type=Path, default=DEFAULT_BUSINESS_MAP)
    build_parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--report-template", type=Path, default=DEFAULT_REPORT_TEMPLATE)
    build_parser.add_argument("--validation-days", type=int, default=7)
    build_parser.add_argument("--calibration-days", type=int, default=3)
    build_parser.add_argument("--min-business-support", type=float, default=30.0)
    build_parser.add_argument("--min-business-nodes", type=int, default=20)
    build_parser.add_argument("--min-segment-support", type=float, default=2.0)
    build_parser.add_argument("--smoothing-alpha", type=float, default=20.0)
    build_parser.add_argument("--learning-rate", type=float, default=0.70)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build(args)
    if args.command == "check":
        return check(args)
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
