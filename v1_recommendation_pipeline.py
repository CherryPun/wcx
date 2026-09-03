#!/usr/bin/env python3
"""V1 closed loop for node business recommendation and correction.

This script intentionally starts with a transparent, dependency-light model:

1. Build node-business training pairs from local CSV artifacts.
2. Normalize miner cost and operator profit within each node.
3. Train global and segment-level business score models.
4. Export Top3 recommendations for nodes.
5. Export an existing-node correction candidate list using the best local
   current-business proxy available in the aggregated outcome artifact.

The live current-business source is still a later Superset integration point.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_NODES = HERE / "multibusiness_nodes.csv"
DEFAULT_OUTCOMES = HERE / "multibusiness_outcomes.csv"
DEFAULT_FIELD_CONTRACT = HERE / "node_profile_v1_fields.json"

OUTCOME_RECOMMENDATION_COLUMNS = [
    "node_id",
    "business",
    "business_name",
    "cum_cost_7d",
    "cum_revenue_7d",
    "outcome_distinct_days",
    "business_active_days",
]

OUTPUT_TRAINING = "v1_training_pairs.csv"
OUTPUT_MODEL = "v1_business_score_model.json"
OUTPUT_RECOMMENDATIONS = "v1_node_recommendations.csv"
OUTPUT_CORRECTIONS = "v1_existing_node_correction_candidates.csv"
OUTPUT_METRICS = "v1_model_metrics.json"
OUTPUT_BUSINESS_MAP = "v1_business_name_map.csv"
OUTPUT_BUSINESS_RISK_PROFILE = "v1_business_risk_profile.csv"
OUTPUT_CURRENT_FROM_WIDE = "v1_current_business_from_wide.csv"

UNKNOWN = "__UNKNOWN__"
CURRENT_BUSINESS_HISTORY_SOURCE = "historical_from_multibusiness_outcomes_latest_business_online_day"
TARGET_MODE_ABSOLUTE = "absolute"
TARGET_MODE_UNIT_BANDWIDTH = "unit_bandwidth"
TARGET_MODE_HYBRID = "hybrid"
TARGET_MODES = {TARGET_MODE_ABSOLUTE, TARGET_MODE_UNIT_BANDWIDTH, TARGET_MODE_HYBRID}
CONSTRUCTION_BANDWIDTH_COLUMNS = [
    "buildBandwidth",
    "build_bandwidth",
    "bw",
    "baseInfo_bandwidth",
    "bandwidth",
]
TARGET_BANDWIDTH_COLUMNS = list(dict.fromkeys([
    *CONSTRUCTION_BANDWIDTH_COLUMNS,
    "actualbandwidth",
    "netbenchlimitbandwidth",
    "analysis_yesterday_p95_bw",
    "analysis_dby_p95_bw",
]))


SEGMENT_LEVELS = [
    ("province_isp_resourcetype", ["province", "isp", "resourcetype"]),
    ("province_isp", ["province", "isp"]),
    ("isp_resourcetype", ["isp", "resourcetype"]),
    ("province", ["province"]),
    ("isp", ["isp"]),
]


@dataclass
class Score:
    business: str
    score: float
    source: str
    support: int
    business_name: str = ""
    miner_score: float | None = None
    operator_score: float | None = None
    cum_cost_7d: float | None = None
    cum_revenue_7d: float | None = None
    cum_profit_7d: float | None = None
    outcome_distinct_days: float | None = None


QUALITY_RISK_RULES = [
    ("quality_retransrate", 5.0, 15.0, "质量重传率"),
    ("join_net_tcpretransrate", 5.0, 15.0, "入网重传率"),
    ("prom_retrans_ratio", 5.0, 15.0, "Prometheus重传率"),
    ("quality_pinglossrate", 10.0, 30.0, "质量丢包率"),
    ("join_net_pinglossrate", 10.0, 30.0, "入网丢包率"),
    ("quality_rtt", 100.0, 300.0, "质量RTT"),
    ("join_net_pingrtt", 100.0, 300.0, "入网RTT"),
    ("cpu_load1_per_core", 4.0, 8.0, "单核1分钟负载"),
    ("mem_used_ratio", 0.85, 0.95, "内存使用率"),
    ("disk_used_ratio", 0.90, 0.97, "磁盘使用率"),
    ("maxioutil", 20.0, 50.0, "最大IO利用率"),
    ("smart_case_temperature", 55.0, 65.0, "磁盘温度"),
]


ACTION_PRIORITY = {
    "review_switch_candidate": 1,
    "review_current_business_not_in_model": 2,
    "current_business_unknown": 3,
    "keep_in_top3_review": 4,
    "observe": 5,
    "keep_top1": 6,
}

PROFILE_SUMMARY_FIELDS = [
    "province",
    "city",
    "isp",
    "nattype",
    "dialtype",
    "resourcetype",
    "deliverytype",
    "device_type",
    "node_manufacturer",
    "node_model",
    "isvm",
    "bw",
    "actualbandwidth",
    "corenum",
    "memtotal",
    "totaldisksize",
    "quality_retransrate",
    "quality_pinglossrate",
    "quality_rtt",
    "join_net_tcpretransrate",
    "join_net_pinglossrate",
    "join_net_pingrtt",
    "prom_retrans_ratio",
    "cpu_load1_per_core",
    "mem_used_ratio",
    "disk_used_ratio",
    "maxioutil",
]

BUSINESS_RISK_PROFILE_FIELDS = [
    "bw",
    "actualbandwidth",
    "corenum",
    "memtotal",
    "totaldisksize",
    "quality_retransrate",
    "quality_pinglossrate",
    "quality_rtt",
    "join_net_tcpretransrate",
    "join_net_pinglossrate",
    "join_net_pingrtt",
    "prom_retrans_ratio",
    "cpu_load1_per_core",
    "mem_used_ratio",
    "disk_used_ratio",
    "maxioutil",
]


def clean_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text in {"\\N", "\\\\N", "nan", "NaN", "None", "<NA>"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def safe_category(value: Any) -> str:
    text = clean_cell(value)
    return text if text else UNKNOWN


def to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return 0.0
        return parsed
    except (TypeError, ValueError):
        return 0.0


def load_field_contract(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fields_for_group(contract: dict[str, Any], group_name: str) -> list[str]:
    for group in contract.get("groups", []):
        if group.get("name") == group_name:
            return [field["field"] for field in group.get("fields", [])]
    return []


def v1_profile_fields(contract: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for group_name in ["identity", "common_profile", "network_shape_optional", "network_quality_risk"]:
        fields.extend(fields_for_group(contract, group_name))
    return list(dict.fromkeys(fields))


def model_feature_fields(contract: dict[str, Any], include_optional_network: bool = True) -> list[str]:
    fields = []
    for group in contract.get("groups", []):
        if group.get("name") not in {"identity", "common_profile", "network_shape_optional"}:
            continue
        for field in group.get("fields", []):
            value = field.get("new_node_model")
            if value is True or (include_optional_network and value == "optional"):
                fields.append(field["field"])
    return list(dict.fromkeys(fields))


def read_sources(nodes_path: Path, outcomes_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(nodes_path, dtype={"node_id": "string"}, low_memory=False)
    outcomes = pd.read_csv(
        outcomes_path,
        dtype={"node_id": "string", "business": "string", "business_name": "string"},
        low_memory=False,
    )
    nodes["node_id"] = nodes["node_id"].map(clean_cell)
    outcomes["node_id"] = outcomes["node_id"].map(clean_cell)
    outcomes["business"] = outcomes["business"].map(clean_cell)
    if "business_name" not in outcomes.columns:
        outcomes["business_name"] = ""
    outcomes["business_name"] = outcomes["business_name"].fillna("").map(clean_cell)
    return nodes, outcomes


def read_outcomes_for_recommendation(outcomes_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        outcomes_path,
        dtype={"node_id": "string", "business": "string", "business_name": "string"},
        usecols=lambda column: column in OUTCOME_RECOMMENDATION_COLUMNS,
        low_memory=False,
    )


def detect_column(frame: pd.DataFrame, candidates: list[str], required: bool = True) -> str:
    lower_map = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        lowered = candidate.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    if required:
        raise KeyError(f"missing expected column, tried: {', '.join(candidates)}")
    return ""


def build_business_name_map(outcomes: pd.DataFrame, business_map_path: Path | None = None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if {"business", "business_name"}.issubset(outcomes.columns):
        for row in outcomes[["business", "business_name"]].drop_duplicates().itertuples(index=False):
            business = clean_cell(getattr(row, "business"))
            name = clean_cell(getattr(row, "business_name"))
            if business and name:
                mapping.setdefault(business, name)

    if business_map_path:
        frame = pd.read_csv(business_map_path, dtype="string", low_memory=False)
        business_column = detect_column(frame, ["business", "business_id", "customerId", "customer_id"])
        name_column = detect_column(frame, ["business_name", "customerName", "customer_name", "name"])
        for row in frame[[business_column, name_column]].drop_duplicates().itertuples(index=False):
            business = clean_cell(getattr(row, business_column))
            name = clean_cell(getattr(row, name_column))
            if business and name:
                mapping[business] = name
    return mapping


def apply_business_names(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if not mapping:
        return frame
    output = frame.copy()
    if "business_name" not in output.columns:
        output["business_name"] = ""
    output["business_name"] = output["business_name"].fillna("").map(clean_cell)
    output["business"] = output["business"].map(clean_cell)
    missing = output["business_name"].eq("")
    output.loc[missing, "business_name"] = output.loc[missing, "business"].map(mapping).fillna("")
    return output


def business_map_frame(mapping: dict[str, str]) -> pd.DataFrame:
    rows = [{"business": business, "business_name": name} for business, name in sorted(mapping.items())]
    return pd.DataFrame(rows, columns=["business", "business_name"])


def normalize_within_node(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    low = numeric.min()
    high = numeric.max()
    if not math.isfinite(float(high - low)) or abs(float(high - low)) < 1e-12:
        return pd.Series(np.full(len(numeric), 0.5), index=values.index)
    return (numeric - low) / (high - low)


def robust_global_normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(numeric) == 0:
        return numeric
    low = numeric.quantile(0.01)
    high = numeric.quantile(0.99)
    if not math.isfinite(float(high - low)) or abs(float(high - low)) < 1e-12:
        low = numeric.min()
        high = numeric.max()
    if not math.isfinite(float(high - low)) or abs(float(high - low)) < 1e-12:
        return pd.Series(np.full(len(numeric), 0.5), index=values.index)
    return ((numeric.clip(lower=low, upper=high) - low) / (high - low)).clip(0.0, 1.0)


def construction_bandwidth_mbps(frame: pd.DataFrame, floor_mbps: float = 1.0) -> pd.Series:
    """Use configured construction bandwidth as the unit-yield denominator."""
    bandwidth = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in CONSTRUCTION_BANDWIDTH_COLUMNS:
        if column not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[column], errors="coerce").where(lambda values: values > 0)
        bandwidth = bandwidth.fillna(candidate)
    bandwidth = bandwidth.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return bandwidth.clip(lower=floor_mbps)


def effective_bandwidth_mbps(frame: pd.DataFrame, floor_mbps: float = 1.0) -> pd.Series:
    """Backward-compatible alias for the construction-bandwidth denominator."""
    return construction_bandwidth_mbps(frame, floor_mbps=floor_mbps)


def prepare_outcomes(outcomes: pd.DataFrame, target_mode: str = TARGET_MODE_ABSOLUTE) -> pd.DataFrame:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"unsupported target_mode: {target_mode}")
    frame = outcomes.copy()
    for column in [
        "cum_cost_7d",
        "cum_revenue_7d",
        "outcome_distinct_days",
        "business_active_days",
        *TARGET_BANDWIDTH_COLUMNS,
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["cum_profit_7d"] = frame["cum_revenue_7d"] - frame["cum_cost_7d"]
    group_fields = outcome_comparison_fields(frame)

    absolute_miner = frame.groupby(group_fields, dropna=False)["cum_cost_7d"].transform(normalize_within_node)
    absolute_operator = frame.groupby(group_fields, dropna=False)["cum_profit_7d"].transform(normalize_within_node)
    frame["construction_bandwidth_mbps"] = construction_bandwidth_mbps(frame)
    frame["effective_bandwidth_mbps"] = frame["construction_bandwidth_mbps"]
    frame["cost_per_bandwidth_mbps"] = frame["cum_cost_7d"] / frame["effective_bandwidth_mbps"]
    frame["revenue_per_bandwidth_mbps"] = frame["cum_revenue_7d"] / frame["effective_bandwidth_mbps"]
    frame["profit_per_bandwidth_mbps"] = frame["cum_profit_7d"] / frame["effective_bandwidth_mbps"]
    unit_miner = robust_global_normalize(frame["cost_per_bandwidth_mbps"])
    unit_operator = robust_global_normalize(frame["profit_per_bandwidth_mbps"])

    if target_mode == TARGET_MODE_UNIT_BANDWIDTH:
        frame["miner_score_norm"] = unit_miner
        frame["operator_score_norm"] = unit_operator
    elif target_mode == TARGET_MODE_HYBRID:
        frame["miner_score_norm"] = 0.5 * absolute_miner + 0.5 * unit_miner
        frame["operator_score_norm"] = 0.5 * absolute_operator + 0.5 * unit_operator
    else:
        frame["miner_score_norm"] = absolute_miner
        frame["operator_score_norm"] = absolute_operator
    frame["combined_score"] = 0.5 * frame["miner_score_norm"] + 0.5 * frame["operator_score_norm"]
    frame["target_mode"] = target_mode
    best_index = frame.groupby(group_fields, dropna=False)["combined_score"].idxmax()
    frame["is_combined_best"] = False
    frame.loc[best_index, "is_combined_best"] = True
    return frame


def outcome_comparison_fields(frame: pd.DataFrame) -> list[str]:
    """Choose the comparison unit for target normalization."""
    if "outcome_group_id" in frame.columns and frame["outcome_group_id"].fillna("").astype(str).str.strip().ne("").any():
        return ["outcome_group_id"]
    if "sample_day" in frame.columns and frame["sample_day"].fillna("").astype(str).str.strip().ne("").any():
        return ["node_id", "sample_day"]
    return ["node_id"]


def build_training_pairs(
    nodes: pd.DataFrame,
    outcomes: pd.DataFrame,
    contract: dict[str, Any],
    target_mode: str = TARGET_MODE_ABSOLUTE,
) -> pd.DataFrame:
    profile_fields = [field for field in v1_profile_fields(contract) if field in nodes.columns]
    node_frame = nodes[["node_id"] + [field for field in profile_fields if field != "node_id"]].drop_duplicates("node_id")
    pairs = outcomes.merge(node_frame, on="node_id", how="left", sort=False)
    daily_profile_overrides = {
        "daily_province": "province",
        "daily_city": "city",
        "daily_isp": "isp",
        "daily_nattype": "nattype",
        "daily_resourcetype": "resourcetype",
        "daily_deliverytype": "deliverytype",
    }
    for daily_field, profile_field in daily_profile_overrides.items():
        if daily_field not in pairs.columns:
            continue
        daily_value = pairs[daily_field].fillna("").map(clean_cell)
        static_value = pairs[profile_field] if profile_field in pairs.columns else ""
        pairs[profile_field] = daily_value.where(daily_value.ne(""), static_value)
    if "daily_scheduleisps" in pairs.columns:
        daily_schedule = pairs["daily_scheduleisps"].fillna("").map(clean_cell)
        static_schedule = pairs["scheduleisps"] if "scheduleisps" in pairs.columns else ""
        pairs["scheduleisps"] = daily_schedule.where(daily_schedule.ne(""), static_schedule)
    if "daily_transprovrate" in pairs.columns:
        daily_rate = pd.to_numeric(pairs["daily_transprovrate"], errors="coerce")
        if "analysis_transprovrate" in pairs.columns:
            static_rate = pd.to_numeric(pairs["analysis_transprovrate"], errors="coerce")
        else:
            static_rate = pd.Series(np.nan, index=pairs.index)
        pairs["analysis_transprovrate"] = daily_rate.fillna(static_rate)
    if "buildBandwidth" in pairs.columns:
        daily_bw = pd.to_numeric(pairs["buildBandwidth"], errors="coerce").where(lambda values: values > 0)
        static_bw = pd.to_numeric(pairs["bw"], errors="coerce") if "bw" in pairs.columns else pd.Series(np.nan, index=pairs.index)
        pairs["bw"] = daily_bw.fillna(static_bw)
    return prepare_outcomes(pairs, target_mode=target_mode)


def split_node_ids(node_ids: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(node_ids), dtype=object)
    rng.shuffle(shuffled)
    train_count = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    return set(shuffled[:train_count]), set(shuffled[train_count:])


def segment_key(row: pd.Series | dict[str, Any], fields: list[str]) -> str:
    return json.dumps([safe_category(row.get(field, "")) for field in fields], ensure_ascii=False)


def latest_nonempty(series: pd.Series) -> str:
    for value in series.dropna().astype(str):
        cleaned = clean_cell(value)
        if cleaned:
            return cleaned
    return ""


def fit_score_model(
    train_pairs: pd.DataFrame,
    contract: dict[str, Any],
    min_business_support: int,
    smoothing_alpha: float,
) -> dict[str, Any]:
    train = train_pairs.copy()
    train["business"] = train["business"].map(clean_cell)
    support = train["business"].value_counts()
    candidate_businesses = sorted(support[support >= min_business_support].index.astype(str).tolist())
    if not candidate_businesses:
        raise RuntimeError("No business meets min support; lower --min-business-support.")
    train = train[train["business"].isin(candidate_businesses)].copy()

    global_rows = (
        train.groupby("business", dropna=False)
        .agg(
            score=("combined_score", "mean"),
            support=("combined_score", "size"),
            miner_score=("miner_score_norm", "mean"),
            operator_score=("operator_score_norm", "mean"),
            business_name=("business_name", latest_nonempty),
        )
        .reset_index()
    )
    global_scores = {
        str(row.business): {
            "score": float(row.score),
            "support": int(row.support),
            "miner_score": float(row.miner_score),
            "operator_score": float(row.operator_score),
            "business_name": clean_cell(row.business_name),
        }
        for row in global_rows.itertuples(index=False)
    }

    segments: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    available_levels = []
    for level_name, fields in SEGMENT_LEVELS:
        fields = [field for field in fields if field in train.columns]
        if not fields:
            continue
        available_levels.append({"name": level_name, "fields": fields})
        working = train.copy()
        working["_segment_key"] = working.apply(lambda row: segment_key(row, fields), axis=1)
        grouped = (
            working.groupby(["_segment_key", "business"], dropna=False)
            .agg(
                score=("combined_score", "mean"),
                support=("combined_score", "size"),
                miner_score=("miner_score_norm", "mean"),
                operator_score=("operator_score_norm", "mean"),
            )
            .reset_index()
        )
        level_payload: dict[str, dict[str, dict[str, Any]]] = {}
        for row in grouped.to_dict(orient="records"):
            level_payload.setdefault(str(row["_segment_key"]), {})[str(row["business"])] = {
                "score": float(row["score"]),
                "support": int(row["support"]),
                "miner_score": float(row["miner_score"]),
                "operator_score": float(row["operator_score"]),
            }
        segments[level_name] = level_payload

    return {
        "version": "v1",
        "model_type": "smoothed_segment_business_score",
        "objective": "combined_score = 0.5 * normalized_cost_within_node + 0.5 * normalized_profit_within_node",
        "candidate_businesses": candidate_businesses,
        "global_scores": global_scores,
        "segment_levels": available_levels,
        "segments": segments,
        "smoothing_alpha": float(smoothing_alpha),
        "min_business_support": int(min_business_support),
        "feature_fields": model_feature_fields(contract),
        "current_business_source": CURRENT_BUSINESS_HISTORY_SOURCE,
    }


def score_business(row: pd.Series | dict[str, Any], business: str, model: dict[str, Any]) -> Score:
    business = clean_cell(business)
    global_info = model["global_scores"].get(business)
    if not global_info:
        return Score(business=business, score=float("nan"), source="not_candidate", support=0)
    global_score = float(global_info["score"])
    global_miner_score = float(global_info.get("miner_score", global_score))
    global_operator_score = float(global_info.get("operator_score", global_score))
    alpha = float(model.get("smoothing_alpha", 20.0))
    for level in model.get("segment_levels", []):
        level_name = level["name"]
        key = segment_key(row, level["fields"])
        business_info = model.get("segments", {}).get(level_name, {}).get(key, {}).get(business)
        if business_info:
            support = int(business_info["support"])
            segment_score = float(business_info["score"])
            segment_miner_score = float(business_info.get("miner_score", global_miner_score))
            segment_operator_score = float(business_info.get("operator_score", global_operator_score))
            smoothed = (segment_score * support + global_score * alpha) / (support + alpha)
            miner_score = (segment_miner_score * support + global_miner_score * alpha) / (support + alpha)
            operator_score = (
                segment_operator_score * support + global_operator_score * alpha
            ) / (support + alpha)
            return Score(
                business=business,
                score=smoothed,
                source=level_name,
                support=support,
                business_name=clean_cell(global_info.get("business_name")),
                miner_score=miner_score,
                operator_score=operator_score,
            )
    return Score(
        business=business,
        score=global_score,
        source="global",
        support=int(global_info["support"]),
        business_name=clean_cell(global_info.get("business_name")),
        miner_score=global_miner_score,
        operator_score=global_operator_score,
    )


def score_node(row: pd.Series | dict[str, Any], model: dict[str, Any], top_k: int) -> list[Score]:
    scores = [score_business(row, business, model) for business in model["candidate_businesses"]]
    scores = [score for score in scores if math.isfinite(score.score)]
    return sorted(scores, key=lambda item: (-item.score, item.business))[:top_k]


def top3_text(scores: list[Score]) -> str:
    return ";".join(f"{score.business}:{score.score:.6g}:{score.source}" for score in scores)


def score_value(value: float | None) -> str | float:
    if value is None:
        return ""
    if not math.isfinite(float(value)):
        return ""
    return float(value)


def score_reason(score: Score) -> str:
    if score.source == "observed_node_outcome":
        return (
            "observed node-business outcome"
            f"; miner_norm={score_value(score.miner_score)}"
            f"; operator_norm={score_value(score.operator_score)}"
            f"; profit_7d={score_value(score.cum_profit_7d)}"
            f"; days={score_value(score.outcome_distinct_days)}"
        )
    return (
        f"matched {score.source} model segment"
        f"; support={score.support}"
        f"; miner_norm={score_value(score.miner_score)}"
        f"; operator_norm={score_value(score.operator_score)}"
    )


def observed_scores_by_node(outcomes: pd.DataFrame, top_k: int) -> dict[str, list[Score]]:
    frame = outcomes.copy()
    if "combined_score" not in frame.columns:
        frame = prepare_outcomes(frame)
    for column in [
        "combined_score",
        "miner_score_norm",
        "operator_score_norm",
        "cum_cost_7d",
        "cum_revenue_7d",
        "cum_profit_7d",
        "outcome_distinct_days",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["node_id"] = frame["node_id"].map(clean_cell)
    frame["business"] = frame["business"].map(clean_cell)
    if "business_name" not in frame.columns:
        frame["business_name"] = ""
    frame["business_name"] = frame["business_name"].fillna("").map(clean_cell)
    frame = frame[frame["node_id"].ne("") & frame["business"].ne("")].copy()
    frame = frame.sort_values(["node_id", "combined_score", "business"], ascending=[True, False, True])
    result: dict[str, list[Score]] = {}
    for node_id, group in frame.groupby("node_id", sort=False):
        scores: list[Score] = []
        for row in group.head(top_k).itertuples(index=False):
            scores.append(
                Score(
                    business=clean_cell(getattr(row, "business")),
                    business_name=clean_cell(getattr(row, "business_name", "")),
                    score=float(getattr(row, "combined_score")),
                    source="observed_node_outcome",
                    support=int(to_float(getattr(row, "outcome_distinct_days", 0))),
                    miner_score=float(getattr(row, "miner_score_norm")),
                    operator_score=float(getattr(row, "operator_score_norm")),
                    cum_cost_7d=float(getattr(row, "cum_cost_7d", 0.0)),
                    cum_revenue_7d=float(getattr(row, "cum_revenue_7d", 0.0)),
                    cum_profit_7d=float(getattr(row, "cum_profit_7d", 0.0)),
                    outcome_distinct_days=float(getattr(row, "outcome_distinct_days", 0.0)),
                )
            )
        result[clean_cell(node_id)] = scores
    return result


def observed_score_lookup(outcomes: pd.DataFrame) -> dict[tuple[str, str], Score]:
    frame = outcomes.copy()
    if "combined_score" not in frame.columns:
        frame = prepare_outcomes(frame)
    for column in [
        "combined_score",
        "miner_score_norm",
        "operator_score_norm",
        "cum_cost_7d",
        "cum_revenue_7d",
        "cum_profit_7d",
        "outcome_distinct_days",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["node_id"] = frame["node_id"].map(clean_cell)
    frame["business"] = frame["business"].map(clean_cell)
    if "business_name" not in frame.columns:
        frame["business_name"] = ""
    frame["business_name"] = frame["business_name"].fillna("").map(clean_cell)
    lookup: dict[tuple[str, str], Score] = {}
    for row in frame.itertuples(index=False):
        node_id = clean_cell(getattr(row, "node_id"))
        business = clean_cell(getattr(row, "business"))
        if not node_id or not business:
            continue
        lookup[(node_id, business)] = Score(
            business=business,
            business_name=clean_cell(getattr(row, "business_name", "")),
            score=float(getattr(row, "combined_score")),
            source="observed_node_outcome",
            support=int(to_float(getattr(row, "outcome_distinct_days", 0))),
            miner_score=float(getattr(row, "miner_score_norm")),
            operator_score=float(getattr(row, "operator_score_norm")),
            cum_cost_7d=float(getattr(row, "cum_cost_7d", 0.0)),
            cum_revenue_7d=float(getattr(row, "cum_revenue_7d", 0.0)),
            cum_profit_7d=float(getattr(row, "cum_profit_7d", 0.0)),
            outcome_distinct_days=float(getattr(row, "outcome_distinct_days", 0.0)),
        )
    return lookup


def missing_profile_fields(row: dict[str, Any], model: dict[str, Any]) -> list[str]:
    missing = []
    for field in model.get("feature_fields", []):
        value = row.get(field, "")
        if clean_cell(value) == "":
            missing.append(field)
    return missing


def risk_from_missing(missing: list[str], feature_count: int) -> tuple[str, str]:
    if feature_count <= 0:
        return "unknown", "feature contract missing"
    ratio = len(missing) / feature_count
    if ratio >= 0.5:
        return "high", "many V1 profile fields missing"
    if ratio > 0:
        return "medium", "some V1 profile fields missing"
    return "low", ""


def quality_risk_flags(row: dict[str, Any]) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []
    for field, medium_threshold, high_threshold, label in QUALITY_RISK_RULES:
        if field not in row:
            continue
        raw_value = row.get(field)
        if clean_cell(raw_value) == "":
            continue
        value = to_float(raw_value)
        if value >= high_threshold:
            flags.append(("high", f"{label} {field}={value:.4g} >= {high_threshold:g}"))
        elif value >= medium_threshold:
            flags.append(("medium", f"{label} {field}={value:.4g} >= {medium_threshold:g}"))
    smart_warning = clean_cell(row.get("smart_critical_warning", ""))
    if smart_warning and smart_warning not in {"0", "0.0", "false", "False"}:
        flags.append(("high", f"SMART critical warning={smart_warning}"))
    zfs_online = clean_cell(row.get("zfs_pool_online", ""))
    if zfs_online and zfs_online not in {"1", "1.0", "true", "True", "online", "ONLINE"}:
        flags.append(("high", f"ZFS pool not online={zfs_online}"))
    return flags


def combine_risk(row: dict[str, Any], model: dict[str, Any]) -> tuple[str, str, list[str]]:
    missing = missing_profile_fields(row, model)
    missing_level, missing_reason = risk_from_missing(missing, len(model.get("feature_fields", [])))
    quality_flags = quality_risk_flags(row)
    levels = [level for level, _ in quality_flags]
    if missing_level in {"high", "medium"}:
        levels.append(missing_level)
    if "high" in levels:
        risk_level = "high"
    elif "medium" in levels:
        risk_level = "medium"
    elif missing_level == "unknown":
        risk_level = "unknown"
    else:
        risk_level = "low"
    reasons = [reason for _, reason in quality_flags]
    if missing_reason:
        reasons.append(missing_reason)
    return risk_level, "; ".join(reasons), missing


def recommendations_for_nodes(
    nodes: pd.DataFrame,
    model: dict[str, Any],
    top_k: int,
    observed_by_node: dict[str, list[Score]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node in nodes.drop_duplicates("node_id").itertuples(index=False):
        row = node._asdict()
        node_id = clean_cell(row.get("node_id"))
        observed_scores = (observed_by_node or {}).get(node_id, [])
        if observed_scores:
            scores = observed_scores[:top_k]
            recommendation_mode = "observed_node_outcome"
        else:
            scores = score_node(row, model, top_k)
            recommendation_mode = "model_segment_fallback"
        output: dict[str, Any] = {"node_id": node_id, "recommendation_mode": recommendation_mode}
        for index, score in enumerate(scores, 1):
            output[f"recommended_business_top{index}"] = score.business
            output[f"recommended_business_name_top{index}"] = score.business_name
            output[f"recommended_score_top{index}"] = score.score
            output[f"recommended_source_top{index}"] = score.source
            output[f"recommended_support_top{index}"] = score.support
            output[f"recommended_miner_score_top{index}"] = score_value(score.miner_score)
            output[f"recommended_operator_score_top{index}"] = score_value(score.operator_score)
            output[f"recommended_cum_cost_7d_top{index}"] = score_value(score.cum_cost_7d)
            output[f"recommended_cum_revenue_7d_top{index}"] = score_value(score.cum_revenue_7d)
            output[f"recommended_cum_profit_7d_top{index}"] = score_value(score.cum_profit_7d)
            output[f"recommended_outcome_days_top{index}"] = score_value(score.outcome_distinct_days)
            output[f"recommendation_reason_top{index}"] = score_reason(score)
            output[f"combined_top{index}"] = score.business
            output[f"combined_score_top{index}"] = score.score
        output["recommended_top3"] = top3_text(scores)
        output["combined_top3_detail"] = output["recommended_top3"]
        risk_level, risk_reason, missing = combine_risk(row, model)
        output["risk_level"] = risk_level
        output["risk_reasons"] = risk_reason
        output["missing_profile_field_count"] = len(missing)
        output["missing_profile_fields"] = ",".join(missing)
        rows.append(output)
    return pd.DataFrame(rows)


def evaluate_recommendations(
    test_pairs: pd.DataFrame,
    test_nodes: pd.DataFrame,
    model: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    rec = recommendations_for_nodes(test_nodes, model, top_k).set_index("node_id")
    work = test_pairs.copy()
    group_fields = outcome_comparison_fields(work)
    if group_fields == ["outcome_group_id"]:
        work["_comparison_group_id"] = work["outcome_group_id"].map(clean_cell)
    elif group_fields == ["node_id", "sample_day"]:
        work["_comparison_group_id"] = work["node_id"].map(clean_cell) + "|" + work["sample_day"].map(clean_cell)
    else:
        work["_comparison_group_id"] = work["node_id"].map(clean_cell)
    true_best = work.loc[work.groupby("_comparison_group_id", dropna=False)["combined_score"].idxmax()]
    outcomes_by_group = {
        clean_cell(group_id): group.set_index("business")
        for group_id, group in work.groupby("_comparison_group_id", sort=False)
    }
    hit1: list[bool] = []
    hit3: list[bool] = []
    observed_coverage = 0
    regrets: list[float] = []
    evaluated = 0
    for true_row in true_best.to_dict(orient="records"):
        node_id = clean_cell(true_row.get("node_id"))
        group_id = clean_cell(true_row.get("_comparison_group_id"))
        if node_id not in rec.index:
            continue
        evaluated += 1
        row = rec.loc[node_id]
        top_businesses = [
            clean_cell(row.get(f"recommended_business_top{index}", ""))
            for index in range(1, top_k + 1)
        ]
        true_business = clean_cell(true_row["business"])
        hit1.append(bool(top_businesses and top_businesses[0] == true_business))
        hit3.append(true_business in set(top_businesses))

        group_outcomes = outcomes_by_group.get(group_id)
        if group_outcomes is None:
            continue
        selected = top_businesses[0] if top_businesses else ""
        if selected in group_outcomes.index:
            observed_coverage += 1
            best_score = float(group_outcomes["combined_score"].max())
            selected_score = float(group_outcomes.loc[selected, "combined_score"])
            regrets.append(max(0.0, (best_score - selected_score) / max(abs(best_score), 1e-9)))

    return {
        "n_nodes": evaluated,
        "n_eval_groups": evaluated,
        "n_eval_unique_nodes": int(true_best["node_id"].nunique()) if "node_id" in true_best else 0,
        "hit_rate_at_1": float(np.mean(hit1)) if hit1 else 0.0,
        "hit_rate_at_3": float(np.mean(hit3)) if hit3 else 0.0,
        "top1_observed_coverage": observed_coverage / evaluated if evaluated else 0.0,
        "observed_regret": float(np.mean(regrets)) if regrets else None,
    }


def infer_current_business_proxy(outcomes: pd.DataFrame) -> pd.DataFrame:
    frame = outcomes.copy()
    frame["business_online_day_parsed"] = pd.to_datetime(frame.get("business_online_day"), errors="coerce")
    for column in ["outcome_distinct_days", "business_active_days", "cum_revenue_7d", "cum_cost_7d"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame.sort_values(
        [
            "node_id",
            "business_online_day_parsed",
            "outcome_distinct_days",
            "business_active_days",
            "cum_revenue_7d",
            "cum_cost_7d",
            "business",
        ],
        ascending=[True, False, False, False, False, False, True],
    )
    current = frame.drop_duplicates("node_id").copy()
    return pd.DataFrame({
        "node_id": current["node_id"].map(clean_cell),
        "current_business": current["business"].map(clean_cell),
        "current_business_name": current["business_name"].fillna("").map(clean_cell),
        "current_business_proxy_day": current["business_online_day"].map(clean_cell),
        "current_business_day": current["business_online_day"].map(clean_cell),
        "current_business_source": CURRENT_BUSINESS_HISTORY_SOURCE,
        "current_business_confidence": "historical_inferred",
    })


def normalize_current_business_frame(
    frame: pd.DataFrame,
    source: str = "authoritative_current_business_csv",
    default_confidence: str = "authoritative",
    business_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    node_column = detect_column(frame, ["node_id", "nodeId"])
    business_column = detect_column(frame, ["current_business", "business", "customerId", "customer_id"])
    name_column = detect_column(
        frame,
        ["current_business_name", "business_name", "customerName", "customer_name"],
        required=False,
    )
    day_column = detect_column(
        frame,
        ["current_business_day", "current_day", "day", "latest_day", "current_business_proxy_day"],
        required=False,
    )
    source_column = detect_column(frame, ["current_business_source", "source"], required=False)
    confidence_column = detect_column(frame, ["current_business_confidence", "confidence"], required=False)

    output = pd.DataFrame({
        "node_id": frame[node_column].map(clean_cell),
        "current_business": frame[business_column].map(clean_cell),
    })
    if name_column:
        output["current_business_name"] = frame[name_column].fillna("").map(clean_cell)
    else:
        output["current_business_name"] = ""
    if day_column:
        output["current_business_day"] = frame[day_column].map(clean_cell)
    else:
        output["current_business_day"] = ""
    if source_column:
        output["current_business_source"] = frame[source_column].fillna("").map(clean_cell)
        output.loc[output["current_business_source"].eq(""), "current_business_source"] = source
    else:
        output["current_business_source"] = source
    if confidence_column:
        output["current_business_confidence"] = frame[confidence_column].fillna("").map(clean_cell)
        output.loc[output["current_business_confidence"].eq(""), "current_business_confidence"] = default_confidence
    else:
        output["current_business_confidence"] = default_confidence

    if business_name_map:
        missing = output["current_business_name"].eq("")
        output.loc[missing, "current_business_name"] = (
            output.loc[missing, "current_business"].map(business_name_map).fillna("")
        )
    output = output[output["node_id"].ne("")].copy()
    output["_day_sort"] = pd.to_datetime(output["current_business_day"], errors="coerce")
    output = output.sort_values(["node_id", "_day_sort"], ascending=[True, False])
    return output.drop_duplicates("node_id").drop(columns=["_day_sort"])


def read_current_business(
    path: Path,
    business_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", low_memory=False)
    return normalize_current_business_frame(
        frame,
        source=f"authoritative_current_business_csv:{path.name}",
        default_confidence="authoritative",
        business_name_map=business_name_map,
    )


def derive_current_business_from_wide(
    wide: pd.DataFrame,
    lookback_days: int,
    min_active_days: int,
    online_only: bool = True,
) -> pd.DataFrame:
    node_column = detect_column(wide, ["nodeId", "node_id"])
    business_column = detect_column(wide, ["customerId", "customer_id", "business", "current_business"])
    name_column = detect_column(wide, ["customerName", "customer_name", "business_name"], required=False)
    day_column = detect_column(wide, ["day", "date"])
    state_column = detect_column(wide, ["state"], required=False)
    peak_column = detect_column(wide, ["peak95", "analyzePeak95"], required=False)
    revenue_column = detect_column(
        wide,
        ["revenue_finalAmount", "revenue_amount", "revenue_estimatedFinalAmount"],
        required=False,
    )
    cost_column = detect_column(wide, ["cost_finalAmount", "cost_settlement", "cost_original"], required=False)
    profit_column = detect_column(wide, ["profit_profitAmount", "profit_estimatedProfitAmount"], required=False)

    frame = pd.DataFrame({
        "node_id": wide[node_column].map(clean_cell),
        "current_business": wide[business_column].map(clean_cell),
        "day": pd.to_datetime(wide[day_column], errors="coerce"),
        "current_business_name": wide[name_column].fillna("").map(clean_cell) if name_column else "",
        "state": wide[state_column].fillna("").map(clean_cell) if state_column else "",
        "peak95": pd.to_numeric(wide[peak_column], errors="coerce").fillna(0.0) if peak_column else 0.0,
        "revenue": pd.to_numeric(wide[revenue_column], errors="coerce").fillna(0.0) if revenue_column else 0.0,
        "cost": pd.to_numeric(wide[cost_column], errors="coerce").fillna(0.0) if cost_column else 0.0,
        "profit": pd.to_numeric(wide[profit_column], errors="coerce").fillna(0.0) if profit_column else 0.0,
    })
    frame = frame[frame["node_id"].ne("") & frame["current_business"].ne("")].copy()
    business_numeric = pd.to_numeric(frame["current_business"], errors="coerce")
    frame = frame[business_numeric.fillna(0) > 0].copy()
    if online_only and state_column:
        online_values = {"online", "ONLINE", "Online", "在线"}
        frame = frame[frame["state"].isin(online_values)].copy()
    if frame.empty:
        return pd.DataFrame(columns=[
            "node_id",
            "current_business",
            "current_business_name",
            "current_business_day",
            "current_business_active_days",
            "current_business_peak95_sum",
            "current_business_revenue_sum",
            "current_business_cost_sum",
            "current_business_profit_sum",
            "current_business_source",
            "current_business_confidence",
        ])
    if lookback_days > 0 and frame["day"].notna().any():
        latest_day = frame["day"].max()
        cutoff = latest_day - pd.Timedelta(days=lookback_days - 1)
        frame = frame[frame["day"] >= cutoff].copy()

    grouped = (
        frame.groupby(["node_id", "current_business"], dropna=False)
        .agg(
            current_business_name=("current_business_name", latest_nonempty),
            current_business_day=("day", "max"),
            current_business_active_days=("day", "nunique"),
            current_business_peak95_sum=("peak95", "sum"),
            current_business_peak95_max=("peak95", "max"),
            current_business_revenue_sum=("revenue", "sum"),
            current_business_cost_sum=("cost", "sum"),
            current_business_profit_sum=("profit", "sum"),
        )
        .reset_index()
    )
    grouped["current_business_confidence"] = np.where(
        grouped["current_business_active_days"] >= min_active_days,
        "wide_high",
        "wide_medium",
    )
    grouped["current_business_source"] = "derived_from_node_day_ops_wide"
    grouped = grouped.sort_values(
        [
            "node_id",
            "current_business_active_days",
            "current_business_day",
            "current_business_peak95_sum",
            "current_business_revenue_sum",
            "current_business",
        ],
        ascending=[True, False, False, False, False, True],
    )
    current = grouped.drop_duplicates("node_id").copy()
    current["current_business_day"] = current["current_business_day"].dt.strftime("%Y-%m-%d")
    return current


def build_correction_candidates(
    nodes: pd.DataFrame,
    outcomes: pd.DataFrame,
    recommendations: pd.DataFrame,
    model: dict[str, Any],
    min_switch_gap: float,
    top_k: int,
    current_business: pd.DataFrame | None = None,
) -> pd.DataFrame:
    proxy_current = infer_current_business_proxy(outcomes)
    if current_business is not None:
        authoritative = current_business.copy()
        authoritative["_current_priority"] = 0
        proxy_current["_current_priority"] = 1
        current = (
            pd.concat([authoritative, proxy_current], ignore_index=True, sort=False)
            .sort_values(["node_id", "_current_priority"], ascending=[True, True])
            .drop_duplicates("node_id")
            .drop(columns=["_current_priority"])
        )
    else:
        current = proxy_current
    outcome_score_lookup = observed_score_lookup(outcomes)
    node_map = nodes.drop_duplicates("node_id").set_index("node_id")
    rec = recommendations.merge(current, on="node_id", how="left")
    rows: list[dict[str, Any]] = []
    for row in rec.itertuples(index=False):
        item = row._asdict()
        node_id = clean_cell(item.get("node_id"))
        current_business = clean_cell(item.get("current_business"))
        top_businesses = [
            clean_cell(item.get(f"recommended_business_top{index}", ""))
            for index in range(1, top_k + 1)
        ]
        top_businesses = [business for business in top_businesses if business]
        top1 = top_businesses[0] if top_businesses else ""
        top1_score = to_float(item.get("recommended_score_top1"))
        current_score = float("nan")
        current_score_source = ""
        observed_current_score = outcome_score_lookup.get((node_id, current_business))
        if observed_current_score:
            current_score = observed_current_score.score
            current_score_source = "observed_node_outcome"
        elif node_id in node_map.index and current_business:
            current_score = score_business(node_map.loc[node_id].to_dict(), current_business, model).score
            if math.isfinite(current_score):
                current_score_source = "model_segment_fallback"
        current_in_top3 = current_business in set(top_businesses)
        current_eq_top1 = bool(current_business and current_business == top1)
        if not current_business:
            action = "current_business_unknown"
        elif current_eq_top1:
            action = "keep_top1"
        elif current_in_top3:
            action = "keep_in_top3_review"
        elif not math.isfinite(current_score):
            action = "review_current_business_not_in_model"
        elif math.isfinite(current_score) and top1_score - current_score >= min_switch_gap:
            action = "review_switch_candidate"
        else:
            action = "observe"
        item.update({
            "current_eq_top1": current_eq_top1,
            "current_business_in_top3": current_in_top3,
            "current_in_top3": current_in_top3,
            "current_business_score": current_score if math.isfinite(current_score) else "",
            "current_business_score_source": current_score_source,
            "recommended_minus_current_score": (
                top1_score - current_score if math.isfinite(current_score) else ""
            ),
            "suggested_action": action,
            "action_priority": ACTION_PRIORITY.get(action, 99),
            "reason_summary": (
                "current business equals recommended top1" if current_eq_top1
                else "current business is within recommended top3" if current_in_top3
                else "current business is not in model candidates" if action == "review_current_business_not_in_model"
                else "current business is outside recommended top3"
            ),
        })
        rows.append(item)
    output = pd.DataFrame(rows)
    sort_columns = ["action_priority", "recommended_minus_current_score", "recommended_score_top1"]
    existing = [column for column in sort_columns if column in output.columns]
    if existing:
        output = output.sort_values(existing, ascending=[True, False, False][: len(existing)])
    return output


def build_business_risk_profile(nodes: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    work = pairs.copy()
    group_fields = outcome_comparison_fields(work)
    if group_fields == ["outcome_group_id"]:
        work["_comparison_group_id"] = work["outcome_group_id"].map(clean_cell)
    elif group_fields == ["node_id", "sample_day"]:
        work["_comparison_group_id"] = work["node_id"].map(clean_cell) + "|" + work["sample_day"].map(clean_cell)
    else:
        work["_comparison_group_id"] = work["node_id"].map(clean_cell)
    best = work.loc[work.groupby("_comparison_group_id", dropna=False)["combined_score"].idxmax()].copy()
    profile_fields = [field for field in BUSINESS_RISK_PROFILE_FIELDS if field in best.columns or field in nodes.columns]
    missing_fields = [field for field in profile_fields if field not in best.columns]
    if missing_fields:
        node_profile = nodes[["node_id"] + missing_fields].drop_duplicates("node_id")
        best = best.merge(node_profile, on="node_id", how="left", sort=False)
    rows: list[dict[str, Any]] = []
    for (business, business_name), group in best.groupby(["business", "business_name"], dropna=False):
        item: dict[str, Any] = {
            "business": clean_cell(business),
            "business_name": clean_cell(business_name),
            "support_nodes": int(group["node_id"].nunique()),
            "combined_score_mean": float(pd.to_numeric(group["combined_score"], errors="coerce").mean()),
            "cum_cost_7d_mean": float(pd.to_numeric(group["cum_cost_7d"], errors="coerce").mean()),
            "cum_revenue_7d_mean": float(pd.to_numeric(group["cum_revenue_7d"], errors="coerce").mean()),
            "cum_profit_7d_mean": float(pd.to_numeric(group["cum_profit_7d"], errors="coerce").mean()),
        }
        for field in profile_fields:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            item[f"{field}_count"] = int(len(values))
            if values.empty:
                item[f"{field}_p05"] = ""
                item[f"{field}_p50"] = ""
                item[f"{field}_p90"] = ""
                item[f"{field}_p95"] = ""
                item[f"{field}_max"] = ""
                continue
            item[f"{field}_p05"] = float(values.quantile(0.05))
            item[f"{field}_p50"] = float(values.quantile(0.50))
            item[f"{field}_p90"] = float(values.quantile(0.90))
            item[f"{field}_p95"] = float(values.quantile(0.95))
            item[f"{field}_max"] = float(values.max())
        rows.append(item)
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(["support_nodes", "business"], ascending=[False, True])
    return output


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def command_build(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_field_contract(args.field_contract)
    nodes, outcomes = read_sources(args.nodes, args.outcomes)
    business_name_map = build_business_name_map(outcomes, args.business_map)
    outcomes = apply_business_names(outcomes, business_name_map)
    if args.limit_nodes:
        keep_nodes = set(nodes["node_id"].head(args.limit_nodes))
        nodes = nodes[nodes["node_id"].isin(keep_nodes)].copy()
        outcomes = outcomes[outcomes["node_id"].isin(keep_nodes)].copy()
    current_business = None
    if args.current_business:
        current_business = read_current_business(args.current_business, business_name_map=business_name_map)
        current_business = current_business[current_business["node_id"].isin(set(nodes["node_id"]))].copy()

    pairs = build_training_pairs(nodes, outcomes, contract, target_mode=args.target_mode)
    pair_node_ids = sorted(pairs["node_id"].dropna().astype(str).unique().tolist())
    train_nodes, test_nodes = split_node_ids(pair_node_ids, args.train_ratio, args.seed)
    train_pairs = pairs[pairs["node_id"].isin(train_nodes)].copy()
    test_pairs = pairs[pairs["node_id"].isin(test_nodes)].copy()
    train_node_frame = nodes[nodes["node_id"].isin(train_nodes)].copy()
    test_node_frame = nodes[nodes["node_id"].isin(test_nodes)].copy()

    model = fit_score_model(
        train_pairs,
        contract,
        min_business_support=args.min_business_support,
        smoothing_alpha=args.smoothing_alpha,
    )
    model["target_mode"] = args.target_mode
    if args.target_mode == TARGET_MODE_UNIT_BANDWIDTH:
        model["objective"] = (
            "combined_score = 0.5 * robust_normalized_cost_per_bandwidth_mbps "
            "+ 0.5 * robust_normalized_profit_per_bandwidth_mbps"
        )
    elif args.target_mode == TARGET_MODE_HYBRID:
        model["objective"] = (
            "combined_score = 0.5 miner + 0.5 operator; each side blends absolute within-group "
            "score 50% and unit-bandwidth score 50%"
        )
    train_metrics = evaluate_recommendations(train_pairs, train_node_frame, model, args.top_k)
    test_metrics = evaluate_recommendations(test_pairs, test_node_frame, model, args.top_k)
    observed_by_node = observed_scores_by_node(pairs, args.top_k)
    all_recommendations = recommendations_for_nodes(nodes, model, args.top_k, observed_by_node=observed_by_node)
    corrections = build_correction_candidates(
        nodes,
        pairs,
        all_recommendations,
        model,
        min_switch_gap=args.min_switch_gap,
        top_k=args.top_k,
        current_business=current_business,
    )
    risk_profile = build_business_risk_profile(nodes, pairs)

    training_path = output_dir / OUTPUT_TRAINING
    model_path = output_dir / OUTPUT_MODEL
    rec_path = output_dir / OUTPUT_RECOMMENDATIONS
    correction_path = output_dir / OUTPUT_CORRECTIONS
    metrics_path = output_dir / OUTPUT_METRICS
    business_map_path = output_dir / OUTPUT_BUSINESS_MAP
    risk_profile_path = output_dir / OUTPUT_BUSINESS_RISK_PROFILE

    pairs.to_csv(training_path, index=False)
    all_recommendations.to_csv(rec_path, index=False)
    corrections.to_csv(correction_path, index=False)
    business_map_frame(business_name_map).to_csv(business_map_path, index=False)
    risk_profile.to_csv(risk_profile_path, index=False)
    write_json(model_path, model)
    metrics = {
        "version": "v1",
        "objective": model["objective"],
        "target_mode": args.target_mode,
        "train": train_metrics,
        "test": test_metrics,
        "diagnostics": {
            "node_rows": int(len(nodes)),
            "outcome_rows": int(len(outcomes)),
            "training_pair_rows": int(len(pairs)),
            "train_nodes": int(len(train_nodes)),
            "test_nodes": int(len(test_nodes)),
            "candidate_businesses": int(len(model["candidate_businesses"])),
            "min_business_support": int(args.min_business_support),
            "segment_levels": model["segment_levels"],
            "recommendation_mode_counts": all_recommendations["recommendation_mode"].value_counts().to_dict(),
            "correction_action_counts": corrections["suggested_action"].value_counts().to_dict(),
            "current_business_source_counts": corrections["current_business_source"].value_counts(dropna=False).to_dict(),
            "current_business_confidence_counts": (
                corrections["current_business_confidence"].value_counts(dropna=False).to_dict()
            ),
            "business_name_map_rows": int(len(business_name_map)),
            "business_risk_profile_rows": int(len(risk_profile)),
            "current_business_source": (
                str(args.current_business) if args.current_business else CURRENT_BUSINESS_HISTORY_SOURCE
            ),
            "current_business_warning": (
                "Correction output uses the provided current business file."
                if args.current_business
                else (
                    "Correction output uses historical latest business_online_day as current business policy."
                )
            ),
            "output_files": {
                "training_pairs": str(training_path),
                "model": str(model_path),
                "recommendations": str(rec_path),
                "correction_candidates": str(correction_path),
                "business_name_map": str(business_map_path),
                "business_risk_profile": str(risk_profile_path),
            },
        },
    }
    write_json(metrics_path, metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def command_recommend_node(args: argparse.Namespace) -> int:
    with args.model.open(encoding="utf-8") as handle:
        model = json.load(handle)
    nodes = pd.read_csv(args.nodes, dtype={"node_id": "string"}, low_memory=False)
    nodes["node_id"] = nodes["node_id"].map(clean_cell)
    node_id = clean_cell(args.node_id)
    node = nodes[nodes["node_id"] == node_id]
    if node.empty:
        print(f"node_id not found: {args.node_id}", file=sys.stderr)
        return 1
    row = node.iloc[0].to_dict()
    scores: list[Score] = []
    recommendation_mode = "model_segment_fallback"
    if not args.model_only and args.outcomes.exists():
        outcomes = read_outcomes_for_recommendation(args.outcomes)
        if args.business_map:
            outcomes = apply_business_names(outcomes, build_business_name_map(outcomes, args.business_map))
        outcomes["node_id"] = outcomes["node_id"].map(clean_cell)
        outcomes = outcomes[outcomes["node_id"] == node_id].copy()
        scores = observed_scores_by_node(outcomes, args.top_k).get(node_id, [])
        if scores:
            recommendation_mode = "observed_node_outcome"
    if not scores:
        scores = score_node(row, model, args.top_k)
    risk_level, risk_reasons, missing = combine_risk(row, model)
    payload = {
        "node_id": node_id,
        "recommendation_mode": recommendation_mode,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "missing_profile_field_count": len(missing),
        "missing_profile_fields": missing,
        "top3": [
            {
                "business": score.business,
                "business_name": score.business_name,
                "score": score.score,
                "source": score.source,
                "support": score.support,
                "miner_score": score_value(score.miner_score),
                "operator_score": score_value(score.operator_score),
                "cum_cost_7d": score_value(score.cum_cost_7d),
                "cum_revenue_7d": score_value(score.cum_revenue_7d),
                "cum_profit_7d": score_value(score.cum_profit_7d),
                "outcome_distinct_days": score_value(score.outcome_distinct_days),
                "reason": score_reason(score),
            }
            for score in scores
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"node_id: {payload['node_id']}")
        print(f"mode: {payload['recommendation_mode']}")
        print(f"risk: {payload['risk_level']} {payload['risk_reasons']}".rstrip())
        for index, score in enumerate(payload["top3"], 1):
            name = f" ({score['business_name']})" if score["business_name"] else ""
            print(
                f"{index}. {score['business']}{name} "
                f"score={score['score']:.6g} source={score['source']} support={score['support']}"
            )
            print(f"   reason: {score['reason']}")
    return 0


def command_check(args: argparse.Namespace) -> int:
    required = [
        args.output_dir / OUTPUT_TRAINING,
        args.output_dir / OUTPUT_MODEL,
        args.output_dir / OUTPUT_RECOMMENDATIONS,
        args.output_dir / OUTPUT_CORRECTIONS,
        args.output_dir / OUTPUT_METRICS,
        args.output_dir / OUTPUT_BUSINESS_MAP,
        args.output_dir / OUTPUT_BUSINESS_RISK_PROFILE,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 1

    rec = pd.read_csv(args.output_dir / OUTPUT_RECOMMENDATIONS, nrows=100)
    corrections = pd.read_csv(args.output_dir / OUTPUT_CORRECTIONS, nrows=100)
    with (args.output_dir / OUTPUT_METRICS).open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    required_rec_cols = {
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
    required_corr_cols = {
        "node_id",
        "current_business",
        "current_business_source",
        "current_business_score_source",
        "current_eq_top1",
        "current_in_top3",
        "suggested_action",
        "reason_summary",
    }
    errors = []
    if not required_rec_cols.issubset(rec.columns):
        errors.append(f"recommendations missing columns: {sorted(required_rec_cols - set(rec.columns))}")
    if not required_corr_cols.issubset(corrections.columns):
        errors.append(f"corrections missing columns: {sorted(required_corr_cols - set(corrections.columns))}")
    if metrics.get("test", {}).get("n_nodes", 0) <= 0:
        errors.append("metrics test.n_nodes is zero")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("V1 artifacts check passed")
    print(json.dumps({
        "test": metrics.get("test"),
        "files": [str(path) for path in required],
    }, ensure_ascii=False, indent=2))
    return 0


def scalar_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return clean_cell(value)


def scalar_float_or_empty(value: Any) -> float | str:
    text = scalar_text(value)
    if not text:
        return ""
    try:
        parsed = float(text)
    except ValueError:
        return ""
    if not math.isfinite(parsed):
        return ""
    return parsed


def scalar_bool(value: Any) -> bool:
    text = scalar_text(value).lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def first_nonempty_text(*values: Any) -> str:
    for value in values:
        text = scalar_text(value)
        if text:
            return text
    return ""


def first_row_by_node_id(path: Path, node_id: str) -> dict[str, Any] | None:
    frame = pd.read_csv(path, dtype={"node_id": "string"}, low_memory=False)
    frame["node_id"] = frame["node_id"].map(clean_cell)
    matched = frame[frame["node_id"] == node_id]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def audit_payload(
    node_id: str,
    node_row: dict[str, Any],
    recommendation_row: dict[str, Any],
    correction_row: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    profile = {
        field: scalar_text(node_row.get(field))
        for field in PROFILE_SUMMARY_FIELDS
        if field in node_row and scalar_text(node_row.get(field)) != ""
    }
    top3 = []
    for index in range(1, top_k + 1):
        business = scalar_text(recommendation_row.get(f"recommended_business_top{index}"))
        if not business:
            continue
        top3.append({
            "rank": index,
            "business": business,
            "business_name": scalar_text(recommendation_row.get(f"recommended_business_name_top{index}")),
            "score": scalar_float_or_empty(recommendation_row.get(f"recommended_score_top{index}")),
            "source": scalar_text(recommendation_row.get(f"recommended_source_top{index}")),
            "support": scalar_float_or_empty(recommendation_row.get(f"recommended_support_top{index}")),
            "miner_score": scalar_float_or_empty(recommendation_row.get(f"recommended_miner_score_top{index}")),
            "operator_score": scalar_float_or_empty(recommendation_row.get(f"recommended_operator_score_top{index}")),
            "cum_cost_7d": scalar_float_or_empty(recommendation_row.get(f"recommended_cum_cost_7d_top{index}")),
            "cum_revenue_7d": scalar_float_or_empty(recommendation_row.get(f"recommended_cum_revenue_7d_top{index}")),
            "cum_profit_7d": scalar_float_or_empty(recommendation_row.get(f"recommended_cum_profit_7d_top{index}")),
            "outcome_distinct_days": scalar_float_or_empty(
                recommendation_row.get(f"recommended_outcome_days_top{index}")
            ),
            "reason": scalar_text(recommendation_row.get(f"recommendation_reason_top{index}")),
        })

    return {
        "node_id": node_id,
        "profile": profile,
        "current_business": {
            "business": scalar_text(correction_row.get("current_business")),
            "business_name": scalar_text(correction_row.get("current_business_name")),
            "score": scalar_float_or_empty(correction_row.get("current_business_score")),
            "score_source": scalar_text(correction_row.get("current_business_score_source")),
            "source": scalar_text(correction_row.get("current_business_source")),
            "confidence": scalar_text(correction_row.get("current_business_confidence")),
            "day": first_nonempty_text(
                correction_row.get("current_business_day"),
                correction_row.get("current_business_proxy_day"),
            ),
            "eq_top1": scalar_bool(correction_row.get("current_eq_top1")),
            "in_top3": scalar_bool(correction_row.get("current_in_top3")),
        },
        "recommendation": {
            "mode": scalar_text(recommendation_row.get("recommendation_mode")),
            "top3": top3,
        },
        "decision": {
            "suggested_action": scalar_text(correction_row.get("suggested_action")),
            "reason_summary": scalar_text(correction_row.get("reason_summary")),
            "recommended_minus_current_score": scalar_float_or_empty(
                correction_row.get("recommended_minus_current_score")
            ),
        },
        "risk": {
            "level": first_nonempty_text(recommendation_row.get("risk_level"), correction_row.get("risk_level")),
            "reasons": first_nonempty_text(recommendation_row.get("risk_reasons"), correction_row.get("risk_reasons")),
            "missing_profile_field_count": scalar_float_or_empty(
                recommendation_row.get("missing_profile_field_count")
            ),
            "missing_profile_fields": scalar_text(recommendation_row.get("missing_profile_fields")),
        },
    }


def command_audit_node(args: argparse.Namespace) -> int:
    node_id = clean_cell(args.node_id)
    nodes = pd.read_csv(args.nodes, dtype={"node_id": "string"}, low_memory=False)
    nodes["node_id"] = nodes["node_id"].map(clean_cell)
    node = nodes[nodes["node_id"] == node_id]
    if node.empty:
        print(f"node_id not found in nodes: {args.node_id}", file=sys.stderr)
        return 1
    rec_path = args.output_dir / OUTPUT_RECOMMENDATIONS
    correction_path = args.output_dir / OUTPUT_CORRECTIONS
    recommendation = first_row_by_node_id(rec_path, node_id)
    correction = first_row_by_node_id(correction_path, node_id)
    if recommendation is None:
        print(f"node_id not found in recommendations: {args.node_id}", file=sys.stderr)
        return 1
    if correction is None:
        print(f"node_id not found in corrections: {args.node_id}", file=sys.stderr)
        return 1
    payload = audit_payload(node_id, node.iloc[0].to_dict(), recommendation, correction, args.top_k)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"node_id: {payload['node_id']}")
        print(f"action: {payload['decision']['suggested_action']}")
        print(f"reason: {payload['decision']['reason_summary']}")
        print(
            "current: "
            f"{payload['current_business']['business']} "
            f"{payload['current_business']['business_name']} "
            f"score={payload['current_business']['score']} "
            f"source={payload['current_business']['source']} "
            f"confidence={payload['current_business']['confidence']}"
        )
        print(f"risk: {payload['risk']['level']} {payload['risk']['reasons']}".rstrip())
        print("top3:")
        for item in payload["recommendation"]["top3"]:
            name = f" ({item['business_name']})" if item["business_name"] else ""
            print(
                f"{item['rank']}. {item['business']}{name} "
                f"score={item['score']} source={item['source']}"
            )
    return 0


def command_correction_candidates(args: argparse.Namespace) -> int:
    corrections = pd.read_csv(args.output_dir / OUTPUT_CORRECTIONS, dtype={"node_id": "string"}, low_memory=False)
    if args.action:
        corrections = corrections[corrections["suggested_action"].fillna("") == args.action].copy()
    if args.risk_level:
        corrections = corrections[corrections["risk_level"].fillna("") == args.risk_level].copy()
    if "action_priority" in corrections.columns:
        corrections["action_priority"] = pd.to_numeric(corrections["action_priority"], errors="coerce").fillna(99)
    if "recommended_minus_current_score" in corrections.columns:
        corrections["recommended_minus_current_score"] = pd.to_numeric(
            corrections["recommended_minus_current_score"],
            errors="coerce",
        )
    sort_columns = [column for column in ["action_priority", "recommended_minus_current_score"] if column in corrections]
    if sort_columns:
        ascending = [True, False][: len(sort_columns)]
        corrections = corrections.sort_values(sort_columns, ascending=ascending)
    if args.limit:
        corrections = corrections.head(args.limit)
    if args.json:
        print(corrections.to_json(orient="records", force_ascii=False, indent=2))
    else:
        columns = [
            "node_id",
            "current_business",
            "current_business_name",
            "combined_top1",
            "recommended_business_name_top1",
            "recommended_minus_current_score",
            "suggested_action",
            "risk_level",
            "reason_summary",
        ]
        existing = [column for column in columns if column in corrections.columns]
        print(corrections[existing].to_csv(index=False))
    return 0


def parse_where_clauses(clauses: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for clause in clauses:
        if "=" not in clause:
            raise ValueError(f"invalid --where clause, expected field=value: {clause}")
        field, value = clause.split("=", 1)
        field = field.strip()
        if not field:
            raise ValueError(f"invalid --where field: {clause}")
        filters[field] = clean_cell(value)
    return filters


def filter_nodes(nodes: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    frame = nodes.copy()
    for field, expected in filters.items():
        if field not in frame.columns:
            raise KeyError(f"unknown node field: {field}")
        series = frame[field].map(clean_cell)
        frame = frame[series == expected].copy()
    return frame


def command_recommend_filter(args: argparse.Namespace) -> int:
    with args.model.open(encoding="utf-8") as handle:
        model = json.load(handle)
    nodes = pd.read_csv(args.nodes, dtype={"node_id": "string"}, low_memory=False)
    nodes["node_id"] = nodes["node_id"].map(clean_cell)
    try:
        filters = parse_where_clauses(args.where or [])
        matched = filter_nodes(nodes, filters)
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.limit:
        matched = matched.head(args.limit).copy()
    matched_ids = set(matched["node_id"].map(clean_cell))
    outcomes = read_outcomes_for_recommendation(args.outcomes)
    if args.business_map:
        outcomes = apply_business_names(outcomes, build_business_name_map(outcomes, args.business_map))
    outcomes["node_id"] = outcomes["node_id"].map(clean_cell)
    outcomes = outcomes[outcomes["node_id"].isin(matched_ids)].copy()
    observed = observed_scores_by_node(outcomes, args.top_k)
    recommendations = recommendations_for_nodes(matched, model, args.top_k, observed_by_node=observed)
    if args.json:
        print(recommendations.to_json(orient="records", force_ascii=False, indent=2))
    else:
        columns = [
            "node_id",
            "recommendation_mode",
            "combined_top1",
            "combined_score_top1",
            "combined_top2",
            "combined_score_top2",
            "combined_top3",
            "combined_score_top3",
            "risk_level",
            "risk_reasons",
        ]
        existing = [column for column in columns if column in recommendations.columns]
        print(recommendations[existing].to_csv(index=False))
    return 0


def command_derive_current_business(args: argparse.Namespace) -> int:
    wide = pd.read_csv(args.wide_csv, dtype="string", low_memory=False)
    current = derive_current_business_from_wide(
        wide,
        lookback_days=args.lookback_days,
        min_active_days=args.min_active_days,
        online_only=not args.include_offline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(args.output, index=False)
    print(json.dumps({
        "output": str(args.output),
        "rows": int(len(current)),
        "source": "derived_from_node_day_ops_wide",
        "lookback_days": int(args.lookback_days),
        "confidence_counts": current.get(
            "current_business_confidence",
            pd.Series(dtype="string"),
        ).value_counts(dropna=False).to_dict(),
    }, ensure_ascii=False, indent=2))
    return 0


def command_export_business_map(args: argparse.Namespace) -> int:
    outcomes = pd.read_csv(
        args.outcomes,
        dtype={"business": "string", "business_name": "string"},
        usecols=lambda column: column in {"business", "business_name"},
        low_memory=False,
    )
    mapping = build_business_name_map(outcomes, args.business_map)
    frame = business_map_frame(mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(json.dumps({
        "output": str(args.output),
        "rows": int(len(frame)),
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and use V1 node business recommendation artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build V1 training, model, recommendation, and correction artifacts.")
    build.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    build.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    build.add_argument("--field-contract", type=Path, default=DEFAULT_FIELD_CONTRACT)
    build.add_argument("--current-business", type=Path, help="Optional authoritative current business CSV.")
    build.add_argument("--business-map", type=Path, help="Optional business id/name mapping CSV.")
    build.add_argument("--output-dir", type=Path, default=HERE)
    build.add_argument("--train-ratio", type=float, default=0.8)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--min-business-support", type=int, default=30)
    build.add_argument("--smoothing-alpha", type=float, default=20.0)
    build.add_argument("--min-switch-gap", type=float, default=0.05)
    build.add_argument("--top-k", type=int, default=3)
    build.add_argument("--limit-nodes", type=int, help="Optional small-sample build for smoke testing.")
    build.add_argument(
        "--target-mode",
        choices=sorted(TARGET_MODES),
        default=TARGET_MODE_ABSOLUTE,
        help="Training target: absolute amounts, unit-bandwidth efficiency, or a 50/50 hybrid.",
    )
    build.set_defaults(func=command_build)

    recommend = subparsers.add_parser("recommend-node", help="Recommend Top3 business for one local node.")
    recommend.add_argument("node_id")
    recommend.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    recommend.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    recommend.add_argument("--model", type=Path, default=HERE / OUTPUT_MODEL)
    recommend.add_argument("--business-map", type=Path, help="Optional business id/name mapping CSV.")
    recommend.add_argument("--top-k", type=int, default=3)
    recommend.add_argument("--model-only", action="store_true", help="Ignore observed local node-business outcomes.")
    recommend.add_argument("--json", action="store_true")
    recommend.set_defaults(func=command_recommend_node)

    recommend_filter = subparsers.add_parser(
        "recommend-filter",
        help="Recommend Top3 businesses for local nodes matched by one or more field=value filters.",
    )
    recommend_filter.add_argument("--where", action="append", required=True, help="Exact node filter, e.g. province=江苏")
    recommend_filter.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    recommend_filter.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    recommend_filter.add_argument("--model", type=Path, default=HERE / OUTPUT_MODEL)
    recommend_filter.add_argument("--business-map", type=Path, help="Optional business id/name mapping CSV.")
    recommend_filter.add_argument("--top-k", type=int, default=3)
    recommend_filter.add_argument("--limit", type=int, default=20)
    recommend_filter.add_argument("--json", action="store_true")
    recommend_filter.set_defaults(func=command_recommend_filter)

    audit = subparsers.add_parser(
        "audit-node",
        help="Show node profile, current business, Top3 recommendation, action, and risk for one node.",
    )
    audit.add_argument("node_id")
    audit.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    audit.add_argument("--output-dir", type=Path, default=HERE)
    audit.add_argument("--top-k", type=int, default=3)
    audit.add_argument("--json", action="store_true")
    audit.set_defaults(func=command_audit_node)

    candidates = subparsers.add_parser(
        "correction-candidates",
        help="List existing-node correction candidates from generated V1 artifacts.",
    )
    candidates.add_argument("--output-dir", type=Path, default=HERE)
    candidates.add_argument("--action", help="Filter by suggested_action, e.g. review_switch_candidate.")
    candidates.add_argument("--risk-level", help="Filter by risk_level, e.g. high.")
    candidates.add_argument("--limit", type=int, default=20)
    candidates.add_argument("--json", action="store_true")
    candidates.set_defaults(func=command_correction_candidates)

    derive = subparsers.add_parser(
        "derive-current-business",
        help="Derive one current business per node from a Superset node_day_ops_wide CSV export.",
    )
    derive.add_argument("wide_csv", type=Path)
    derive.add_argument("--output", type=Path, default=HERE / OUTPUT_CURRENT_FROM_WIDE)
    derive.add_argument("--lookback-days", type=int, default=7)
    derive.add_argument("--min-active-days", type=int, default=3)
    derive.add_argument("--include-offline", action="store_true", help="Do not filter state=online rows.")
    derive.set_defaults(func=command_derive_current_business)

    business_map = subparsers.add_parser("export-business-map", help="Export known business id/name mappings.")
    business_map.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    business_map.add_argument("--business-map", type=Path, help="Optional external mapping CSV to merge.")
    business_map.add_argument("--output", type=Path, default=HERE / OUTPUT_BUSINESS_MAP)
    business_map.set_defaults(func=command_export_business_map)

    check = subparsers.add_parser("check", help="Check generated V1 artifacts.")
    check.add_argument("--output-dir", type=Path, default=HERE)
    check.set_defaults(func=command_check)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
