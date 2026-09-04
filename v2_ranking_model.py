#!/usr/bin/env python3
"""V2 profile-based ranking model for new-node business recommendation.

This is a dependency-light ranking baseline for nodes with no observed
node-business outcome. It learns from historical node-business pairs, then
scores candidate businesses from node profile segments.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v1_recommendation_pipeline as v1


HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "v1_training_pairs.csv"
DEFAULT_NODES = HERE / "multibusiness_nodes.csv"
DEFAULT_FIELD_CONTRACT = v1.DEFAULT_FIELD_CONTRACT
DEFAULT_RISK_PROFILE = HERE / v1.OUTPUT_BUSINESS_RISK_PROFILE

OUTPUT_MODEL = "v2_ranking_model.json"
OUTPUT_RECOMMENDATIONS = "v2_node_recommendations.csv"
OUTPUT_METRICS = "v2_model_metrics.json"
OUTPUT_FRONTEND_REPORT = "v2_frontend_condition_business_report.csv"
OUTPUT_FRONTEND_REPORT_JSON = "v2_frontend_condition_business_report.json"
OUTPUT_FRONTEND_REPORT_DATA_JS = "v2_frontend_condition_business_report_data.js"
UNKNOWN_BUSINESS_NAME = "未命名业务"
NODE_SIZE_LARGE = "large_node"
NODE_SIZE_SMALL_BOX = "small_box"
NODE_SIZE_UNKNOWN = "unknown"

DEFAULT_CONDITION_SEGMENTS = [
    ("province_isp", ["province", "isp"]),
    ("province_isp_resource", ["province", "isp", "resourcetype"]),
    ("province_isp_city_resource", ["province", "isp", "city", "resourcetype"]),
    ("province_isp_resource_network", ["province", "isp", "resourcetype", "nattype", "dialtype"]),
    (
        "province_isp_resource_delivery_device",
        ["province", "isp", "resourcetype", "deliverytype", "device_type"],
    ),
    (
        "province_isp_resource_bandwidth",
        ["province", "isp", "resourcetype", "bw_bucket", "actualbandwidth_bucket"],
    ),
    (
        "province_isp_resource_schedule",
        ["province", "isp", "resourcetype", "scheduleisps", "analysis_transprovrate_bucket"],
    ),
    (
        "province_isp_resource_network_schedule",
        ["province", "isp", "resourcetype", "network_schedule_type"],
    ),
    (
        "province_isp_resource_schedule_flags",
        ["province", "isp", "resourcetype", "scheduleisps", "join_isbantransprov", "join_isipv6schedule"],
    ),
    (
        "province_isp_city_resource_network_bandwidth",
        ["province", "isp", "city", "resourcetype", "nattype", "dialtype", "bw_bucket"],
    ),
]

RANK_SCORE_WEIGHTS = {
    "source_rate": 0.60,
    "champion": 0.20,
    "expected_best_rate": 0.15,
    "expected_score": 0.05,
}
UNIT_BANDWIDTH_RANK_SCORE_WEIGHTS = {
    "source_rate": 0.20,
    "champion": 0.10,
    "expected_best_rate": 0.20,
    "expected_score": 0.50,
}

RISK_SEVERITY = {"low": 0, "unknown": 1, "medium": 2, "high": 3}
RESOURCE_FLOOR_FIELDS = ["bw", "actualbandwidth", "corenum", "memtotal", "totaldisksize"]
QUALITY_CEILING_FIELDS = [
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

NUMERIC_BUCKET_FIELDS = {
    "bw": [100, 500, 1000, 3000, 5000],
    "actualbandwidth": [100, 500, 1000, 3000, 5000],
    "corenum": [4, 8, 16, 32, 64],
    "memtotal": [1_000_000, 4_000_000, 8_000_000, 32_000_000, 64_000_000],
    "totaldisksize": [1e12, 4e12, 8e12, 16e12, 32e12],
    "quality_retransrate": [1, 3, 5, 10, 20],
    "quality_pinglossrate": [1, 3, 10, 30, 60],
    "quality_rtt": [20, 50, 100, 200, 500],
    "prom_retrans_ratio": [1, 3, 5, 10, 20],
    "disk_used_ratio": [0.5, 0.7, 0.85, 0.95, 0.99],
    "cpu_load1_per_core": [1, 2, 4, 8, 16],
    "analysis_transprovrate": [0, 100],
}

V2_SEGMENT_LEVELS = [
    ("province_isp_resourcetype", ["province", "isp", "resourcetype"], 1.40),
    (
        "province_isp_resource_scheduleisp",
        ["province", "isp", "resourcetype", "scheduleisps"],
        1.30,
    ),
    (
        "province_isp_resource_network_schedule",
        ["province", "isp", "resourcetype", "network_schedule_type"],
        1.30,
    ),
    ("province_isp_bw", ["province", "isp", "bw_bucket"], 1.25),
    (
        "province_isp_resource_transprov",
        ["province", "isp", "resourcetype", "analysis_transprovrate_bucket"],
        1.20,
    ),
    ("province_isp", ["province", "isp"], 1.15),
    ("isp_resourcetype", ["isp", "resourcetype"], 1.05),
    ("province_resourcetype", ["province", "resourcetype"], 1.00),
    ("isp_nattype_dialtype", ["isp", "nattype", "dialtype"], 0.95),
    (
        "isp_scheduleisp_transprov",
        ["isp", "scheduleisps", "analysis_transprovrate_bucket"],
        0.90,
    ),
    (
        "isp_resource_network_schedule",
        ["isp", "resourcetype", "network_schedule_type"],
        0.95,
    ),
    ("resourcetype_bw", ["resourcetype", "bw_bucket"], 0.90),
    ("province", ["province"], 0.70),
    ("isp", ["isp"], 0.70),
    ("resourcetype", ["resourcetype"], 0.65),
    ("nattype", ["nattype"], 0.55),
    ("dialtype", ["dialtype"], 0.50),
    ("deliverytype", ["deliverytype"], 0.50),
    ("device_type", ["device_type"], 0.45),
    ("scheduleisps", ["scheduleisps"], 0.45),
    ("network_schedule_type", ["network_schedule_type"], 0.55),
    ("schedule_isp_scope", ["schedule_isp_scope"], 0.35),
    ("schedule_province_scope", ["schedule_province_scope"], 0.35),
    ("bw_bucket", ["bw_bucket"], 0.40),
    ("analysis_transprovrate_bucket", ["analysis_transprovrate_bucket"], 0.35),
    ("join_isbantransprov", ["join_isbantransprov"], 0.35),
    ("join_isipv6schedule", ["join_isipv6schedule"], 0.25),
    ("quality_retransrate_bucket", ["quality_retransrate_bucket"], 0.30),
    ("quality_pinglossrate_bucket", ["quality_pinglossrate_bucket"], 0.30),
    ("quality_rtt_bucket", ["quality_rtt_bucket"], 0.30),
    ("disk_used_ratio_bucket", ["disk_used_ratio_bucket"], 0.25),
]


def clean_cell(value: Any) -> str:
    return v1.clean_cell(value)


def is_generated_business_id_label(value: Any) -> bool:
    return clean_cell(value).startswith("business_id:")


def score_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return parsed


def optional_number(value: Any) -> float | None:
    text = clean_cell(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def bucket_value(value: Any, edges: list[float]) -> str:
    text = clean_cell(value)
    if not text:
        return v1.UNKNOWN
    numeric = score_value(text)
    for edge in edges:
        if numeric <= edge:
            return f"<= {edge:g}"
    return f"> {edges[-1]:g}"


CARRIER_ALIASES = {
    "电信": "电信",
    "中国电信": "电信",
    "ctcc": "电信",
    "telecom": "电信",
    "联通": "联通",
    "中国联通": "联通",
    "cucc": "联通",
    "unicom": "联通",
    "移动": "移动",
    "中国移动": "移动",
    "cmcc": "移动",
    "mobile": "移动",
}


def normalize_carrier(value: Any) -> str:
    text = clean_cell(value).lower()
    return CARRIER_ALIASES.get(text, "")


def parse_schedule_isps(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        text = clean_cell(value)
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                raw_values = parsed if isinstance(parsed, list) else [text]
            except json.JSONDecodeError:
                raw_values = re.split(r"[,|;/、\s]+", text.strip("[]"))
        else:
            raw_values = re.split(r"[,|;/、\s]+", text)
    carriers = [normalize_carrier(item) for item in raw_values]
    return list(dict.fromkeys(carrier for carrier in carriers if carrier))


def canonical_schedule_isps(row: pd.Series | dict[str, Any]) -> str:
    """Return exact target carriers; an empty schedule means the node's own ISP."""
    schedule_value = row.get("scheduleisps_text")
    if not clean_cell(schedule_value):
        schedule_value = row.get("scheduleisps")
    target_isps = parse_schedule_isps(schedule_value)
    if not target_isps:
        local_isp = normalize_carrier(row.get("isp")) or normalize_carrier(row.get("join_isp"))
        target_isps = [local_isp] if local_isp else []
    return "|".join(sorted(set(target_isps)))


def effective_transprov_rate(row: pd.Series | dict[str, Any]) -> float | None:
    for field in ("analysis_transprovrate", "join_transprovrate"):
        text = clean_cell(row.get(field))
        if not text:
            continue
        value = optional_number(text)
        if value is None:
            return None
        if abs(value) < 1e-9:
            return 0.0
        if abs(value - 100.0) < 1e-9:
            return 100.0
        return None
    return 0.0


def infer_schedule_isp_scope(row: pd.Series | dict[str, Any]) -> str:
    local_isp = normalize_carrier(row.get("isp")) or normalize_carrier(row.get("join_isp"))
    if not local_isp:
        return "未知网络"
    target_isps = parse_schedule_isps(canonical_schedule_isps(row))
    has_local = local_isp in target_isps
    has_other = any(target != local_isp for target in target_isps)
    if has_local and has_other:
        return "混合网络"
    return "异网" if has_other else "本网"


def infer_schedule_province_scope(row: pd.Series | dict[str, Any]) -> str:
    rate = effective_transprov_rate(row)
    if rate is None:
        return "未知省份"
    return "本省" if rate == 0 else "出省"


def infer_network_schedule_type(row: pd.Series | dict[str, Any]) -> str:
    isp_scope = infer_schedule_isp_scope(row)
    province_scope = infer_schedule_province_scope(row)
    if isp_scope in {"本网", "异网"} and province_scope in {"本省", "出省"}:
        return f"{isp_scope}{province_scope}"
    return f"{isp_scope}+{province_scope}"


def max_profile_bandwidth(row: pd.Series | dict[str, Any]) -> float:
    return max(
        score_value(row.get("bw")),
        score_value(row.get("bandwidth")),
        score_value(row.get("actualbandwidth")),
    )


def infer_node_size_type(row: pd.Series | dict[str, Any]) -> str:
    delivery = clean_cell(row.get("deliverytype")).lower()
    resource = clean_cell(row.get("resourcetype")).lower()
    join_resource = clean_cell(row.get("join_resourcetype")).lower()
    supply_type = clean_cell(row.get("analysis_supply_side_delivery_type"))
    node_delivery = clean_cell(row.get("analysis_nodedeliverytype")).lower()
    device_type = clean_cell(row.get("device_type")).lower()

    if (
        delivery == "smallbox"
        or supply_type == "小盒子"
        or node_delivery in {"droid", "h618", "aml", "n1", "openwrt"}
        or device_type in {"micro.b", "micro.c", "droid.a", "droid.b"}
    ):
        return NODE_SIZE_SMALL_BOX

    bandwidth = max_profile_bandwidth(row)
    core_count = score_value(row.get("corenum"))
    if (
        resource == "dedicated"
        or join_resource == "dedicated"
        or delivery in {"idc", "dedicated"}
        or supply_type in {"机房", "专线"}
    ):
        return NODE_SIZE_LARGE
    if supply_type == "汇聚" and bandwidth >= 100:
        return NODE_SIZE_LARGE
    if device_type.startswith(("jarvis", "ant.")) and bandwidth >= 100:
        return NODE_SIZE_LARGE
    if bandwidth >= 1000 and core_count >= 8:
        return NODE_SIZE_LARGE
    return NODE_SIZE_UNKNOWN


def add_node_size_type(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["node_size_type"] = output.apply(infer_node_size_type, axis=1)
    return output


def add_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    output = add_node_size_type(frame)
    output["scheduleisps"] = output.apply(canonical_schedule_isps, axis=1)
    output["analysis_transprovrate"] = output.apply(effective_transprov_rate, axis=1)
    output["schedule_isp_scope"] = output.apply(infer_schedule_isp_scope, axis=1)
    output["schedule_province_scope"] = output.apply(infer_schedule_province_scope, axis=1)
    output["network_schedule_type"] = output.apply(infer_network_schedule_type, axis=1)
    for field, edges in NUMERIC_BUCKET_FIELDS.items():
        if field in output.columns:
            output[f"{field}_bucket"] = output[field].map(lambda value: bucket_value(value, edges))
    return output


def segment_key(row: pd.Series | dict[str, Any], fields: list[str]) -> str:
    return json.dumps([v1.safe_category(row.get(field, "")) for field in fields], ensure_ascii=False)


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if not math.isfinite(high - low) or abs(high - low) < 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def split_node_ids(node_ids: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    return v1.split_node_ids(node_ids, train_ratio, seed)


def latest_nonempty(series: pd.Series) -> str:
    return v1.latest_nonempty(series)


def prepare_pairs(path: Path, business_map: Path | None = None) -> pd.DataFrame:
    pairs = pd.read_csv(path, dtype={"node_id": "string", "business": "string"}, low_memory=False)
    pairs["node_id"] = pairs["node_id"].map(clean_cell)
    pairs["business"] = pairs["business"].map(clean_cell)
    if "business_name" not in pairs.columns:
        pairs["business_name"] = ""
    pairs["business_name"] = pairs["business_name"].fillna("").map(clean_cell)
    if business_map:
        mapping = v1.build_business_name_map(pairs[["business", "business_name"]], business_map)
        pairs = v1.apply_business_names(pairs, mapping)
    for column in [
        "combined_score",
        "miner_score_norm",
        "operator_score_norm",
        "cum_cost_7d",
        "cum_revenue_7d",
        "cum_profit_7d",
        "outcome_distinct_days",
        "sample_weight",
    ]:
        if column in pairs.columns:
            pairs[column] = pd.to_numeric(pairs[column], errors="coerce").fillna(0.0)
    pairs["sample_weight"] = v1.sample_weights(pairs)
    return add_buckets(pairs)


def load_feature_fields(path: Path) -> list[str]:
    if not path.exists():
        return []
    return v1.model_feature_fields(v1.load_field_contract(path))


def load_business_risk_profiles(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path, dtype="string", low_memory=False)
    if "business" not in frame.columns:
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        business = clean_cell(row.get("business"))
        if not business:
            continue
        profiles[business] = {column: clean_cell(value) for column, value in row.items()}
    return profiles


def candidate_businesses(pairs: pd.DataFrame, min_business_support: int) -> list[str]:
    work = pairs[["business"]].copy()
    work["sample_weight"] = v1.sample_weights(pairs)
    support = work.groupby("business", dropna=False)["sample_weight"].sum()
    return sorted(support[support >= min_business_support].index.astype(str).tolist())


def parse_rank_score_weights(text: str | None) -> dict[str, float]:
    if not text:
        return dict(RANK_SCORE_WEIGHTS)
    weights = dict(RANK_SCORE_WEIGHTS)
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"invalid rank score weight: {part}")
        key, value = part.split("=", 1)
        key = clean_cell(key)
        if key not in RANK_SCORE_WEIGHTS:
            raise ValueError(f"unknown rank score weight: {key}")
        parsed = float(value)
        if parsed < 0:
            raise ValueError(f"rank score weight must be non-negative: {key}")
        weights[key] = parsed
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("rank score weights sum to zero")
    return {key: float(value) / total for key, value in weights.items()}


def comparison_group_fields(pairs: pd.DataFrame) -> list[str]:
    if "outcome_group_id" in pairs.columns and pairs["outcome_group_id"].fillna("").astype(str).str.strip().ne("").any():
        return ["outcome_group_id"]
    if "sample_day" in pairs.columns and pairs["sample_day"].fillna("").astype(str).str.strip().ne("").any():
        return ["node_id", "sample_day"]
    return ["node_id"]


def add_comparison_group_id(pairs: pd.DataFrame) -> pd.DataFrame:
    output = pairs.copy()
    fields = comparison_group_fields(output)
    if fields == ["outcome_group_id"]:
        output["_comparison_group_id"] = output["outcome_group_id"].map(clean_cell)
    elif fields == ["node_id", "sample_day"]:
        output["_comparison_group_id"] = (
            output["node_id"].map(clean_cell) + "|" + output["sample_day"].map(clean_cell)
        )
    else:
        output["_comparison_group_id"] = output["node_id"].map(clean_cell)
    return output


def true_best_by_node(pairs: pd.DataFrame) -> pd.DataFrame:
    work = add_comparison_group_id(pairs)
    index = work.groupby("_comparison_group_id", dropna=False)["combined_score"].idxmax()
    return work.loc[index].copy()


def fit_v2_model(
    train_pairs: pd.DataFrame,
    min_business_support: int,
    smoothing_alpha: float,
    best_alpha: float,
    champion_min_support: int,
    feature_fields: list[str] | None = None,
    business_risk_profiles: dict[str, dict[str, Any]] | None = None,
    risk_min_profile_count: int = 20,
    risk_min_best_support: int = 3,
    rank_score_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    train = train_pairs.copy()
    train["sample_weight"] = v1.sample_weights(train)
    candidates = candidate_businesses(train, min_business_support)
    if not candidates:
        raise RuntimeError("No candidate business meets min support.")
    train = train[train["business"].isin(candidates)].copy()
    train = add_comparison_group_id(train)
    best = true_best_by_node(train)
    train_node_count = int(train["node_id"].nunique())
    train_group_count = int(train["_comparison_group_id"].nunique())

    global_rows = v1.weighted_group_summary(
        train,
        ["business"],
        [
            "combined_score",
            "miner_score_norm",
            "operator_score_norm",
            "cum_cost_7d",
            "cum_revenue_7d",
            "cum_profit_7d",
        ],
    ).rename(columns={
        "effective_support": "pair_support",
        "combined_score": "score",
        "miner_score_norm": "miner_score",
        "operator_score_norm": "operator_score",
    })
    global_names = (
        train.groupby("business", dropna=False)["business_name"]
        .agg(latest_nonempty)
        .rename("business_name")
        .reset_index()
    )
    global_rows = global_rows.merge(global_names, on="business", how="left")
    best_counts = best.groupby("business", dropna=False)["sample_weight"].sum().to_dict()
    train_group_weight = float(best["sample_weight"].sum())
    global_scores: dict[str, dict[str, Any]] = {}
    for row in global_rows.to_dict(orient="records"):
        business = clean_cell(row["business"])
        best_support = float(best_counts.get(business, 0.0))
        global_scores[business] = {
            "pair_support": float(row["pair_support"]),
            "raw_pair_support": int(row["raw_support"]),
            "best_support": best_support,
            "best_rate": best_support / max(train_group_weight, 1e-12),
            "score": float(row["score"]),
            "miner_score": float(row["miner_score"]),
            "operator_score": float(row["operator_score"]),
            "cum_cost_7d": float(row["cum_cost_7d"]),
            "cum_revenue_7d": float(row["cum_revenue_7d"]),
            "cum_profit_7d": float(row["cum_profit_7d"]),
            "business_name": clean_cell(row["business_name"]),
        }

    segment_payload: dict[str, dict[str, Any]] = {}
    available_levels: list[dict[str, Any]] = []
    for name, fields, weight in V2_SEGMENT_LEVELS:
        fields = [field for field in fields if field in train.columns]
        if not fields:
            continue
        available_levels.append({"name": name, "fields": fields, "weight": weight})
        pair_work = train.copy()
        pair_work["_segment_key"] = pair_work.apply(lambda row: segment_key(row, fields), axis=1)
        pair_stats = v1.weighted_group_summary(
            pair_work,
            ["_segment_key", "business"],
            ["combined_score"],
        ).rename(columns={
            "combined_score": "score",
            "effective_support": "pair_support",
        })
        best_work = best.copy()
        best_work["_segment_key"] = best_work.apply(lambda row: segment_key(row, fields), axis=1)
        segment_nodes = best_work.groupby("_segment_key")["sample_weight"].sum().to_dict()
        best_stats = (
            best_work.groupby(["_segment_key", "business"], dropna=False)
            .agg(
                best_support=("sample_weight", "sum"),
                raw_best_support=("_comparison_group_id", "nunique"),
            )
            .reset_index()
        )
        best_lookup = {
            (str(row["_segment_key"]), str(row["business"])): float(row["best_support"])
            for row in best_stats.to_dict(orient="records")
        }
        level_rows: dict[str, dict[str, Any]] = {}
        for row in pair_stats.to_dict(orient="records"):
            key = str(row["_segment_key"])
            business = clean_cell(row["business"])
            level_rows.setdefault(key, {})[business] = {
                "score": float(row["score"]),
                "pair_support": float(row["pair_support"]),
                "raw_pair_support": int(row["raw_support"]),
                "best_support": best_lookup.get((key, business), 0),
                "segment_nodes": float(segment_nodes.get(key, 0.0)),
            }
        for key, business_rows in level_rows.items():
            ranked = sorted(
                business_rows.items(),
                key=lambda item: (
                    -float(item[1].get("best_support", 0)),
                    -float(item[1].get("score", 0.0)),
                    item[0],
                ),
            )
            for rank, (business, info) in enumerate(ranked, 1):
                if float(info.get("best_support", 0)) >= champion_min_support:
                    info["champion_rank"] = rank
        segment_payload[name] = level_rows

    return {
        "version": "v2",
        "model_type": "profile_segment_ranker",
        "objective": "rank by combined_score plus historical best-business probability",
        "candidate_businesses": candidates,
        "train_node_count": train_node_count,
        "train_comparison_group_count": train_group_count,
        "train_effective_group_support": train_group_weight,
        "global_scores": global_scores,
        "segment_levels": available_levels,
        "segments": segment_payload,
        "smoothing_alpha": float(smoothing_alpha),
        "best_alpha": float(best_alpha),
        "champion_min_support": int(champion_min_support),
        "min_business_support": int(min_business_support),
        "sample_weight_policy": "1 / consecutive valid days in the same node-business run",
        "rank_score_weights": rank_score_weights or dict(RANK_SCORE_WEIGHTS),
        "feature_fields": feature_fields or [],
        "business_risk_profiles": business_risk_profiles or {},
        "risk_min_profile_count": int(risk_min_profile_count),
        "risk_min_best_support": int(risk_min_best_support),
        "bucket_edges": NUMERIC_BUCKET_FIELDS,
    }


def score_one_business(row: dict[str, Any], business: str, model: dict[str, Any]) -> dict[str, Any]:
    global_info = model["global_scores"][business]
    score_terms = [(float(global_info["score"]), 1.0, "global")]
    best_terms = [(float(global_info["best_rate"]), 1.0, "global")]
    alpha = float(model.get("smoothing_alpha", 30.0))
    best_alpha = float(model.get("best_alpha", 50.0))
    best_source = "global"
    best_source_support = float(global_info.get("best_support", 0))
    best_source_rate = float(global_info.get("best_rate", 0.0))
    best_source_weight = 1.0
    champion_score = 0.0
    champion_source = ""
    champion_source_weight = 0.0

    for level in model.get("segment_levels", []):
        key = segment_key(row, level["fields"])
        info = model.get("segments", {}).get(level["name"], {}).get(key, {}).get(business)
        if not info:
            continue
        weight = float(level.get("weight", 1.0))
        pair_support = float(info.get("pair_support", 0))
        segment_score = float(info.get("score", global_info["score"]))
        smoothed_score = (
            segment_score * pair_support + float(global_info["score"]) * alpha
        ) / (pair_support + alpha)
        score_terms.append((smoothed_score, weight * math.log1p(pair_support), level["name"]))

        segment_nodes = float(info.get("segment_nodes", 0))
        best_support = float(info.get("best_support", 0))
        prior_rate = float(global_info.get("best_rate", 0.0))
        smoothed_best_rate = (best_support + best_alpha * prior_rate) / (segment_nodes + best_alpha)
        best_weight = weight * math.log1p(segment_nodes)
        best_terms.append((smoothed_best_rate, best_weight, level["name"]))
        if best_weight > best_source_weight and best_support > 0:
            best_source = level["name"]
            best_source_support = best_support
            best_source_rate = smoothed_best_rate
            best_source_weight = best_weight
        champion_rank = int(info.get("champion_rank", 0) or 0)
        if champion_rank > 0:
            champion_contribution = weight * math.log1p(best_support) / math.sqrt(champion_rank)
            champion_score += champion_contribution
            if champion_contribution > champion_source_weight:
                champion_source = f"{level['name']}#rank{champion_rank}"
                champion_source_weight = champion_contribution

    score_weight_sum = sum(weight for _, weight, _ in score_terms)
    best_weight_sum = sum(weight for _, weight, _ in best_terms)
    expected_score = sum(value * weight for value, weight, _ in score_terms) / max(score_weight_sum, 1e-9)
    expected_best_rate = sum(value * weight for value, weight, _ in best_terms) / max(best_weight_sum, 1e-9)

    return {
        "business": business,
        "business_name": clean_cell(global_info.get("business_name")),
        "expected_score": expected_score,
        "expected_best_rate": expected_best_rate,
        "miner_score": float(global_info.get("miner_score", 0.0)),
        "operator_score": float(global_info.get("operator_score", 0.0)),
        "cum_cost_7d": float(global_info.get("cum_cost_7d", 0.0)),
        "cum_revenue_7d": float(global_info.get("cum_revenue_7d", 0.0)),
        "cum_profit_7d": float(global_info.get("cum_profit_7d", 0.0)),
        "source": best_source,
        "support": best_source_support,
        "source_best_rate": best_source_rate,
        "champion_score": champion_score,
        "champion_source": champion_source,
    }


def highest_risk(levels: list[str]) -> str:
    if not levels:
        return "low"
    return max(levels, key=lambda level: RISK_SEVERITY.get(level, 0))


def profile_range_risk_flags(row: dict[str, Any], business: str, model: dict[str, Any]) -> list[tuple[str, str]]:
    profile = model.get("business_risk_profiles", {}).get(business, {})
    if not profile:
        return []
    min_count = int(model.get("risk_min_profile_count", 20))
    flags: list[tuple[str, str]] = []

    support_nodes = optional_number(profile.get("support_nodes"))
    if support_nodes is not None and support_nodes < max(min_count, 30):
        flags.append(("medium", f"business history support is low: support_nodes={support_nodes:g}"))

    for field in RESOURCE_FLOOR_FIELDS:
        value = optional_number(row.get(field))
        count = optional_number(profile.get(f"{field}_count"))
        p05 = optional_number(profile.get(f"{field}_p05"))
        p50 = optional_number(profile.get(f"{field}_p50"))
        if value is None or count is None or p05 is None or count < min_count or p05 <= 0:
            continue
        if value < p05:
            level = "high" if p50 is not None and p50 > 0 and value < 0.5 * p50 else "medium"
            flags.append((level, f"{field}={value:.4g} below business p05={p05:.4g}"))

    for field in QUALITY_CEILING_FIELDS:
        value = optional_number(row.get(field))
        count = optional_number(profile.get(f"{field}_count"))
        p90 = optional_number(profile.get(f"{field}_p90"))
        p95 = optional_number(profile.get(f"{field}_p95"))
        if value is None or count is None or p90 is None or count < min_count:
            continue
        if (p95 is None or p95 <= 0) and p90 <= 0:
            continue
        if p95 is not None and value > p95:
            flags.append(("high", f"{field}={value:.4g} above business p95={p95:.4g}"))
        elif value > p90:
            flags.append(("medium", f"{field}={value:.4g} above business p90={p90:.4g}"))
    return flags


def recommendation_confidence_flags(score: dict[str, Any], model: dict[str, Any]) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []
    if clean_cell(score.get("source")) == "global":
        flags.append(("medium", "no matched profile segment; using global fallback"))
    min_best_support = int(model.get("risk_min_best_support", 3))
    support = float(score.get("support", 0) or 0)
    if 0 < support < min_best_support:
        flags.append(("medium", f"matched segment best support is low: support={support:.3g}"))
    if not clean_cell(score.get("business_name")):
        flags.append(("medium", "business name missing from Superset mapping"))
    return flags


def combine_business_risk(
    row: dict[str, Any],
    score: dict[str, Any],
    model: dict[str, Any],
    node_risk: tuple[str, str, list[str]],
) -> tuple[str, str]:
    node_level, node_reasons, _ = node_risk
    flags: list[tuple[str, str]] = []
    if node_level in {"medium", "high", "unknown"}:
        reason = node_reasons or f"node profile risk={node_level}"
        flags.append((node_level, reason))
    flags.extend(recommendation_confidence_flags(score, model))
    flags.extend(profile_range_risk_flags(row, clean_cell(score.get("business")), model))

    level = highest_risk([risk_level for risk_level, _ in flags])
    reasons: list[str] = []
    seen: set[str] = set()
    for _, reason in flags:
        for part in str(reason).split(";"):
            cleaned = part.strip()
            if cleaned and cleaned not in seen:
                reasons.append(cleaned)
                seen.add(cleaned)
    return level, "; ".join(reasons)


def score_node(row: dict[str, Any], model: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    prepared = add_buckets(pd.DataFrame([row])).iloc[0].to_dict()
    scored = [score_one_business(prepared, business, model) for business in model["candidate_businesses"]]
    score_norm = minmax([item["expected_score"] for item in scored])
    best_norm = minmax([math.sqrt(max(item["expected_best_rate"], 0.0)) for item in scored])
    champion_norm = minmax([item["champion_score"] for item in scored])
    source_rate_norm = minmax([item["source_best_rate"] for item in scored])
    weights = {**RANK_SCORE_WEIGHTS, **model.get("rank_score_weights", {})}
    node_risk = v1.combine_risk(prepared, model)
    for item, score_part, best_part, champion_part, source_part in zip(
        scored,
        score_norm,
        best_norm,
        champion_norm,
        source_rate_norm,
    ):
        item["rank_score"] = (
            float(weights.get("source_rate", 0.60)) * source_part
            + float(weights.get("champion", 0.20)) * champion_part
            + float(weights.get("expected_best_rate", 0.15)) * best_part
            + float(weights.get("expected_score", 0.05)) * score_part
        )
        item["rank_score_components"] = {
            "source_rate_norm": source_part,
            "champion_norm": champion_part,
            "expected_best_rate_norm": best_part,
            "expected_score_norm": score_part,
        }
        risk_level, risk_reasons = combine_business_risk(prepared, item, model, node_risk)
        item["risk_level"] = risk_level
        item["risk_reasons"] = risk_reasons
        item["reason"] = (
            f"{item['source']} source_best_rate={item['source_best_rate']:.6g}; "
            f"support={item['support']}; expected_score={item['expected_score']:.6g}; "
            f"champion={item['champion_source'] or 'none'}"
        )
    return sorted(scored, key=lambda item: (-item["rank_score"], item["business"]))[:top_k]


def recommendations_for_nodes(nodes: pd.DataFrame, model: dict[str, Any], top_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node in nodes.drop_duplicates("node_id").itertuples(index=False):
        row = node._asdict()
        node_id = clean_cell(row.get("node_id"))
        scores = score_node(row, model, top_k=top_k)
        output: dict[str, Any] = {
            "node_id": node_id,
            "recommendation_mode": "v2_profile_segment_ranker",
        }
        for index, score in enumerate(scores, 1):
            output[f"v2_business_top{index}"] = score["business"]
            output[f"v2_business_name_top{index}"] = business_display_name(
                score["business"],
                score["business_name"],
            )
            output[f"v2_rank_score_top{index}"] = score["rank_score"]
            output[f"v2_expected_score_top{index}"] = score["expected_score"]
            output[f"v2_expected_best_rate_top{index}"] = score["expected_best_rate"]
            output[f"v2_source_top{index}"] = score["source"]
            output[f"v2_support_top{index}"] = score["support"]
            output[f"v2_reason_top{index}"] = score["reason"]
            output[f"v2_risk_level_top{index}"] = score["risk_level"]
            output[f"v2_risk_reasons_top{index}"] = score["risk_reasons"]
        output["v2_top3_detail"] = ";".join(
            f"{score['business']}:{score['rank_score']:.6g}:{score['source']}"
            for score in scores
        )
        rows.append(output)
    return pd.DataFrame(rows)


def default_display_columns(top_k: int) -> list[str]:
    columns = ["node_id", "recommendation_mode"]
    for index in range(1, top_k + 1):
        columns.extend([
            f"v2_business_top{index}",
            f"v2_business_name_top{index}",
            f"v2_rank_score_top{index}",
            f"v2_reason_top{index}",
            f"v2_risk_level_top{index}",
            f"v2_risk_reasons_top{index}",
        ])
    return columns


def load_nodes(path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(path, dtype={"node_id": "string"}, low_memory=False)
    nodes["node_id"] = nodes["node_id"].map(clean_cell)
    return add_buckets(nodes)


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


def parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    values: list[str] = []
    for part in text.replace("\n", ",").split(","):
        cleaned = clean_cell(part)
        if cleaned:
            values.append(cleaned)
    return values


def parse_node_id_inputs(args: argparse.Namespace) -> list[str]:
    node_ids: list[str] = []
    for node_id in args.node_id or []:
        cleaned = clean_cell(node_id)
        if cleaned:
            node_ids.append(cleaned)
    node_ids.extend(parse_csv_list(args.node_ids))
    if args.node_id_file:
        text = args.node_id_file.read_text(encoding="utf-8")
        node_ids.extend(parse_csv_list(text))
    return list(dict.fromkeys(node_ids))


def print_or_write_frame(frame: pd.DataFrame, args: argparse.Namespace, columns: list[str] | None = None) -> None:
    output = frame.copy()
    if columns:
        existing = [column for column in columns if column in output.columns]
        output = output[existing]
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(json.dumps({"output": str(args.output), "rows": int(len(output))}, ensure_ascii=False, indent=2))
    elif getattr(args, "json", False):
        print(output.to_json(orient="records", force_ascii=False, indent=2))
    else:
        print(output.to_csv(index=False))


def evaluate_model(test_pairs: pd.DataFrame, test_nodes: pd.DataFrame, model: dict[str, Any], top_k: int) -> dict[str, Any]:
    test = test_pairs[test_pairs["business"].isin(set(model["candidate_businesses"]))].copy()
    test = add_comparison_group_id(test)
    true_best = true_best_by_node(test)
    recommendations = recommendations_for_nodes(test_nodes, model, top_k).set_index("node_id")
    outcomes_by_group = {
        clean_cell(group_id): group.set_index("business")
        for group_id, group in test.groupby("_comparison_group_id", sort=False)
    }
    hit1: list[bool] = []
    hit3: list[bool] = []
    ndcg3: list[float] = []
    metric_weights: list[float] = []
    regrets: list[float] = []
    regret_weights: list[float] = []
    observed_coverage_weight = 0.0
    top1_counts: dict[str, int] = {}

    for true_row in true_best.to_dict(orient="records"):
        node_id = clean_cell(true_row.get("node_id"))
        group_id = clean_cell(true_row.get("_comparison_group_id"))
        if node_id not in recommendations.index:
            continue
        rec = recommendations.loc[node_id]
        top_businesses = [
            clean_cell(rec.get(f"v2_business_top{index}", ""))
            for index in range(1, top_k + 1)
        ]
        top_businesses = [business for business in top_businesses if business]
        if not top_businesses:
            continue
        top1_counts[top_businesses[0]] = top1_counts.get(top_businesses[0], 0) + 1
        true_business = clean_cell(true_row["business"])
        hit1.append(top_businesses[0] == true_business)
        hit3.append(true_business in set(top_businesses))
        weight = float(true_row.get("sample_weight", 1.0) or 1.0)
        metric_weights.append(weight)

        group_outcomes = outcomes_by_group[group_id]
        best_score = float(group_outcomes["combined_score"].max())
        dcg = 0.0
        ideal = 0.0
        ideal_scores = sorted(pd.to_numeric(group_outcomes["combined_score"], errors="coerce").fillna(0.0), reverse=True)
        for rank, business in enumerate(top_businesses, 1):
            gain = float(group_outcomes.loc[business, "combined_score"]) if business in group_outcomes.index else 0.0
            dcg += gain / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_scores[:top_k], 1):
            ideal += float(gain) / math.log2(rank + 1)
        ndcg3.append(dcg / ideal if ideal > 0 else 0.0)

        selected = top_businesses[0]
        if selected in group_outcomes.index:
            observed_coverage_weight += weight
            selected_score = float(group_outcomes.loc[selected, "combined_score"])
            regrets.append(max(0.0, (best_score - selected_score) / max(abs(best_score), 1e-9)))
            regret_weights.append(weight)

    n_nodes = len(hit1)
    effective_weight = float(sum(metric_weights))
    return {
        "n_nodes": n_nodes,
        "n_eval_groups": n_nodes,
        "n_eval_unique_nodes": int(true_best["node_id"].nunique()) if "node_id" in true_best else 0,
        "effective_eval_weight": effective_weight,
        "hit_rate_at_1": float(np.average(hit1, weights=metric_weights)) if hit1 else 0.0,
        "hit_rate_at_3": float(np.average(hit3, weights=metric_weights)) if hit3 else 0.0,
        "ndcg_at_3": float(np.average(ndcg3, weights=metric_weights)) if ndcg3 else 0.0,
        "top1_observed_coverage": (
            observed_coverage_weight / effective_weight if effective_weight else 0.0
        ),
        "observed_regret": (
            float(np.average(regrets, weights=regret_weights)) if regrets else None
        ),
        "top1_distribution": dict(sorted(top1_counts.items(), key=lambda item: item[1], reverse=True)[:20]),
    }


def command_build(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = prepare_pairs(args.pairs, args.business_map)
    try:
        rank_score_weights = parse_rank_score_weights(args.rank_score_weights)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    feature_fields = load_feature_fields(args.field_contract)
    business_risk_profiles = load_business_risk_profiles(None if args.no_risk_profile else args.risk_profile)
    if args.limit_nodes:
        keep = set(pairs["node_id"].drop_duplicates().head(args.limit_nodes))
        pairs = pairs[pairs["node_id"].isin(keep)].copy()
    node_ids = sorted(pairs["node_id"].dropna().astype(str).unique().tolist())
    train_nodes, test_nodes = split_node_ids(node_ids, args.train_ratio, args.seed)
    train_pairs = pairs[pairs["node_id"].isin(train_nodes)].copy()
    test_pairs = pairs[pairs["node_id"].isin(test_nodes)].copy()
    test_node_frame = test_pairs.drop_duplicates("node_id").copy()

    model = fit_v2_model(
        train_pairs,
        min_business_support=args.min_business_support,
        smoothing_alpha=args.smoothing_alpha,
        best_alpha=args.best_alpha,
        champion_min_support=args.champion_min_support,
        feature_fields=feature_fields,
        business_risk_profiles=business_risk_profiles,
        risk_min_profile_count=args.risk_min_profile_count,
        risk_min_best_support=args.risk_min_best_support,
        rank_score_weights=rank_score_weights,
    )
    recommendations = recommendations_for_nodes(
        pairs.drop_duplicates("node_id"),
        model,
        args.top_k,
    )
    metrics = evaluate_model(test_pairs, test_node_frame, model, args.top_k)

    model_path = output_dir / OUTPUT_MODEL
    rec_path = output_dir / OUTPUT_RECOMMENDATIONS
    metrics_path = output_dir / OUTPUT_METRICS
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    recommendations.to_csv(rec_path, index=False)
    payload = {
        "version": "v2",
        "model_type": model["model_type"],
        "test": metrics,
        "diagnostics": {
            "pair_rows": int(len(pairs)),
            "pair_effective_support": float(v1.sample_weights(pairs).sum()),
            "train_nodes": int(len(train_nodes)),
            "test_nodes": int(len(test_nodes)),
            "comparison_groups": int(add_comparison_group_id(pairs)["_comparison_group_id"].nunique()),
            "candidate_businesses": int(len(model["candidate_businesses"])),
            "champion_min_support": int(args.champion_min_support),
            "rank_score_weights": model["rank_score_weights"],
            "segment_levels": model["segment_levels"],
            "feature_fields": int(len(feature_fields)),
            "business_risk_profiles": int(len(business_risk_profiles)),
            "output_files": {
                "model": str(model_path),
                "recommendations": str(rec_path),
            },
        },
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def command_recommend_node(args: argparse.Namespace) -> int:
    model = json.loads(args.model.read_text(encoding="utf-8"))
    nodes = load_nodes(args.nodes)
    node_id = clean_cell(args.node_id)
    node = nodes[nodes["node_id"] == node_id]
    if node.empty:
        print(f"node_id not found: {args.node_id}", file=sys.stderr)
        return 1
    scores = score_node(node.iloc[0].to_dict(), model, args.top_k)
    display_scores = []
    for score in scores:
        item = dict(score)
        item["business_name"] = business_display_name(item["business"], item.get("business_name"))
        display_scores.append(item)
    payload = {
        "node_id": node_id,
        "recommendation_mode": "v2_profile_segment_ranker",
        "top3": display_scores,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"node_id: {node_id}")
        for index, item in enumerate(display_scores, 1):
            name = f" ({item['business_name']})" if item["business_name"] else ""
            print(
                f"{index}. {item['business']}{name} "
                f"rank_score={item['rank_score']:.6g} source={item['source']} support={item['support']}"
            )
            print(f"   reason: {item['reason']}")
    return 0


def command_recommend_batch(args: argparse.Namespace) -> int:
    model = json.loads(args.model.read_text(encoding="utf-8"))
    nodes = load_nodes(args.nodes)
    node_ids = parse_node_id_inputs(args)
    if not node_ids:
        print("no node ids provided; use --node-id, --node-ids, or --node-id-file", file=sys.stderr)
        return 1
    node_lookup = {clean_cell(row["node_id"]): row for row in nodes.to_dict(orient="records")}
    matched_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for node_id in node_ids:
        row = node_lookup.get(node_id)
        if row is None:
            missing.append(node_id)
        else:
            matched_rows.append(row)
    recommendations = recommendations_for_nodes(pd.DataFrame(matched_rows), model, args.top_k) if matched_rows else pd.DataFrame()
    if args.json:
        payload = {
            "requested": len(node_ids),
            "matched": int(len(recommendations)),
            "missing_node_ids": missing,
            "recommendations": recommendations.to_dict(orient="records"),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    if missing:
        print(f"missing_node_ids: {','.join(missing)}", file=sys.stderr)
    print_or_write_frame(recommendations, args, default_display_columns(args.top_k))
    return 0


def command_recommend_filter(args: argparse.Namespace) -> int:
    model = json.loads(args.model.read_text(encoding="utf-8"))
    nodes = load_nodes(args.nodes)
    try:
        filters = parse_where_clauses(args.where or [])
        matched = filter_nodes(nodes, filters)
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.limit:
        matched = matched.head(args.limit).copy()
    recommendations = recommendations_for_nodes(matched, model, args.top_k)
    print_or_write_frame(recommendations, args, default_display_columns(args.top_k))
    return 0


def parse_group_fields(args: argparse.Namespace, filters: dict[str, str]) -> list[str]:
    if args.group_by:
        fields = [field.strip() for field in args.group_by.split(",") if field.strip()]
    else:
        fields = [
            "city",
            "resourcetype",
            "deliverytype",
            "device_type",
            "nattype",
            "dialtype",
            "bw_bucket",
            "actualbandwidth_bucket",
        ]
    return [field for field in fields if field not in filters]


def parse_segment_specs(specs: list[str] | None) -> list[tuple[str, list[str]]]:
    if not specs:
        return DEFAULT_CONDITION_SEGMENTS
    parsed: list[tuple[str, list[str]]] = []
    for index, spec in enumerate(specs, 1):
        if ":" in spec:
            name, fields_text = spec.split(":", 1)
            name = clean_cell(name) or f"custom_{index}"
        else:
            name = f"custom_{index}"
            fields_text = spec
        fields = [field.strip() for field in fields_text.split(",") if field.strip()]
        if not fields:
            raise ValueError(f"invalid --segment, expected name:field1,field2 or field1,field2: {spec}")
        parsed.append((name, fields))
    return parsed


def load_or_score_recommendations(
    matched: pd.DataFrame,
    model: dict[str, Any],
    top_k: int,
    recommendations_path: Path | None,
    score_live: bool,
) -> pd.DataFrame:
    if recommendations_path and recommendations_path.exists() and not score_live:
        rec = pd.read_csv(recommendations_path, dtype={"node_id": "string"}, low_memory=False)
        rec["node_id"] = rec["node_id"].map(clean_cell)
        return matched[["node_id"]].merge(rec, on="node_id", how="inner", sort=False)
    return recommendations_for_nodes(matched, model, top_k)


def condition_key(row: dict[str, Any], fields: list[str]) -> str:
    return json.dumps({field: clean_cell(row.get(field)) for field in fields}, ensure_ascii=False, sort_keys=True)


def condition_text(row: dict[str, Any], fields: list[str]) -> str:
    return ";".join(f"{field}={clean_cell(row.get(field))}" for field in fields)


def business_display_name(business: Any, business_name: Any) -> str:
    name = clean_cell(business_name)
    if name and not is_generated_business_id_label(name):
        return name
    business_id = clean_cell(business)
    return UNKNOWN_BUSINESS_NAME if business_id else ""


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def confidence_for_condition(
    node_count: int,
    top1_share: float,
    vote_share: float,
    high_risk_rate: float,
) -> tuple[str, str]:
    if node_count >= 50 and top1_share >= 0.60 and high_risk_rate <= 0.20:
        return "high", "large segment, stable top1, low high-risk rate"
    if node_count >= 10 and (top1_share >= 0.40 or vote_share >= 0.55) and high_risk_rate <= 0.35:
        return "medium", "usable segment, review before operation"
    return "low", "small or divergent segment; use as exploration signal"


def build_frontend_condition_report(
    nodes: pd.DataFrame,
    recommendations: pd.DataFrame,
    segment_specs: list[tuple[str, list[str]]],
    top_k: int,
    min_nodes: int,
) -> pd.DataFrame:
    joined = nodes.merge(recommendations, on="node_id", how="inner", sort=False, suffixes=("", "_rec"))
    if joined.empty:
        return pd.DataFrame()

    output_rows: list[dict[str, Any]] = []
    for segment_name, fields in segment_specs:
        available_fields = [field for field in fields if field in joined.columns]
        if len(available_fields) != len(fields):
            missing = sorted(set(fields) - set(available_fields))
            raise KeyError(f"segment {segment_name} missing node fields: {', '.join(missing)}")

        work = joined.copy()
        for field in available_fields:
            work[field] = work[field].map(v1.safe_category)
        group_counts = work.groupby(available_fields, dropna=False)["node_id"].nunique().to_dict()
        long_rows: list[dict[str, Any]] = []
        for rank in range(1, top_k + 1):
            business_col = f"v2_business_top{rank}"
            if business_col not in work.columns:
                continue
            name_col = f"v2_business_name_top{rank}"
            score_col = f"v2_rank_score_top{rank}"
            expected_col = f"v2_expected_score_top{rank}"
            risk_col = f"v2_risk_level_top{rank}"
            part_columns = ["node_id"] + available_fields + [business_col]
            for column in [name_col, score_col, expected_col, risk_col]:
                if column in work.columns:
                    part_columns.append(column)
            part = work[part_columns].copy()
            part = part.rename(columns={
                business_col: "business",
                name_col: "business_name",
                score_col: "rank_score",
                expected_col: "expected_score",
                risk_col: "risk_level",
            })
            part["rank_position"] = rank
            part["vote_weight"] = 1.0 / rank
            long_rows.extend(part.to_dict(orient="records"))
        if not long_rows:
            continue

        long = pd.DataFrame(long_rows)
        long["business"] = long["business"].map(clean_cell)
        long = long[long["business"].ne("")].copy()
        if long.empty:
            continue
        if "business_name" not in long.columns:
            long["business_name"] = ""
        if "rank_score" not in long.columns:
            long["rank_score"] = 0.0
        if "expected_score" not in long.columns:
            long["expected_score"] = 0.0
        if "risk_level" not in long.columns:
            long["risk_level"] = ""
        long["business_name"] = long["business_name"].fillna("").map(clean_cell)
        long["rank_score"] = pd.to_numeric(long["rank_score"], errors="coerce")
        long["expected_score"] = pd.to_numeric(long["expected_score"], errors="coerce")
        long["risk_level"] = long["risk_level"].fillna("").map(clean_cell)
        long["is_top1"] = long["rank_position"].eq(1)
        long["is_high_risk"] = long["risk_level"].eq("high")
        long["is_medium_risk"] = long["risk_level"].eq("medium")
        long["is_low_risk"] = long["risk_level"].eq("low")

        grouped_business = (
            long.groupby(available_fields + ["business"], dropna=False)
            .agg(
                weighted_votes=("vote_weight", "sum"),
                top1_node_count=("is_top1", "sum"),
                top3_node_count=("node_id", "nunique"),
                business_name=("business_name", latest_nonempty),
                avg_rank_score=("rank_score", "mean"),
                avg_expected_score=("expected_score", "mean"),
                high_risk_count=("is_high_risk", "sum"),
                medium_risk_count=("is_medium_risk", "sum"),
                low_risk_count=("is_low_risk", "sum"),
            )
            .reset_index()
        )
        grouped_business["node_count"] = grouped_business.apply(
            lambda row: int(group_counts.get(tuple(row[field] for field in available_fields), 0)),
            axis=1,
        )
        grouped_business = grouped_business[grouped_business["node_count"] >= min_nodes].copy()
        if grouped_business.empty:
            continue
        grouped_business["vote_share"] = grouped_business["weighted_votes"] / grouped_business["node_count"].clip(lower=1)
        grouped_business["top1_share"] = grouped_business["top1_node_count"] / grouped_business["node_count"].clip(lower=1)
        grouped_business["top3_share"] = grouped_business["top3_node_count"] / grouped_business["node_count"].clip(lower=1)
        grouped_business["high_risk_rate"] = grouped_business["high_risk_count"] / grouped_business["top3_node_count"].clip(lower=1)
        grouped_business["medium_risk_rate"] = grouped_business["medium_risk_count"] / grouped_business["top3_node_count"].clip(lower=1)
        grouped_business = grouped_business.sort_values(
            available_fields + ["weighted_votes", "top1_node_count", "avg_rank_score", "business"],
            ascending=[True] * len(available_fields) + [False, False, False, True],
        )

        for group_values, group in grouped_business.groupby(available_fields, dropna=False, sort=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group = group.head(top_k).copy()
            base = {field: value for field, value in zip(available_fields, group_values)}
            top = group.iloc[0]
            confidence, confidence_reason = confidence_for_condition(
                int(top["node_count"]),
                float(top["top1_share"]),
                float(top["vote_share"]),
                float(top["high_risk_rate"]),
            )
            row: dict[str, Any] = {
                "segment_level": segment_name,
                "segment_fields": ",".join(available_fields),
                "condition_key": condition_key(base, available_fields),
                "condition_text": condition_text(base, available_fields),
                **base,
                "node_count": int(top["node_count"]),
                "confidence_level": confidence,
                "confidence_reason": confidence_reason,
            }
            detail_parts: list[str] = []
            for rank, candidate in enumerate(group.to_dict(orient="records"), 1):
                row[f"business_top{rank}"] = clean_cell(candidate["business"])
                row[f"business_name_top{rank}"] = business_display_name(
                    candidate["business"],
                    candidate.get("business_name"),
                )
                row[f"vote_share_top{rank}"] = float(candidate["vote_share"])
                row[f"top1_share_top{rank}"] = float(candidate["top1_share"])
                row[f"top3_share_top{rank}"] = float(candidate["top3_share"])
                row[f"avg_rank_score_top{rank}"] = float(candidate["avg_rank_score"])
                row[f"avg_expected_score_top{rank}"] = float(candidate["avg_expected_score"])
                row[f"high_risk_rate_top{rank}"] = float(candidate["high_risk_rate"])
                row[f"medium_risk_rate_top{rank}"] = float(candidate["medium_risk_rate"])
                detail_parts.append(
                    f"{row[f'business_top{rank}']}:{row[f'vote_share_top{rank}']:.6g}:{row[f'top1_share_top{rank}']:.6g}"
                )
            row["top_business_detail"] = ";".join(detail_parts)
            output_rows.append(row)

    report = pd.DataFrame(output_rows)
    if report.empty:
        return report
    return report.sort_values(
        ["segment_level", "node_count", "confidence_level", "vote_share_top1"],
        ascending=[True, False, True, False],
    )


def command_recommend_segments(args: argparse.Namespace) -> int:
    model = json.loads(args.model.read_text(encoding="utf-8"))
    nodes = load_nodes(args.nodes)
    try:
        filters = parse_where_clauses(args.where or [])
        matched = filter_nodes(nodes, filters)
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if matched.empty:
        print("no matched nodes", file=sys.stderr)
        return 1

    group_fields = parse_group_fields(args, filters)
    missing_group_fields = [field for field in group_fields if field not in matched.columns]
    if missing_group_fields:
        print(f"unknown group fields: {', '.join(missing_group_fields)}", file=sys.stderr)
        return 1

    rec = load_or_score_recommendations(
        matched,
        model,
        args.top_k,
        args.recommendations,
        args.score_live,
    )
    joined = matched.merge(rec, on="node_id", how="inner", sort=False, suffixes=("", "_rec"))
    if joined.empty:
        print("no recommendations for matched nodes", file=sys.stderr)
        return 1
    for field in group_fields:
        joined[field] = joined[field].map(v1.safe_category)
    top1 = "v2_business_top1"
    name1 = "v2_business_name_top1"
    score1 = "v2_rank_score_top1"
    expected1 = "v2_expected_score_top1"
    risk1 = "v2_risk_level_top1"
    joined[top1] = joined[top1].map(clean_cell)
    joined[name1] = joined.get(name1, "").fillna("").map(clean_cell)
    joined[score1] = pd.to_numeric(joined.get(score1), errors="coerce")
    joined[expected1] = pd.to_numeric(joined.get(expected1), errors="coerce")
    joined[risk1] = joined.get(risk1, "").fillna("").map(clean_cell)

    group_node_counts = joined.groupby(group_fields, dropna=False)["node_id"].nunique().rename("node_count")
    business_counts = (
        joined.groupby(group_fields + [top1], dropna=False)
        .agg(
            business_node_count=("node_id", "nunique"),
            business_name=(name1, latest_nonempty),
            avg_rank_score=(score1, "mean"),
            avg_expected_score=(expected1, "mean"),
            high_risk_nodes=(risk1, lambda values: int((values == "high").sum())),
            medium_risk_nodes=(risk1, lambda values: int((values == "medium").sum())),
            low_risk_nodes=(risk1, lambda values: int((values == "low").sum())),
            sample_node_ids=("node_id", lambda values: ",".join(values.astype(str).head(5))),
        )
        .reset_index()
        .merge(group_node_counts.reset_index(), on=group_fields, how="left")
    )
    business_counts["business_share"] = (
        business_counts["business_node_count"] / business_counts["node_count"].clip(lower=1)
    )
    business_counts = business_counts[business_counts["node_count"] >= args.min_nodes].copy()
    business_counts = business_counts.sort_values(
        ["node_count", "business_share", "avg_rank_score", top1],
        ascending=[False, False, False, True],
    )
    winners = business_counts.drop_duplicates(group_fields).rename(columns={
        top1: "recommended_business",
        "business_name": "recommended_business_name",
    })
    columns = group_fields + [
        "node_count",
        "recommended_business",
        "recommended_business_name",
        "business_node_count",
        "business_share",
        "avg_rank_score",
        "avg_expected_score",
        "high_risk_nodes",
        "medium_risk_nodes",
        "low_risk_nodes",
        "sample_node_ids",
    ]
    if args.limit:
        winners = winners.head(args.limit).copy()
    print_or_write_frame(winners, args, columns)
    return 0


def command_export_condition_report(args: argparse.Namespace) -> int:
    model = json.loads(args.model.read_text(encoding="utf-8"))
    nodes = load_nodes(args.nodes)
    try:
        filters = parse_where_clauses(args.where or [])
        nodes = filter_nodes(nodes, filters) if filters else nodes
        segment_specs = parse_segment_specs(args.segment)
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if nodes.empty:
        print("no matched nodes", file=sys.stderr)
        return 1

    recommendations = load_or_score_recommendations(
        nodes,
        model,
        args.top_k,
        args.recommendations,
        args.score_live,
    )
    try:
        report = build_frontend_condition_report(
            nodes,
            recommendations,
            segment_specs,
            args.top_k,
            args.min_nodes,
        )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if report.empty:
        print("empty condition report", file=sys.stderr)
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_data_js.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_csv, index=False)
    payload = json_safe({
        "version": "v2_frontend_condition_business_report",
        "rows": int(len(report)),
        "filters": filters,
        "min_nodes": int(args.min_nodes),
        "top_k": int(args.top_k),
        "segment_levels": [
            {"name": name, "fields": fields}
            for name, fields in segment_specs
        ],
        "columns": report.columns.tolist(),
        "records": report.to_dict(orient="records"),
    })
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_data_js.write_text(
        "window.CONDITION_REPORT_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "csv": str(args.output_csv),
        "json": str(args.output_json),
        "data_js": str(args.output_data_js),
        "rows": int(len(report)),
        "segment_level_counts": report["segment_level"].value_counts().to_dict(),
    }, ensure_ascii=False, indent=2))
    return 0


def command_check(args: argparse.Namespace) -> int:
    required = [
        args.output_dir / OUTPUT_MODEL,
        args.output_dir / OUTPUT_RECOMMENDATIONS,
        args.output_dir / OUTPUT_METRICS,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 1
    rec = pd.read_csv(args.output_dir / OUTPUT_RECOMMENDATIONS, nrows=100)
    with (args.output_dir / OUTPUT_METRICS).open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    required_columns = {
        "node_id",
        "v2_business_top1",
        "v2_business_top2",
        "v2_business_top3",
        "v2_reason_top1",
        "v2_risk_level_top1",
        "v2_risk_reasons_top1",
        "v2_top3_detail",
    }
    errors = []
    if not required_columns.issubset(rec.columns):
        errors.append(f"recommendations missing columns: {sorted(required_columns - set(rec.columns))}")
    if metrics.get("test", {}).get("n_nodes", 0) <= 0:
        errors.append("metrics test.n_nodes is zero")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("V2 artifacts check passed")
    print(json.dumps({"test": metrics.get("test"), "files": [str(path) for path in required]}, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and use V2 new-node ranking recommendations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Train V2 ranking model and export recommendations.")
    build.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    build.add_argument("--business-map", type=Path)
    build.add_argument("--field-contract", type=Path, default=DEFAULT_FIELD_CONTRACT)
    build.add_argument("--risk-profile", type=Path, default=DEFAULT_RISK_PROFILE)
    build.add_argument("--no-risk-profile", action="store_true")
    build.add_argument("--output-dir", type=Path, default=HERE)
    build.add_argument("--train-ratio", type=float, default=0.8)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--min-business-support", type=int, default=30)
    build.add_argument("--smoothing-alpha", type=float, default=30.0)
    build.add_argument("--best-alpha", type=float, default=50.0)
    build.add_argument("--champion-min-support", type=int, default=3)
    build.add_argument("--risk-min-profile-count", type=int, default=20)
    build.add_argument("--risk-min-best-support", type=int, default=3)
    build.add_argument(
        "--rank-score-weights",
        help=(
            "Comma separated weights, e.g. "
            "source_rate=0.2,champion=0.1,expected_best_rate=0.2,expected_score=0.5"
        ),
    )
    build.add_argument("--top-k", type=int, default=3)
    build.add_argument("--limit-nodes", type=int)
    build.set_defaults(func=command_build)

    recommend = subparsers.add_parser("recommend-node", help="Recommend Top3 businesses for one node using V2 model.")
    recommend.add_argument("node_id")
    recommend.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    recommend.add_argument("--model", type=Path, default=HERE / OUTPUT_MODEL)
    recommend.add_argument("--top-k", type=int, default=3)
    recommend.add_argument("--json", action="store_true")
    recommend.set_defaults(func=command_recommend_node)

    batch = subparsers.add_parser("recommend-batch", help="Recommend Top3 businesses for multiple node IDs.")
    batch.add_argument("--node-id", action="append", help="One node ID; can be repeated.")
    batch.add_argument("--node-ids", help="Comma or newline separated node IDs.")
    batch.add_argument("--node-id-file", type=Path, help="Text file with comma/newline separated node IDs.")
    batch.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    batch.add_argument("--model", type=Path, default=HERE / OUTPUT_MODEL)
    batch.add_argument("--top-k", type=int, default=3)
    batch.add_argument("--output", type=Path)
    batch.add_argument("--json", action="store_true")
    batch.set_defaults(func=command_recommend_batch)

    recommend_filter = subparsers.add_parser(
        "recommend-filter",
        help="Recommend Top3 businesses for nodes matched by one or more field=value filters.",
    )
    recommend_filter.add_argument("--where", action="append", required=True, help="Exact node filter, e.g. province=江苏")
    recommend_filter.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    recommend_filter.add_argument("--model", type=Path, default=HERE / OUTPUT_MODEL)
    recommend_filter.add_argument("--top-k", type=int, default=3)
    recommend_filter.add_argument("--limit", type=int, default=20)
    recommend_filter.add_argument("--output", type=Path)
    recommend_filter.add_argument("--json", action="store_true")
    recommend_filter.set_defaults(func=command_recommend_filter)

    segments = subparsers.add_parser(
        "recommend-segments",
        help="Summarize best businesses by profile segments under one or more field=value filters.",
    )
    segments.add_argument("--where", action="append", required=True, help="Exact node filter, e.g. province=江苏")
    segments.add_argument(
        "--group-by",
        help="Comma separated segment fields. Default: city,resourcetype,deliverytype,device_type,nattype,dialtype,bw_bucket,actualbandwidth_bucket",
    )
    segments.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    segments.add_argument("--model", type=Path, default=HERE / OUTPUT_MODEL)
    segments.add_argument("--recommendations", type=Path, default=HERE / OUTPUT_RECOMMENDATIONS)
    segments.add_argument("--score-live", action="store_true", help="Ignore precomputed recommendation CSV and score matched nodes live.")
    segments.add_argument("--top-k", type=int, default=3)
    segments.add_argument("--min-nodes", type=int, default=5)
    segments.add_argument("--limit", type=int, default=20)
    segments.add_argument("--output", type=Path)
    segments.add_argument("--json", action="store_true")
    segments.set_defaults(func=command_recommend_segments)

    frontend = subparsers.add_parser(
        "export-condition-report",
        help="Export frontend-ready condition-combination business recommendation report.",
    )
    frontend.add_argument("--where", action="append", help="Optional exact node filter, e.g. province=江苏")
    frontend.add_argument(
        "--segment",
        action="append",
        help="Custom segment spec: name:field1,field2 or field1,field2. Can be repeated.",
    )
    frontend.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    frontend.add_argument("--model", type=Path, default=HERE / OUTPUT_MODEL)
    frontend.add_argument("--recommendations", type=Path, default=HERE / OUTPUT_RECOMMENDATIONS)
    frontend.add_argument("--score-live", action="store_true", help="Ignore precomputed recommendation CSV and score nodes live.")
    frontend.add_argument("--top-k", type=int, default=3)
    frontend.add_argument("--min-nodes", type=int, default=10)
    frontend.add_argument("--output-csv", type=Path, default=HERE / OUTPUT_FRONTEND_REPORT)
    frontend.add_argument("--output-json", type=Path, default=HERE / OUTPUT_FRONTEND_REPORT_JSON)
    frontend.add_argument("--output-data-js", type=Path, default=HERE / OUTPUT_FRONTEND_REPORT_DATA_JS)
    frontend.set_defaults(func=command_export_condition_report)

    check = subparsers.add_parser("check", help="Check generated V2 artifacts.")
    check.add_argument("--output-dir", type=Path, default=HERE)
    check.set_defaults(func=command_check)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
