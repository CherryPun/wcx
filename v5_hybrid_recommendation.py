#!/usr/bin/env python3
"""V5 fair temporal benchmark and hybrid node-business recommender.

V5 keeps the audited V4 daily outcomes and ten-field profile whitelist. It
compares the V4 hierarchical baseline with profile-only gradient boosting,
optionally using stabilized inverse-propensity weights. Existing nodes can
receive a separately validated, business-agnostic node residual correction.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v1_recommendation_pipeline as v1
import build_v3_daily_business_training as v3
import v4_outcome_recommendation as v4

try:
    from scipy import sparse
    import xgboost as xgb
except ImportError:  # pragma: no cover - exercised by the CLI error path
    sparse = None
    xgb = None


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "recent_month_large_mainstream_v5_hybrid"

OUTPUT_MODEL = "v5_hybrid_model.json"
OUTPUT_BASELINE = "v5_baseline_model.json"
OUTPUT_MINER_BOOSTER = "v5_miner_booster.json"
OUTPUT_PLATFORM_BOOSTER = "v5_platform_booster.json"
OUTPUT_RECOMMENDATIONS = "v5_node_recommendations.csv"
OUTPUT_MISMATCH_WATCHLIST = "v5_current_not_top1_watchlist.csv"
OUTPUT_SWITCH_CANDIDATES = "v5_review_switch_candidates.csv"
OUTPUT_VALIDATION = "v5_temporal_validation_predictions.csv"
OUTPUT_BUSINESS_METRICS = "v5_business_training_metrics.csv"
OUTPUT_COMPARISON = "v5_model_comparison.csv"
OUTPUT_CONCENTRATION = "v5_recommendation_concentration.csv"
OUTPUT_SUMMARY = "v5_training_summary.json"
OUTPUT_REPORT_DATA = "v5_frontend_report_data.js"
OUTPUT_REPORT_HTML = "v5_business_recommendation_report.html"

TARGETS = [v4.TARGET_MINER, v4.TARGET_PLATFORM]
ALLOWED_SCHEDULE_TYPES = {"本网本省", "本网出省", "异网本省", "异网出省"}
FEATURE_FIELDS = [
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
]
ENCODE_FIELDS = ["business", *FEATURE_FIELDS]


def require_ml_dependencies() -> None:
    if sparse is None or xgb is None:
        raise RuntimeError(
            "V5 requires scipy and xgboost; install requirements-v5.txt in a virtual environment"
        )


def clean_feature(value: Any) -> str:
    return v4.clean(value)


def filter_supported_schedule_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    valid = frame["network_schedule_type"].isin(ALLOWED_SCHEDULE_TYPES)
    return frame[valid].copy(), int((~valid).sum())


def filter_training_window(
    frame: pd.DataFrame,
    start_day: str | None = None,
    end_day: str | None = None,
) -> pd.DataFrame:
    """Filter daily outcomes and recompute run weights inside the selected window."""
    output = frame.copy()
    sample_days = pd.to_datetime(output["sample_day"], errors="coerce")
    valid = sample_days.notna()
    if start_day:
        valid &= sample_days.ge(pd.Timestamp(start_day))
    if end_day:
        valid &= sample_days.le(pd.Timestamp(end_day))
    output = output.loc[valid].copy()
    if output.empty:
        raise RuntimeError("the selected V5 training window contains no daily outcomes")
    result = v3.add_consecutive_sample_weights(output)
    # 时效权重（新）：近端加权；半衰期天数由 env V5_RECENCY_HALF_LIFE 控制，默认 0=关闭
    half_life = float(os.getenv("V5_RECENCY_HALF_LIFE", "0") or 0)
    if half_life > 0:
        day = pd.to_datetime(result["sample_day"], errors="coerce")
        age = (day.max() - day).dt.days.clip(lower=0)
        factor = 0.5 ** (age / half_life)
        factor = factor / factor.mean()
        cap = float(os.getenv("V5_RECENCY_WEIGHT_CAP", "0") or 0)
        if cap > 0:
            factor = factor.clip(upper=cap)
            factor = factor / factor.mean()
        result = result.copy()
        result["sample_weight"] = pd.to_numeric(result["sample_weight"], errors="coerce").fillna(1.0) * factor
    return result


def fit_encoder(frame: pd.DataFrame) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    offset = 0
    for field in ENCODE_FIELDS:
        values = sorted(frame[field].fillna(v4.UNKNOWN).map(clean_feature).unique().tolist())
        if v4.UNKNOWN not in values:
            values.append(v4.UNKNOWN)
        fields[field] = {"offset": offset, "values": values}
        offset += len(values)
    return {"fields": fields, "column_count": offset}


def encode_frame(frame: pd.DataFrame, encoder: dict[str, Any]):
    require_ml_dependencies()
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    for field, spec in encoder["fields"].items():
        lookup = {value: index for index, value in enumerate(spec["values"])}
        values = frame[field].fillna(v4.UNKNOWN).map(clean_feature)
        local_columns = values.map(lookup).fillna(lookup[v4.UNKNOWN]).to_numpy(dtype=np.int32)
        row_parts.append(np.arange(len(frame), dtype=np.int32))
        col_parts.append(local_columns + int(spec["offset"]))
    rows = np.concatenate(row_parts) if row_parts else np.array([], dtype=np.int32)
    columns = np.concatenate(col_parts) if col_parts else np.array([], dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(len(frame), int(encoder["column_count"])),
        dtype=np.float32,
    )


def clip_target(
    frame: pd.DataFrame,
    target: str,
    bounds: dict[str, dict[str, list[float]]],
) -> np.ndarray:
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    lows = frame["business"].map({key: value[target][0] for key, value in bounds.items()})
    highs = frame["business"].map({key: value[target][1] for key, value in bounds.items()})
    return np.clip(values, lows.to_numpy(dtype=float), highs.to_numpy(dtype=float))


def baseline_predictions(frame: pd.DataFrame, model: dict[str, Any]) -> dict[str, np.ndarray]:
    miner: list[float] = []
    platform: list[float] = []
    for row in frame.to_dict(orient="records"):
        item = v4.predict_business(row, str(row["business"]), model)
        miner.append(float(item[v4.TARGET_MINER]))
        platform.append(float(item[v4.TARGET_PLATFORM]))
    return {
        v4.TARGET_MINER: np.asarray(miner, dtype=float),
        v4.TARGET_PLATFORM: np.asarray(platform, dtype=float),
    }


def propensity_weights(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    signature_fields = sorted({field for _, fields in v4.PROPENSITY_LEVELS for field in fields})
    cache: dict[tuple[str, ...], float] = {}
    factors: list[float] = []
    global_rates = model["propensity"]["global_rates"]
    for row in frame.to_dict(orient="records"):
        business = str(row["business"])
        signature = (business, *(clean_feature(row.get(field)) for field in signature_fields))
        probability = cache.get(signature)
        if probability is None:
            probability = float(v4.propensity_for(row, business, model)[0])
            cache[signature] = probability
        prior = float(global_rates.get(business, 0.0))
        factors.append(float(np.clip(prior / max(probability, 0.02), 0.25, 4.0)))
    base = v1.sample_weights(frame).to_numpy(dtype=float)
    weighted = base * np.asarray(factors, dtype=float)
    if weighted.sum() > 0:
        weighted *= base.sum() / weighted.sum()
    return weighted


def fit_booster(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    encoder: dict[str, Any],
    target: str,
    bounds: dict[str, dict[str, list[float]]],
    train_weights: np.ndarray,
    rounds: int,
    early_stopping_rounds: int,
    seed: int,
) -> Any:
    require_ml_dependencies()
    train_matrix = encode_frame(train, encoder)
    calibration_matrix = encode_frame(calibration, encoder)
    dtrain = xgb.DMatrix(
        train_matrix,
        label=clip_target(train, target, bounds),
        weight=train_weights,
    )
    dcalibration = xgb.DMatrix(
        calibration_matrix,
        label=clip_target(calibration, target, bounds),
        weight=v1.sample_weights(calibration).to_numpy(dtype=float),
    )
    parameters = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.045,
        "max_depth": 6,
        "min_child_weight": 4.0,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "lambda": 8.0,
        "alpha": 0.2,
        "tree_method": "hist",
        "seed": seed,
        "nthread": 6,
    }
    return xgb.train(
        parameters,
        dtrain,
        num_boost_round=rounds,
        evals=[(dcalibration, "calibration")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )


def fit_final_booster(
    frame: pd.DataFrame,
    encoder: dict[str, Any],
    target: str,
    bounds: dict[str, dict[str, list[float]]],
    weights: np.ndarray,
    rounds: int,
    seed: int,
) -> Any:
    require_ml_dependencies()
    matrix = encode_frame(frame, encoder)
    dtrain = xgb.DMatrix(
        matrix,
        label=clip_target(frame, target, bounds),
        weight=weights,
    )
    parameters = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.045,
        "max_depth": 6,
        "min_child_weight": 4.0,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "lambda": 8.0,
        "alpha": 0.2,
        "tree_method": "hist",
        "seed": seed,
        "nthread": 6,
    }
    return xgb.train(parameters, dtrain, num_boost_round=max(int(rounds), 1), verbose_eval=False)


def booster_predictions(
    frame: pd.DataFrame,
    booster: Any,
    encoder: dict[str, Any],
    target: str,
    bounds: dict[str, dict[str, list[float]]],
) -> np.ndarray:
    matrix = xgb.DMatrix(encode_frame(frame, encoder))
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None:
        predictions = booster.predict(matrix)
    else:
        predictions = booster.predict(matrix, iteration_range=(0, int(best_iteration) + 1))
    output = np.asarray(predictions, dtype=float)
    lows = frame["business"].map({key: value[target][0] for key, value in bounds.items()})
    highs = frame["business"].map({key: value[target][1] for key, value in bounds.items()})
    return np.clip(output, lows.to_numpy(dtype=float), highs.to_numpy(dtype=float))


def metrics_row(
    split: str,
    cohort: str,
    model_name: str,
    target: str,
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, Any]:
    metrics = v4.weighted_metrics(frame[target], pd.Series(predictions), v1.sample_weights(frame))
    return {"split": split, "cohort": cohort, "model": model_name, "target": target, **metrics}


def fit_node_effects(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, dict[str, float]]:
    work = frame[["node_id", "sample_weight", *TARGETS]].copy()
    output: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        work["_weighted_residual"] = (
            work[target].to_numpy(dtype=float) - predictions[target]
        ) * work["sample_weight"].to_numpy(dtype=float)
        grouped = work.groupby("node_id", sort=False).agg(
            residual_sum=("_weighted_residual", "sum"),
            effective_support=("sample_weight", "sum"),
        )
        effects = grouped["residual_sum"] / (grouped["effective_support"] + alpha)
        output[target] = {str(key): float(value) for key, value in effects.items()}
    return output


def apply_node_effect(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
    effects: dict[str, dict[str, float]],
    bounds: dict[str, dict[str, list[float]]],
) -> np.ndarray:
    correction = frame["node_id"].map(effects.get(target, {})).fillna(0.0).to_numpy(dtype=float)
    lows = frame["business"].map({key: value[target][0] for key, value in bounds.items()})
    highs = frame["business"].map({key: value[target][1] for key, value in bounds.items()})
    return np.clip(
        predictions + correction,
        lows.to_numpy(dtype=float),
        highs.to_numpy(dtype=float),
    )


def weighted_quantile(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray, q: float) -> float:
    return float(v1.weighted_quantile(pd.Series(values), pd.Series(weights), q))


def fit_intervals(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    baseline_model: dict[str, Any],
    minimum_support: float = 8.0,
) -> dict[str, dict[str, Any]]:
    work = frame[["business", "sample_weight"]].copy()
    scales: list[float] = []
    for row in frame.to_dict(orient="records"):
        item = v4.predict_business(row, str(row["business"]), baseline_model)
        scales.append(math.sqrt(1.0 + 5.0 / max(float(item["local_effective_support"]) + 5.0, 5.0)))
    scale = np.asarray(scales, dtype=float)
    for target in TARGETS:
        prefix = "miner" if target == v4.TARGET_MINER else "platform"
        work[f"{prefix}_scaled_residual"] = (
            np.abs(frame[target].to_numpy(dtype=float) - predictions[target]) / scale
        )
    weights = work["sample_weight"]
    global_values = {}
    for prefix in ["miner", "platform"]:
        residual = work[f"{prefix}_scaled_residual"]
        global_values[f"{prefix}_radius80"] = weighted_quantile(residual, weights, 0.80)
        global_values[f"{prefix}_radius90"] = weighted_quantile(residual, weights, 0.90)
    output: dict[str, dict[str, Any]] = {}
    for business in baseline_model["candidate_businesses"]:
        group = work[work["business"].eq(business)]
        effective_support = float(group["sample_weight"].sum())
        item: dict[str, Any] = {
            **global_values,
            "source": "global_calibration",
            "effective_support": effective_support,
        }
        if effective_support >= minimum_support:
            for prefix in ["miner", "platform"]:
                residual = group[f"{prefix}_scaled_residual"]
                item[f"{prefix}_radius80"] = weighted_quantile(
                    residual, group["sample_weight"], 0.80
                )
                item[f"{prefix}_radius90"] = weighted_quantile(
                    residual, group["sample_weight"], 0.90
                )
            item["source"] = "business_calibration"
        output[str(business)] = item
    return output


@dataclass
class RuntimeModel:
    metadata: dict[str, Any]
    baseline: dict[str, Any]
    boosters: dict[str, Any]


def raw_runtime_predictions(frame: pd.DataFrame, runtime: RuntimeModel) -> dict[str, np.ndarray]:
    metadata = runtime.metadata
    selected = metadata["selected_sources"]
    needed_baseline = any(source == "hierarchical_baseline" for source in selected.values())
    baseline = baseline_predictions(frame, runtime.baseline) if needed_baseline else {}
    output: dict[str, np.ndarray] = {}
    for target in TARGETS:
        source = selected[target]
        if source == "hierarchical_baseline":
            prediction = baseline[target]
        else:
            prediction = booster_predictions(
                frame,
                runtime.boosters[target],
                metadata["encoder"],
                target,
                metadata["target_bounds"],
            )
        if metadata["warm_start_enabled"].get(target, False):
            prediction = apply_node_effect(
                frame,
                prediction,
                target,
                metadata["node_effects"],
                metadata["target_bounds"],
            )
        output[target] = prediction
    return output


def confidence_for(item: dict[str, Any], validation: dict[str, Any]) -> str:
    miner_bounds = item["target_bounds"][v4.TARGET_MINER]
    platform_bounds = item["target_bounds"][v4.TARGET_PLATFORM]
    miner_span = max(miner_bounds[1] - miner_bounds[0], 1e-9)
    platform_span = max(platform_bounds[1] - platform_bounds[0], 1e-9)
    uncertainty_ratio = 0.5 * (
        (item["miner_high90"] - item["miner_low90"]) / 2 / miner_span
        + (item["platform_high90"] - item["platform_low90"]) / 2 / platform_span
    )
    item["uncertainty_ratio"] = uncertainty_ratio
    support = float(validation.get("test_effective_support", 0.0))
    coverage = float(validation.get("joint_average_coverage90", 0.0))
    if (
        item["local_effective_support"] >= 30
        and item["global_effective_support"] >= 100
        and item["propensity"] >= 0.03
        and support >= 20
        and 0.78 <= coverage <= 0.99
        and uncertainty_ratio <= 0.75
        and not item["ood_reasons"]
    ):
        return "high"
    if (
        item["local_effective_support"] >= 10
        and item["global_effective_support"] >= 30
        and item["propensity"] >= 0.01
        and support >= 5
        and uncertainty_ratio <= 1.5
        and len(item["ood_reasons"]) <= 1
    ):
        return "medium"
    return "low"


def score_profile_rows(frame: pd.DataFrame, runtime: RuntimeModel) -> list[list[dict[str, Any]]]:
    candidates = runtime.metadata["candidate_businesses"]
    repeated = frame.loc[frame.index.repeat(len(candidates))].copy().reset_index(drop=True)
    repeated["business"] = np.tile(candidates, len(frame))
    predictions = raw_runtime_predictions(repeated, runtime)
    results: list[list[dict[str, Any]]] = []
    for position in range(len(frame)):
        start = position * len(candidates)
        row_items: list[dict[str, Any]] = []
        source_row = repeated.iloc[start].to_dict()
        row_ood = v4.ood_reasons(source_row, runtime.baseline)
        for offset, business in enumerate(candidates):
            query_position = start + offset
            query = repeated.iloc[query_position].to_dict()
            base = v4.predict_business(query, business, runtime.baseline, row_ood_reasons=row_ood)
            calibration = runtime.metadata["calibration"].get(business, {})
            support_scale = math.sqrt(
                1.0 + 5.0 / max(float(base["local_effective_support"]) + 5.0, 5.0)
            )
            miner = float(predictions[v4.TARGET_MINER][query_position])
            platform = float(predictions[v4.TARGET_PLATFORM][query_position])
            miner_radius80 = float(calibration.get("miner_radius80", 0.0)) * support_scale
            miner_radius90 = float(calibration.get("miner_radius90", 0.0)) * support_scale
            platform_radius80 = float(calibration.get("platform_radius80", 0.0)) * support_scale
            platform_radius90 = float(calibration.get("platform_radius90", 0.0)) * support_scale
            item = {
                **base,
                v4.TARGET_MINER: miner,
                v4.TARGET_PLATFORM: platform,
                "miner_low80": miner - miner_radius80,
                "miner_high80": miner + miner_radius80,
                "miner_low90": miner - miner_radius90,
                "miner_high90": miner + miner_radius90,
                "platform_low80": platform - platform_radius80,
                "platform_high80": platform + platform_radius80,
                "platform_low90": platform - platform_radius90,
                "platform_high90": platform + platform_radius90,
                "target_bounds": runtime.metadata["target_bounds"][business],
            }
            validation = runtime.metadata.get("validation_by_business", {}).get(business, {})
            item["confidence"] = confidence_for(item, validation)
            row_items.append(item)

        miner_values = [item[v4.TARGET_MINER] for item in row_items]
        platform_values = [item[v4.TARGET_PLATFORM] for item in row_items]
        miner_low = min(miner_values)
        miner_span = max(max(miner_values) - miner_low, 1e-12)
        platform_low = min(platform_values)
        platform_span = max(max(platform_values) - platform_low, 1e-12)
        for item in row_items:
            item["miner_score"] = (item[v4.TARGET_MINER] - miner_low) / miner_span
            item["platform_score"] = (item[v4.TARGET_PLATFORM] - platform_low) / platform_span
            item["combined_score"] = 0.5 * item["miner_score"] + 0.5 * item["platform_score"]
            item["combined_low90"] = 0.5 * (
                (item["miner_low90"] - miner_low) / miner_span
                + (item["platform_low90"] - platform_low) / platform_span
            )
            item["combined_high90"] = 0.5 * (
                (item["miner_high90"] - miner_low) / miner_span
                + (item["platform_high90"] - platform_low) / platform_span
            )
        results.append(sorted(row_items, key=lambda item: (-item["combined_score"], item["business"])))
    return results


def observed_predictions(frame: pd.DataFrame, runtime: RuntimeModel) -> pd.DataFrame:
    predictions = raw_runtime_predictions(frame, runtime)
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(frame.to_dict(orient="records")):
        business = str(row["business"])
        base = v4.predict_business(row, business, runtime.baseline)
        calibration = runtime.metadata["calibration"].get(business, {})
        support_scale = math.sqrt(
            1.0 + 5.0 / max(float(base["local_effective_support"]) + 5.0, 5.0)
        )
        miner = float(predictions[v4.TARGET_MINER][position])
        platform = float(predictions[v4.TARGET_PLATFORM][position])
        rows.append({
            "node_id": row["node_id"],
            "sample_day": pd.Timestamp(row["sample_day"]).strftime("%Y-%m-%d"),
            "business": business,
            "business_name": base["business_name"],
            "sample_weight": float(row["sample_weight"]),
            "actual_miner_unit_income": float(row[v4.TARGET_MINER]),
            "actual_platform_unit_profit": float(row[v4.TARGET_PLATFORM]),
            "predicted_miner_unit_income": miner,
            "predicted_platform_unit_profit": platform,
            "local_effective_support": base["local_effective_support"],
            "propensity": base["propensity"],
            "confidence": base["confidence"],
            "miner_low90": miner - float(calibration.get("miner_radius90", 0.0)) * support_scale,
            "miner_high90": miner + float(calibration.get("miner_radius90", 0.0)) * support_scale,
            "platform_low90": platform - float(calibration.get("platform_radius90", 0.0)) * support_scale,
            "platform_high90": platform + float(calibration.get("platform_radius90", 0.0)) * support_scale,
        })
    return pd.DataFrame(rows)


def temporal_metrics(
    observed: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    weights = v1.sample_weights(observed)
    miner = v4.weighted_metrics(
        observed["actual_miner_unit_income"], observed["predicted_miner_unit_income"], weights
    )
    platform = v4.weighted_metrics(
        observed["actual_platform_unit_profit"], observed["predicted_platform_unit_profit"], weights
    )
    miner_covered = observed["actual_miner_unit_income"].between(
        observed["miner_low90"], observed["miner_high90"]
    )
    platform_covered = observed["actual_platform_unit_profit"].between(
        observed["platform_low90"], observed["platform_high90"]
    )
    metrics = {
        "raw_rows": int(len(observed)),
        "effective_support": float(weights.sum()),
        "unique_nodes": int(observed["node_id"].nunique()),
        "miner": miner,
        "platform": platform,
        "miner_interval_coverage90": float(np.average(miner_covered, weights=weights)),
        "platform_interval_coverage90": float(np.average(platform_covered, weights=weights)),
    }
    rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for business, group in observed.groupby("business"):
        group_weights = v1.sample_weights(group)
        business_miner = v4.weighted_metrics(
            group["actual_miner_unit_income"], group["predicted_miner_unit_income"], group_weights
        )
        business_platform = v4.weighted_metrics(
            group["actual_platform_unit_profit"], group["predicted_platform_unit_profit"], group_weights
        )
        miner_coverage = float(np.average(
            group["actual_miner_unit_income"].between(group["miner_low90"], group["miner_high90"]),
            weights=group_weights,
        ))
        platform_coverage = float(np.average(
            group["actual_platform_unit_profit"].between(
                group["platform_low90"], group["platform_high90"]
            ),
            weights=group_weights,
        ))
        item = {
            "business": str(business),
            "business_name": group["business_name"].iloc[0],
            "test_rows": int(len(group)),
            "test_effective_support": float(group_weights.sum()),
            "test_nodes": int(group["node_id"].nunique()),
            "miner_mae": business_miner["mae"],
            "miner_rmse": business_miner["rmse"],
            "miner_r2": business_miner["r2"],
            "platform_mae": business_platform["mae"],
            "platform_rmse": business_platform["rmse"],
            "platform_r2": business_platform["r2"],
            "miner_coverage90": miner_coverage,
            "platform_coverage90": platform_coverage,
            "joint_average_coverage90": 0.5 * (miner_coverage + platform_coverage),
        }
        rows.append(item)
        lookup[str(business)] = item
    return metrics, rows, lookup


def ranking_and_dr(frame: pd.DataFrame, runtime: RuntimeModel) -> dict[str, Any]:
    work = frame.copy().reset_index(drop=True)
    signatures = work[["node_id", *FEATURE_FIELDS]].astype(str).agg(v4.KEY_SEPARATOR.join, axis=1)
    unique = work.loc[~signatures.duplicated()].copy()
    unique["_signature"] = signatures.loc[~signatures.duplicated()].to_numpy()
    rankings = score_profile_rows(unique, runtime)
    ranking_lookup = dict(zip(unique["_signature"], rankings))
    weights = v1.sample_weights(work).to_numpy(dtype=float)
    hit1: list[bool] = []
    hit3: list[bool] = []
    dm_miner: list[float] = []
    dm_platform: list[float] = []
    dr_miner: list[float] = []
    dr_platform: list[float] = []
    observed_miner: list[float] = []
    observed_platform: list[float] = []
    correction_weights: list[float] = []
    distribution: dict[str, float] = {}
    matched_weight = 0.0
    for position, row in enumerate(work.to_dict(orient="records")):
        ranked = ranking_lookup[signatures.iloc[position]]
        policy = ranked[0]
        actual_business = str(row["business"])
        top_ids = [item["business"] for item in ranked[:3]]
        hit1.append(top_ids[0] == actual_business)
        hit3.append(actual_business in top_ids)
        distribution[top_ids[0]] = distribution.get(top_ids[0], 0.0) + weights[position]
        actual = next(item for item in ranked if item["business"] == actual_business)
        probability = max(float(actual["propensity"]), 0.05)
        correction = 1.0 / probability if top_ids[0] == actual_business else 0.0
        dm_miner.append(policy[v4.TARGET_MINER])
        dm_platform.append(policy[v4.TARGET_PLATFORM])
        dr_miner.append(
            policy[v4.TARGET_MINER]
            + correction * (float(row[v4.TARGET_MINER]) - actual[v4.TARGET_MINER])
        )
        dr_platform.append(
            policy[v4.TARGET_PLATFORM]
            + correction * (float(row[v4.TARGET_PLATFORM]) - actual[v4.TARGET_PLATFORM])
        )
        observed_miner.append(float(row[v4.TARGET_MINER]))
        observed_platform.append(float(row[v4.TARGET_PLATFORM]))
        correction_weights.append(weights[position] * correction)
        if correction:
            matched_weight += weights[position]
    corrections = np.asarray(correction_weights, dtype=float)
    effective_support = float(weights.sum())
    correction_ess = (
        float(corrections.sum() ** 2 / np.sum(corrections ** 2))
        if np.sum(corrections ** 2) > 0 else 0.0
    )
    return {
        "evaluated_rows": int(len(work)),
        "effective_support": effective_support,
        "unique_profile_signatures": int(len(unique)),
        "observed_business_hit_at_1": float(np.average(hit1, weights=weights)),
        "observed_business_hit_at_3": float(np.average(hit3, weights=weights)),
        "top1_distribution": [
            {"business": key, "effective_top1": value, "share": value / effective_support}
            for key, value in sorted(distribution.items(), key=lambda item: -item[1])
        ],
        "direct_method": {
            "miner_unit_income": float(np.average(dm_miner, weights=weights)),
            "platform_unit_profit": float(np.average(dm_platform, weights=weights)),
        },
        "doubly_robust": {
            "miner_unit_income": float(np.average(dr_miner, weights=weights)),
            "platform_unit_profit": float(np.average(dr_platform, weights=weights)),
            "policy_observed_match_rate": matched_weight / effective_support,
            "correction_effective_sample_size": correction_ess,
            "propensity_floor": 0.05,
        },
        "observed_policy": {
            "miner_unit_income": float(np.average(observed_miner, weights=weights)),
            "platform_unit_profit": float(np.average(observed_platform, weights=weights)),
        },
        "warning": (
            "DR is an observational diagnostic, not causal proof; unmeasured allocation factors may remain."
        ),
    }


def recommendation_output(
    row: dict[str, Any],
    ranked: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    selected = ranked[:top_k]
    gap = selected[0]["combined_score"] - selected[1]["combined_score"]
    action = v4.recommendation_action(selected[0], gap)
    bandwidth = max(v4.finite_number(row.get("bw"), 0.0), 0.0)
    current_business = v1.clean_cell(row.get("current_business"))
    current_name = v1.clean_cell(row.get("current_business_name"))
    lookup = {item["business"]: item for item in ranked}
    current = lookup.get(current_business)
    if not current_business:
        current_status = "current_business_unknown"
    elif current is None:
        current_status = "current_business_not_in_candidate_set"
    elif current_business == selected[0]["business"]:
        current_status = "already_top1"
    else:
        current_status = "current_not_top1"
    model_current_not_top1 = current_status == "current_not_top1"
    interval_dominates = bool(
        current is not None
        and selected[0]["combined_low90"] > current["combined_high90"]
        and gap >= 0.03
    )
    review = model_current_not_top1 and action != "暂不推荐执行" and interval_dominates
    current_cost = v4.finite_number(row.get("current_business_cost_sum"), float("nan"))
    current_profit = v4.finite_number(row.get("current_business_profit_sum"), float("nan"))
    output: dict[str, Any] = {
        "node_id": row["node_id"],
        "province": v4.clean(row.get("province")),
        "city": v4.clean(row.get("city")),
        "isp": v4.clean(row.get("isp")),
        "resourcetype": v4.clean(row.get("resourcetype")),
        "deliverytype": v4.clean(row.get("deliverytype")),
        "nattype": v4.clean(row.get("nattype")),
        "scheduleisps": v4.clean(row.get("scheduleisps")),
        "network_schedule_type": v4.clean(row.get("network_schedule_type")),
        "cpu_bucket": v4.clean(row.get("corenum_bucket")),
        "memory_bucket": v4.clean(row.get("memtotal_bucket")),
        "disk_bucket": v4.clean(row.get("totaldisksize_bucket")),
        "ipv6_capability": v4.clean(row.get("ipv6_capability")),
        "bandwidth_bucket": v4.clean(row.get("bw_bucket")),
        "packet_loss_satisfaction_pct": v4.finite_number(
            row.get("overall_packet_loss_benchmark_satisfaction_pct"), float("nan")
        ),
        "packet_loss_satisfaction_bucket": v4.clean(
            row.get("packet_loss_satisfaction_bucket")
        ),
        "build_bandwidth_mbps": bandwidth,
        "recommendation_action": action,
        "recommendation_confidence": selected[0]["confidence"],
        "top1_score_gap": gap,
        "recommendation_reason": v4.confidence_reason(selected[0]),
        "current_business": current_business,
        "current_business_name": current_name,
        "current_business_day": v1.clean_cell(row.get("current_business_day")),
        "current_business_source": v1.clean_cell(row.get("current_business_source")),
        "current_business_confidence": v1.clean_cell(row.get("current_business_confidence")),
        "current_business_status": current_status,
        "current_business_in_model": current is not None,
        "model_current_not_top1": model_current_not_top1,
        "interval_dominates_current": interval_dominates,
        "review_switch_candidate": review,
        "current_actual_miner_income_1d": current_cost,
        "current_actual_platform_profit_1d": current_profit,
        "current_actual_miner_unit_income": current_cost / bandwidth if bandwidth > 0 else float("nan"),
        "current_actual_platform_unit_profit": current_profit / bandwidth if bandwidth > 0 else float("nan"),
        "current_predicted_combined_score": current["combined_score"] if current else float("nan"),
        "current_predicted_miner_unit_income": current[v4.TARGET_MINER] if current else float("nan"),
        "current_predicted_platform_unit_profit": current[v4.TARGET_PLATFORM] if current else float("nan"),
        "top1_vs_current_predicted_miner_unit_delta": (
            selected[0][v4.TARGET_MINER] - current[v4.TARGET_MINER]
            if current else float("nan")
        ),
        "top1_vs_current_predicted_platform_unit_delta": (
            selected[0][v4.TARGET_PLATFORM] - current[v4.TARGET_PLATFORM]
            if current else float("nan")
        ),
    }
    details: list[dict[str, Any]] = []
    for index, item in enumerate(selected, 1):
        prefix = f"top{index}"
        output.update({
            f"business_{prefix}": item["business"],
            f"business_name_{prefix}": item["business_name"],
            f"combined_score_{prefix}": item["combined_score"],
            f"miner_unit_income_{prefix}": item[v4.TARGET_MINER],
            f"platform_unit_profit_{prefix}": item[v4.TARGET_PLATFORM],
            f"estimated_miner_income_1d_{prefix}": item[v4.TARGET_MINER] * bandwidth,
            f"estimated_platform_profit_1d_{prefix}": item[v4.TARGET_PLATFORM] * bandwidth,
            f"miner_unit_low90_{prefix}": item["miner_low90"],
            f"miner_unit_high90_{prefix}": item["miner_high90"],
            f"platform_unit_low90_{prefix}": item["platform_low90"],
            f"platform_unit_high90_{prefix}": item["platform_high90"],
            f"confidence_{prefix}": item["confidence"],
            f"local_effective_support_{prefix}": item["local_effective_support"],
            f"global_effective_support_{prefix}": item["global_effective_support"],
            f"historical_assignment_probability_{prefix}": item["propensity"],
            f"coverage_level_{prefix}": item["coverage_level"],
            f"reason_{prefix}": v4.confidence_reason(item),
        })
        details.append({key: value for key, value in item.items() if key not in {"ood_reasons", "target_bounds"}})
    output["top3_detail"] = json.dumps(details, ensure_ascii=False, separators=(",", ":"))
    return output


def recommend_nodes(nodes: pd.DataFrame, runtime: RuntimeModel, chunk_size: int = 400) -> pd.DataFrame:
    unique = nodes.drop_duplicates("node_id").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(unique), chunk_size):
        chunk = unique.iloc[start:start + chunk_size]
        rankings = score_profile_rows(chunk, runtime)
        for row, ranked in zip(chunk.to_dict(orient="records"), rankings):
            rows.append(recommendation_output(row, ranked))
    return pd.DataFrame(rows)


def comparison_summary(
    baseline_metrics: dict[str, Any],
    selected_metrics: dict[str, Any],
    selected_sources: dict[str, str],
    warm_enabled: dict[str, bool],
    cold_start: dict[str, Any],
    warm_start: dict[str, Any],
) -> dict[str, Any]:
    return {
        "baseline": baseline_metrics,
        "selected": selected_metrics,
        "relative_change": {
            "miner_rmse": (
                selected_metrics["miner"]["rmse"] / baseline_metrics["miner"]["rmse"] - 1.0
            ),
            "platform_rmse": (
                selected_metrics["platform"]["rmse"] / baseline_metrics["platform"]["rmse"] - 1.0
            ),
            "miner_r2_delta": selected_metrics["miner"]["r2"] - baseline_metrics["miner"]["r2"],
            "platform_r2_delta": (
                selected_metrics["platform"]["r2"] - baseline_metrics["platform"]["r2"]
            ),
        },
        "selected_sources": selected_sources,
        "warm_start_enabled": warm_enabled,
        "cold_start_test": cold_start,
        "warm_start_test": warm_start,
    }


def credibility_summary(
    temporal: dict[str, Any],
    ranking_dr: dict[str, Any],
    concentration: dict[str, Any],
    recommendations: pd.DataFrame,
    business_metrics: pd.DataFrame,
) -> dict[str, Any]:
    summary = v4.credibility_summary(
        temporal, ranking_dr, concentration, recommendations, business_metrics
    )
    blockers: list[str] = []
    low_share = float(recommendations["recommendation_confidence"].eq("low").mean())
    platform_cross_zero_share = float(
        recommendations["platform_unit_low90_top1"].lt(0).mean()
    )
    if temporal["platform"]["r2"] < 0.10:
        blockers.append("平台利润时间外R²低于0.10")
    if low_share > 0.50:
        blockers.append("低可信度节点超过50%")
    if platform_cross_zero_share > 0.50:
        blockers.append("超过50%的Top1平台利润区间跨零")
    if blockers:
        summary["level"] = "low"
    summary["blocking_reasons"] = blockers
    summary["low_confidence_node_share"] = low_share
    summary["top1_platform_interval_crosses_zero_share"] = platform_cross_zero_share
    return summary


def load_runtime(output_dir: Path) -> RuntimeModel:
    require_ml_dependencies()
    metadata = json.loads((output_dir / OUTPUT_MODEL).read_text(encoding="utf-8"))
    baseline = json.loads((output_dir / OUTPUT_BASELINE).read_text(encoding="utf-8"))
    boosters: dict[str, Any] = {}
    files = {
        v4.TARGET_MINER: OUTPUT_MINER_BOOSTER,
        v4.TARGET_PLATFORM: OUTPUT_PLATFORM_BOOSTER,
    }
    for target, source in metadata["selected_sources"].items():
        if source == "hierarchical_baseline":
            continue
        booster = xgb.Booster()
        booster.load_model(output_dir / files[target])
        boosters[target] = booster
    return RuntimeModel(metadata=metadata, baseline=baseline, boosters=boosters)


def build(args: argparse.Namespace) -> int:
    require_ml_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    business_names = v4.load_business_names(args.business_map)
    pairs = v4.load_training_pairs(
        args.pairs, args.historical_profiles, args.latest_pressure_profiles
    )
    pairs = filter_training_window(pairs, args.start_day, args.end_day)
    pairs, excluded_mixed_schedule_rows = filter_supported_schedule_rows(pairs)
    train, calibration, test, windows = v4.temporal_split(
        pairs, args.validation_days, args.calibration_days
    )
    candidates = v4.candidate_businesses(
        train, args.min_business_support, args.min_business_nodes
    )
    if len(candidates) < 3:
        raise RuntimeError("fewer than three businesses meet the V5 support thresholds")
    train = train[train["business"].isin(candidates)].copy()
    calibration = calibration[calibration["business"].isin(candidates)].copy()
    test = test[test["business"].isin(candidates)].copy()

    evaluation_baseline = v4.fit_model(
        train,
        candidates,
        business_names,
        smoothing_alpha=args.smoothing_alpha,
        min_segment_support=args.min_segment_support,
        learning_rate=args.learning_rate,
    )
    v4.attach_calibration(evaluation_baseline, calibration)
    baseline_calibration = baseline_predictions(calibration, evaluation_baseline)
    baseline_test = baseline_predictions(test, evaluation_baseline)
    baseline_train = baseline_predictions(train, evaluation_baseline)
    bounds = v4.target_bounds(train, candidates)
    encoder = fit_encoder(train)
    standard_weights = v1.sample_weights(train).to_numpy(dtype=float)
    ipw_weights = propensity_weights(train, evaluation_baseline)

    boosters: dict[str, dict[str, Any]] = {"xgboost_standard": {}, "xgboost_propensity": {}}
    prediction_bank: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "hierarchical_baseline": {
            "train": baseline_train,
            "calibration": baseline_calibration,
            "test": baseline_test,
        }
    }
    comparison_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        comparison_rows.append(metrics_row(
            "calibration", "all", "hierarchical_baseline", target,
            calibration, baseline_calibration[target]
        ))
        comparison_rows.append(metrics_row(
            "test", "all", "hierarchical_baseline", target, test, baseline_test[target]
        ))

    for model_name, weights in [
        ("xgboost_standard", standard_weights),
        ("xgboost_propensity", ipw_weights),
    ]:
        prediction_bank[model_name] = {"train": {}, "calibration": {}, "test": {}}
        for target_index, target in enumerate(TARGETS):
            booster = fit_booster(
                train,
                calibration,
                encoder,
                target,
                bounds,
                weights,
                rounds=args.boost_rounds,
                early_stopping_rounds=args.early_stopping_rounds,
                seed=args.seed + target_index,
            )
            boosters[model_name][target] = booster
            for split_name, split in [("train", train), ("calibration", calibration), ("test", test)]:
                prediction_bank[model_name][split_name][target] = booster_predictions(
                    split, booster, encoder, target, bounds
                )
            comparison_rows.append(metrics_row(
                "calibration", "all", model_name, target,
                calibration, prediction_bank[model_name]["calibration"][target]
            ))
            comparison_rows.append(metrics_row(
                "test", "all", model_name, target,
                test, prediction_bank[model_name]["test"][target]
            ))

    comparison = pd.DataFrame(comparison_rows)
    selected_sources: dict[str, str] = {}
    for target in TARGETS:
        target_rows = comparison[
            comparison["split"].eq("calibration") & comparison["target"].eq(target)
        ]
        selected_sources[target] = str(target_rows.sort_values(["rmse", "mae", "model"]).iloc[0]["model"])

    selected_train = {
        target: prediction_bank[selected_sources[target]]["train"][target] for target in TARGETS
    }
    selected_calibration_cold = {
        target: prediction_bank[selected_sources[target]]["calibration"][target] for target in TARGETS
    }
    selected_test_cold = {
        target: prediction_bank[selected_sources[target]]["test"][target] for target in TARGETS
    }
    effects = fit_node_effects(train, selected_train, args.node_effect_alpha)
    warm_enabled: dict[str, bool] = {}
    selected_calibration: dict[str, np.ndarray] = {}
    selected_test: dict[str, np.ndarray] = {}
    seen_calibration = calibration["node_id"].isin(set(train["node_id"]))
    for target in TARGETS:
        warm_calibration = apply_node_effect(
            calibration, selected_calibration_cold[target], target, effects, bounds
        )
        cold_metrics = v4.weighted_metrics(
            calibration.loc[seen_calibration, target],
            pd.Series(selected_calibration_cold[target][seen_calibration.to_numpy()]),
            v1.sample_weights(calibration.loc[seen_calibration]),
        )
        warm_metrics = v4.weighted_metrics(
            calibration.loc[seen_calibration, target],
            pd.Series(warm_calibration[seen_calibration.to_numpy()]),
            v1.sample_weights(calibration.loc[seen_calibration]),
        )
        warm_enabled[target] = (
            warm_metrics["rmse"]
            <= cold_metrics["rmse"] * (1.0 - args.warm_min_relative_improvement)
        )
        if warm_enabled[target]:
            selected_calibration[target] = warm_calibration
            selected_test[target] = apply_node_effect(
                test, selected_test_cold[target], target, effects, bounds
            )
        else:
            selected_calibration[target] = selected_calibration_cold[target]
            selected_test[target] = selected_test_cold[target]
        comparison_rows.append(metrics_row(
            "calibration", "seen_nodes", "selected_cold", target,
            calibration.loc[seen_calibration], selected_calibration_cold[target][seen_calibration.to_numpy()]
        ))
        comparison_rows.append(metrics_row(
            "calibration", "seen_nodes", "selected_hybrid", target,
            calibration.loc[seen_calibration], selected_calibration[target][seen_calibration.to_numpy()]
        ))

    calibration_map = fit_intervals(
        calibration, selected_calibration, evaluation_baseline
    )
    evaluation_metadata = {
        "version": "v5",
        "candidate_businesses": candidates,
        "selected_sources": selected_sources,
        "warm_start_enabled": warm_enabled,
        "encoder": encoder,
        "target_bounds": bounds,
        "node_effects": effects,
        "calibration": calibration_map,
        "validation_by_business": {},
    }
    evaluation_runtime = RuntimeModel(
        metadata=evaluation_metadata,
        baseline=evaluation_baseline,
        boosters={
            target: boosters[selected_sources[target]][target]
            for target in TARGETS
            if selected_sources[target] != "hierarchical_baseline"
        },
    )
    validation_predictions = observed_predictions(test, evaluation_runtime)
    selected_temporal, business_validation, validation_lookup = temporal_metrics(
        validation_predictions
    )
    evaluation_metadata["validation_by_business"] = validation_lookup
    evaluation_baseline["validation_by_business"] = validation_lookup
    ranking_dr = ranking_and_dr(test, evaluation_runtime)

    baseline_observed = test[["node_id", "business", "sample_weight", *TARGETS]].copy()
    baseline_observed["predicted_miner"] = baseline_test[v4.TARGET_MINER]
    baseline_observed["predicted_platform"] = baseline_test[v4.TARGET_PLATFORM]
    baseline_temporal = {
        "miner": v4.weighted_metrics(
            baseline_observed[v4.TARGET_MINER], baseline_observed["predicted_miner"],
            baseline_observed["sample_weight"]
        ),
        "platform": v4.weighted_metrics(
            baseline_observed[v4.TARGET_PLATFORM], baseline_observed["predicted_platform"],
            baseline_observed["sample_weight"]
        ),
    }

    seen_test = test["node_id"].isin(set(train["node_id"]))
    cohort_metrics: dict[str, dict[str, Any]] = {}
    for cohort_name, mask in [("cold_start", ~seen_test), ("warm_start", seen_test)]:
        cohort_metrics[cohort_name] = {}
        for target in TARGETS:
            cohort_metrics[cohort_name]["miner" if target == v4.TARGET_MINER else "platform"] = (
                v4.weighted_metrics(
                    test.loc[mask, target],
                    pd.Series(selected_test[target][mask.to_numpy()]),
                    v1.sample_weights(test.loc[mask]),
                )
            )
            comparison_rows.append(metrics_row(
                "test", cohort_name, "selected_hybrid", target,
                test.loc[mask], selected_test[target][mask.to_numpy()]
            ))
        cohort_metrics[cohort_name]["rows"] = int(mask.sum())
        cohort_metrics[cohort_name]["nodes"] = int(test.loc[mask, "node_id"].nunique())

    all_pairs = pairs[pairs["business"].isin(candidates)].copy()
    final_baseline = v4.fit_model(
        all_pairs,
        candidates,
        business_names,
        smoothing_alpha=args.smoothing_alpha,
        min_segment_support=args.min_segment_support,
        learning_rate=args.learning_rate,
    )
    final_baseline["calibration"] = calibration_map
    final_baseline["validation_by_business"] = validation_lookup
    final_baseline["temporal_windows"] = windows
    final_bounds = v4.target_bounds(all_pairs, candidates)
    final_encoder = fit_encoder(all_pairs)
    final_boosters: dict[str, Any] = {}
    final_source_predictions: dict[str, np.ndarray] = {}
    final_baseline_predictions = baseline_predictions(all_pairs, final_baseline)
    final_standard_weights = v1.sample_weights(all_pairs).to_numpy(dtype=float)
    final_ipw_weights = propensity_weights(all_pairs, final_baseline)
    for target_index, target in enumerate(TARGETS):
        source = selected_sources[target]
        if source == "hierarchical_baseline":
            final_source_predictions[target] = final_baseline_predictions[target]
            continue
        evaluation_booster = boosters[source][target]
        rounds = int(getattr(evaluation_booster, "best_iteration", args.boost_rounds - 1)) + 1
        weights = final_ipw_weights if source == "xgboost_propensity" else final_standard_weights
        final_booster = fit_final_booster(
            all_pairs, final_encoder, target, final_bounds, weights, rounds, args.seed + target_index
        )
        final_boosters[target] = final_booster
        final_source_predictions[target] = booster_predictions(
            all_pairs, final_booster, final_encoder, target, final_bounds
        )
    final_effects = fit_node_effects(all_pairs, final_source_predictions, args.node_effect_alpha)
    final_metadata = {
        "version": "v5",
        "model_type": "time_validated_profile_gradient_boosting_with_optional_node_residual",
        "objective": (
            "rank by 0.5 * candidate-normalized miner unit income + "
            "0.5 * candidate-normalized platform unit profit"
        ),
        "candidate_businesses": candidates,
        "business_names": {key: business_names.get(key, "") for key in candidates},
        "feature_whitelist": FEATURE_FIELDS,
        "allowed_schedule_types": sorted(ALLOWED_SCHEDULE_TYPES),
        "selected_sources": selected_sources,
        "warm_start_enabled": warm_enabled,
        "warm_start_minimum_calibration_improvement": args.warm_min_relative_improvement,
        "encoder": final_encoder,
        "target_bounds": final_bounds,
        "node_effects": final_effects,
        "node_effect_alpha": args.node_effect_alpha,
        "calibration": calibration_map,
        "validation_by_business": validation_lookup,
        "temporal_windows": windows,
        "sample_weight_policy": "1 / consecutive valid days in the same node-business run",
        "propensity_weight_policy": "stabilized and clipped to [0.25, 4.0], selected only on calibration",
    }
    final_runtime = RuntimeModel(final_metadata, final_baseline, final_boosters)

    current_nodes = pd.read_csv(args.current_nodes, dtype={"node_id": "string"}, low_memory=False)
    current_business = v4.load_current_business(args.current_business, business_names)
    if not current_business.empty:
        current_nodes = current_nodes.merge(current_business, on="node_id", how="left", sort=False)
    current_nodes = v4.enrich_latest_pressure(current_nodes, args.latest_pressure_profiles)
    current_nodes = v4.prepare_features(current_nodes)
    recommendations = recommend_nodes(current_nodes, final_runtime)
    concentration_frame, concentration = v4.concentration_diagnostics(recommendations)
    business_metrics = v4.training_business_summary(
        pairs, train, calibration, test, candidates, business_names, business_validation
    )
    credibility = credibility_summary(
        selected_temporal, ranking_dr, concentration, recommendations, business_metrics
    )
    model_comparison = comparison_summary(
        baseline_temporal,
        selected_temporal,
        selected_sources,
        warm_enabled,
        cohort_metrics["cold_start"],
        cohort_metrics["warm_start"],
    )
    data_quality = v4.load_source_data_quality(args.source_summary)
    data_quality["excluded_mixed_schedule_rows_v5"] = excluded_mixed_schedule_rows
    summary = {
        "version": "v5",
        "model_type": final_metadata["model_type"],
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
            "feature_whitelist": FEATURE_FIELDS,
            "allowed_schedule_types": sorted(ALLOWED_SCHEDULE_TYPES),
            "latest_pressure_policy": "one latest non-null Superset netbench result per node",
            "latest_pressure_time_alignment": "static latest snapshot reused for every historical day",
            "latest_pressure_validation_caveat": (
                "outcomes are time-split, but latest pressure is not reconstructed as-of each outcome day"
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
            "outlier_policy": "per-business weighted P01-P99 winsorization",
        },
        "model_comparison": model_comparison,
        "temporal_validation": selected_temporal,
        "ranking_and_dr_diagnostics": ranking_dr,
        "recommendation_concentration": concentration,
        "current_business_comparison": v4.current_business_comparison_summary(recommendations),
        "credibility": credibility,
        "report_nodes": int(len(recommendations)),
        "artifacts": {},
    }

    paths = {
        "model": args.output_dir / OUTPUT_MODEL,
        "baseline_model": args.output_dir / OUTPUT_BASELINE,
        "recommendations": args.output_dir / OUTPUT_RECOMMENDATIONS,
        "current_not_top1_watchlist": args.output_dir / OUTPUT_MISMATCH_WATCHLIST,
        "review_switch_candidates": args.output_dir / OUTPUT_SWITCH_CANDIDATES,
        "validation_predictions": args.output_dir / OUTPUT_VALIDATION,
        "business_metrics": args.output_dir / OUTPUT_BUSINESS_METRICS,
        "model_comparison": args.output_dir / OUTPUT_COMPARISON,
        "concentration": args.output_dir / OUTPUT_CONCENTRATION,
        "summary": args.output_dir / OUTPUT_SUMMARY,
        "report_data": args.output_dir / OUTPUT_REPORT_DATA,
        "report_html": args.output_dir / OUTPUT_REPORT_HTML,
    }
    paths["model"].write_text(
        json.dumps(final_metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    paths["baseline_model"].write_text(
        json.dumps(final_baseline, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if v4.TARGET_MINER in final_boosters:
        final_boosters[v4.TARGET_MINER].save_model(args.output_dir / OUTPUT_MINER_BOOSTER)
        paths["miner_booster"] = args.output_dir / OUTPUT_MINER_BOOSTER
    if v4.TARGET_PLATFORM in final_boosters:
        final_boosters[v4.TARGET_PLATFORM].save_model(args.output_dir / OUTPUT_PLATFORM_BOOSTER)
        paths["platform_booster"] = args.output_dir / OUTPUT_PLATFORM_BOOSTER
    recommendations.to_csv(paths["recommendations"], index=False)
    recommendations[recommendations["model_current_not_top1"]].to_csv(
        paths["current_not_top1_watchlist"], index=False
    )
    recommendations[recommendations["review_switch_candidate"]].to_csv(
        paths["review_switch_candidates"], index=False
    )
    validation_predictions.to_csv(paths["validation_predictions"], index=False)
    business_metrics.to_csv(paths["business_metrics"], index=False)
    pd.DataFrame(comparison_rows).to_csv(paths["model_comparison"], index=False)
    concentration_frame.to_csv(paths["concentration"], index=False)
    summary["artifacts"] = {key: str(path) for key, path in paths.items()}
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    v4.write_report_data(
        paths["report_data"], summary, recommendations, business_metrics, concentration_frame
    )
    report_data = paths["report_data"].read_text(encoding="utf-8").replace(
        "window.V4_REPORT_DATA", "window.V5_REPORT_DATA", 1
    )
    paths["report_data"].write_text(report_data, encoding="utf-8")
    template = args.report_template.read_text(encoding="utf-8")
    html = (
        template.replace("大节点业务收益推荐 V4", "大节点业务收益推荐 V5")
        .replace("v4_frontend_report_data.js", OUTPUT_REPORT_DATA)
        .replace("window.V4_REPORT_DATA", "window.V5_REPORT_DATA")
    )
    paths["report_html"].write_text(html, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def check(args: argparse.Namespace) -> int:
    required = [
        OUTPUT_MODEL,
        OUTPUT_BASELINE,
        OUTPUT_RECOMMENDATIONS,
        OUTPUT_MISMATCH_WATCHLIST,
        OUTPUT_SWITCH_CANDIDATES,
        OUTPUT_VALIDATION,
        OUTPUT_BUSINESS_METRICS,
        OUTPUT_COMPARISON,
        OUTPUT_CONCENTRATION,
        OUTPUT_SUMMARY,
        OUTPUT_REPORT_DATA,
        OUTPUT_REPORT_HTML,
    ]
    missing = [name for name in required if not (args.output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing V5 artifacts: {missing}")
    metadata = json.loads((args.output_dir / OUTPUT_MODEL).read_text(encoding="utf-8"))
    summary = json.loads((args.output_dir / OUTPUT_SUMMARY).read_text(encoding="utf-8"))
    recommendations = pd.read_csv(args.output_dir / OUTPUT_RECOMMENDATIONS, low_memory=False)
    switches = pd.read_csv(args.output_dir / OUTPUT_SWITCH_CANDIDATES, low_memory=False)
    watchlist = pd.read_csv(args.output_dir / OUTPUT_MISMATCH_WATCHLIST, low_memory=False)
    if metadata.get("version") != "v5" or len(metadata.get("candidate_businesses", [])) < 3:
        raise RuntimeError("invalid V5 model")
    for target, source in metadata["selected_sources"].items():
        if source != "hierarchical_baseline":
            filename = OUTPUT_MINER_BOOSTER if target == v4.TARGET_MINER else OUTPUT_PLATFORM_BOOSTER
            if not (args.output_dir / filename).exists():
                raise RuntimeError(f"missing selected booster: {filename}")
    if recommendations.empty or summary.get("report_nodes") != len(recommendations):
        raise RuntimeError("invalid V5 recommendations")
    duplicate_top3 = recommendations[["business_top1", "business_top2", "business_top3"]].nunique(
        axis=1, dropna=True
    ).lt(3)
    if duplicate_top3.any():
        raise RuntimeError("V5 recommendations contain duplicate Top3 businesses")
    if not switches.empty and (
        ~switches["review_switch_candidate"].astype(bool)
        | ~switches["interval_dominates_current"].astype(bool)
        | switches["recommendation_action"].eq("暂不推荐执行")
    ).any():
        raise RuntimeError("V5 switch candidate eligibility check failed")
    expected_watchlist = int(
        summary.get("current_business_comparison", {}).get("model_current_not_top1_nodes", 0)
    )
    if len(watchlist) != expected_watchlist or (
        not watchlist.empty and not watchlist["model_current_not_top1"].astype(bool).all()
    ):
        raise RuntimeError("V5 current-not-top1 watchlist mismatch")
    print(json.dumps({
        "status": "V5 artifacts check passed",
        "candidate_businesses": len(metadata["candidate_businesses"]),
        "selected_sources": metadata["selected_sources"],
        "warm_start_enabled": metadata["warm_start_enabled"],
        "report_nodes": len(recommendations),
        "current_not_top1_watchlist": len(watchlist),
        "review_switch_candidates": len(switches),
        "credibility": summary["credibility"]["level"],
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--pairs", type=Path, default=v4.DEFAULT_PAIRS)
    build_parser.add_argument("--historical-profiles", type=Path, default=v4.DEFAULT_HISTORICAL_PROFILES)
    build_parser.add_argument("--latest-pressure-profiles", type=Path, default=v4.DEFAULT_LATEST_PRESSURE_PROFILES)
    build_parser.add_argument("--current-nodes", type=Path, default=v4.DEFAULT_CURRENT_NODES)
    build_parser.add_argument("--current-business", type=Path, default=v4.DEFAULT_CURRENT_BUSINESS)
    build_parser.add_argument("--business-map", type=Path, default=v4.DEFAULT_BUSINESS_MAP)
    build_parser.add_argument("--source-summary", type=Path, default=v4.DEFAULT_SOURCE_SUMMARY)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--report-template", type=Path, default=v4.DEFAULT_REPORT_TEMPLATE)
    build_parser.add_argument("--validation-days", type=int, default=7)
    build_parser.add_argument("--calibration-days", type=int, default=3)
    build_parser.add_argument("--start-day", help="Optional inclusive training-window start day.")
    build_parser.add_argument("--end-day", help="Optional inclusive training-window end day.")
    build_parser.add_argument("--min-business-support", type=float, default=30.0)
    build_parser.add_argument("--min-business-nodes", type=int, default=20)
    build_parser.add_argument("--min-segment-support", type=float, default=2.0)
    build_parser.add_argument("--smoothing-alpha", type=float, default=20.0)
    build_parser.add_argument("--learning-rate", type=float, default=0.70)
    build_parser.add_argument("--node-effect-alpha", type=float, default=3.0)
    build_parser.add_argument("--warm-min-relative-improvement", type=float, default=0.03)
    build_parser.add_argument("--boost-rounds", type=int, default=500)
    build_parser.add_argument("--early-stopping-rounds", type=int, default=35)
    build_parser.add_argument("--seed", type=int, default=20260904)
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
