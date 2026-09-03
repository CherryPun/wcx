#!/usr/bin/env python3
"""Build a frontend report for business recommendations with expected revenue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v1_recommendation_pipeline as v1
import v2_ranking_model as v2


HERE = Path(__file__).resolve().parent
DEFAULT_TRAINING_PAIRS = (
    HERE
    / "recent_month_large_mainstream_1d_unit_bw_network_scope"
    / "v1_training_pairs_large_mainstream_recent_1m.csv"
)
DEFAULT_RECOMMENDATIONS = (
    HERE
    / "current_non_idc_large_scan_network_scope"
    / "current_non_idc_large_recommendations_20260902.csv"
)
DEFAULT_MODEL_SUMMARY = (
    HERE / "recent_month_large_mainstream_1d_unit_bw_network_scope" / "mainstream_large_training_summary.json"
)
DEFAULT_SCAN_SUMMARY = (
    HERE / "current_non_idc_large_scan_network_scope" / "current_non_idc_large_scan_summary_20260902.json"
)
DEFAULT_OUTPUT_HTML = HERE / "business_recommendation_profit_report_network_scope.html"
DEFAULT_OUTPUT_CSV = HERE / "business_recommendation_profit_report_network_scope.csv"
DEFAULT_OUTPUT_DATA_JS = HERE / "business_recommendation_profit_report_network_scope_data.js"

SEGMENT_LEVELS: list[tuple[str, list[str], int]] = [
    (
        "province_isp_resource_network_schedule",
        ["province", "isp", "resourcetype", "network_schedule_type"],
        10,
    ),
    ("province_isp_resource_bw", ["province", "isp", "resourcetype", "bw_bucket"], 20),
    ("province_isp_resource", ["province", "isp", "resourcetype"], 20),
    ("isp_resource_network_schedule", ["isp", "resourcetype", "network_schedule_type"], 20),
    ("province_isp", ["province", "isp"], 30),
    ("isp_resource", ["isp", "resourcetype"], 30),
    ("business_global", [], 1),
]

SWITCH_ACTIONS = {
    "review_switch_candidate",
    "review_current_in_top3_not_top1",
    "review_switch_current_not_mainstream",
    "current_business_unknown",
}

NUMERIC_COLUMNS = [
    "cum_cost_7d",
    "cum_revenue_7d",
    "cum_profit_7d",
    "effective_bandwidth_mbps",
    "cost_per_bandwidth_mbps",
    "revenue_per_bandwidth_mbps",
    "profit_per_bandwidth_mbps",
    "current_business_revenue_sum",
    "current_business_cost_sum",
    "current_business_profit_sum",
    "current_business_peak95_max",
    "current_peak95_mbps",
    "current_profit_per_peak95_mbps",
    "analysis_transprovrate",
    "bandwidth",
    "actualbandwidth",
    "bw",
]


def clean(value: Any) -> str:
    return v1.clean_cell(value)


def num(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_number(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output


def safe_bucket_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = v2.add_buckets(frame)
    for column in [
        "province", "city", "isp", "resourcetype", "bw_bucket", "actualbandwidth_bucket",
        "schedule_isp_scope", "schedule_province_scope", "network_schedule_type",
    ]:
        if column not in output.columns:
            output[column] = ""
        output[column] = output[column].fillna("").map(clean)
    return output


def key_for(row: pd.Series | dict[str, Any], fields: list[str], business: str) -> tuple[str, ...]:
    return tuple(clean(row.get(field, "")) for field in fields) + (clean(business),)


def build_estimate_lookup(training_pairs: pd.DataFrame) -> dict[str, dict[tuple[str, ...], dict[str, Any]]]:
    frame = safe_bucket_frame(training_pairs)
    frame = to_number(frame, NUMERIC_COLUMNS)
    if "cum_profit_7d" not in frame.columns:
        frame["cum_profit_7d"] = frame["cum_revenue_7d"] - frame["cum_cost_7d"]
    if "effective_bandwidth_mbps" not in frame.columns:
        frame["effective_bandwidth_mbps"] = v1.effective_bandwidth_mbps(frame)
    if "profit_per_bandwidth_mbps" not in frame.columns:
        frame["profit_per_bandwidth_mbps"] = frame["cum_profit_7d"] / frame["effective_bandwidth_mbps"].replace(0, np.nan)
    if "revenue_per_bandwidth_mbps" not in frame.columns:
        frame["revenue_per_bandwidth_mbps"] = frame["cum_revenue_7d"] / frame["effective_bandwidth_mbps"].replace(0, np.nan)
    if "cost_per_bandwidth_mbps" not in frame.columns:
        frame["cost_per_bandwidth_mbps"] = frame["cum_cost_7d"] / frame["effective_bandwidth_mbps"].replace(0, np.nan)
    frame = to_number(frame, NUMERIC_COLUMNS)
    group_id = "outcome_group_id" if "outcome_group_id" in frame.columns else "node_id"

    lookup: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}
    for level_name, fields, _min_support in SEGMENT_LEVELS:
        group_columns = fields + ["business"]
        grouped = (
            frame.groupby(group_columns, dropna=False)
            .agg(
                sample_rows=("node_id", "size"),
                sample_groups=(group_id, "nunique"),
                sample_nodes=("node_id", "nunique"),
                business_name=("business_name", v1.latest_nonempty),
                est_cost_1d=("cum_cost_7d", "median"),
                est_revenue_1d=("cum_revenue_7d", "median"),
                est_profit_1d=("cum_profit_7d", "median"),
                avg_cost_1d=("cum_cost_7d", "mean"),
                avg_revenue_1d=("cum_revenue_7d", "mean"),
                avg_profit_1d=("cum_profit_7d", "mean"),
                est_cost_per_mbps=("cost_per_bandwidth_mbps", "median"),
                est_revenue_per_mbps=("revenue_per_bandwidth_mbps", "median"),
                est_profit_per_mbps=("profit_per_bandwidth_mbps", "median"),
                avg_profit_per_mbps=("profit_per_bandwidth_mbps", "mean"),
                p75_profit_1d=("cum_profit_7d", lambda value: pd.to_numeric(value, errors="coerce").quantile(0.75)),
                p25_profit_1d=("cum_profit_7d", lambda value: pd.to_numeric(value, errors="coerce").quantile(0.25)),
            )
            .reset_index()
        )
        level_lookup: dict[tuple[str, ...], dict[str, Any]] = {}
        for record in grouped.to_dict(orient="records"):
            business = clean(record.get("business"))
            key = tuple(clean(record.get(field, "")) for field in fields) + (business,)
            level_lookup[key] = {
                "estimate_source": level_name,
                "estimate_fields": ",".join(fields) if fields else "business",
                "sample_rows": int(num(record.get("sample_rows"))),
                "sample_groups": int(num(record.get("sample_groups"))),
                "sample_nodes": int(num(record.get("sample_nodes"))),
                "business_name": clean(record.get("business_name")),
                "est_cost_1d": num(record.get("est_cost_1d")),
                "est_revenue_1d": num(record.get("est_revenue_1d")),
                "est_profit_1d": num(record.get("est_profit_1d")),
                "avg_cost_1d": num(record.get("avg_cost_1d")),
                "avg_revenue_1d": num(record.get("avg_revenue_1d")),
                "avg_profit_1d": num(record.get("avg_profit_1d")),
                "est_cost_per_mbps": num(record.get("est_cost_per_mbps")),
                "est_revenue_per_mbps": num(record.get("est_revenue_per_mbps")),
                "est_profit_per_mbps": num(record.get("est_profit_per_mbps")),
                "avg_profit_per_mbps": num(record.get("avg_profit_per_mbps")),
                "p75_profit_1d": num(record.get("p75_profit_1d")),
                "p25_profit_1d": num(record.get("p25_profit_1d")),
            }
        lookup[level_name] = level_lookup
    return lookup


def estimate_for(
    row: pd.Series,
    business: str,
    lookup: dict[str, dict[tuple[str, ...], dict[str, Any]]],
) -> dict[str, Any]:
    business = clean(business)
    for level_name, fields, min_support in SEGMENT_LEVELS:
        item = lookup.get(level_name, {}).get(key_for(row, fields, business))
        if item and int(item["sample_groups"]) >= min_support:
            return item
    return {
        "estimate_source": "missing",
        "estimate_fields": "",
        "sample_rows": 0,
        "sample_groups": 0,
        "sample_nodes": 0,
        "business_name": "",
        "est_cost_1d": 0.0,
        "est_revenue_1d": 0.0,
        "est_profit_1d": 0.0,
        "avg_cost_1d": 0.0,
        "avg_revenue_1d": 0.0,
        "avg_profit_1d": 0.0,
        "est_cost_per_mbps": 0.0,
        "est_revenue_per_mbps": 0.0,
        "est_profit_per_mbps": 0.0,
        "avg_profit_per_mbps": 0.0,
        "p75_profit_1d": 0.0,
        "p25_profit_1d": 0.0,
    }


def enrich_recommendations(
    recommendations: pd.DataFrame,
    lookup: dict[str, dict[tuple[str, ...], dict[str, Any]]],
) -> pd.DataFrame:
    frame = safe_bucket_frame(recommendations)
    frame = to_number(frame, NUMERIC_COLUMNS + [f"v2_rank_score_top{idx}" for idx in range(1, 4)])
    frame["effective_bandwidth_mbps"] = v1.effective_bandwidth_mbps(frame)
    if "current_business_profit_sum" not in frame.columns:
        frame["current_business_profit_sum"] = 0.0
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        item: dict[str, Any] = {
            "node_id": clean(row.get("node_id")),
            "province": clean(row.get("province")),
            "city": clean(row.get("city")),
            "isp": clean(row.get("isp")),
            "resourcetype": clean(row.get("resourcetype")),
            "scheduleisps": clean(row.get("scheduleisps")),
            "scheduleisps_text": clean(row.get("scheduleisps_text")),
            "schedule_isp_scope": clean(row.get("schedule_isp_scope")),
            "schedule_province_scope": clean(row.get("schedule_province_scope")),
            "network_schedule_type": clean(row.get("network_schedule_type")),
            "analysis_transprovrate": num(row.get("analysis_transprovrate")),
            "analysis_transprovrate_bucket": clean(row.get("analysis_transprovrate_bucket")),
            "join_isbantransprov": clean(row.get("join_isbantransprov")),
            "join_isipv6schedule": clean(row.get("join_isipv6schedule")),
            "bw_bucket": clean(row.get("bw_bucket")),
            "actualbandwidth_bucket": clean(row.get("actualbandwidth_bucket")),
            "bandwidth": num(row.get("bandwidth")),
            "actualbandwidth": num(row.get("actualbandwidth")),
            "bw": num(row.get("bw")),
            "effective_bandwidth_mbps": num(row.get("effective_bandwidth_mbps")),
            "current_business": clean(row.get("current_business")),
            "current_business_name": clean(row.get("current_business_name")),
            "current_business_is_mainstream": bool_value(row.get("current_business_is_mainstream")),
            "current_business_day": clean(row.get("current_business_day")),
            "current_revenue_1d": num(row.get("current_business_revenue_sum")),
            "current_cost_1d": num(row.get("current_business_cost_sum")),
            "current_profit_1d": num(row.get("current_business_profit_sum")),
            "current_peak95_mbps": num(row.get("current_peak95_mbps")),
            "current_profit_per_peak95_mbps": num(row.get("current_profit_per_peak95_mbps")),
            "suggested_action": clean(row.get("suggested_action")),
            "current_not_best": bool_value(row.get("current_not_best", "")),
            "model_current_not_top1": bool_value(row.get("model_current_not_top1", "")),
        }
        for rank in range(1, 4):
            business = clean(row.get(f"v2_business_top{rank}"))
            estimate = estimate_for(row, business, lookup)
            business_name = clean(row.get(f"v2_business_name_top{rank}")) or estimate["business_name"]
            item[f"business_top{rank}"] = business
            item[f"business_name_top{rank}"] = business_name
            item[f"rank_score_top{rank}"] = num(row.get(f"v2_rank_score_top{rank}"))
            item[f"expected_score_top{rank}"] = num(row.get(f"v2_expected_score_top{rank}"))
            item[f"expected_best_rate_top{rank}"] = num(row.get(f"v2_expected_best_rate_top{rank}"))
            item[f"risk_level_top{rank}"] = clean(row.get(f"v2_risk_level_top{rank}"))
            item[f"risk_reasons_top{rank}"] = clean(row.get(f"v2_risk_reasons_top{rank}"))
            item[f"reason_top{rank}"] = clean(row.get(f"v2_reason_top{rank}"))
            item[f"estimate_source_top{rank}"] = estimate["estimate_source"]
            item[f"estimate_support_groups_top{rank}"] = estimate["sample_groups"]
            item[f"estimate_support_nodes_top{rank}"] = estimate["sample_nodes"]
            item[f"est_revenue_1d_top{rank}"] = estimate["est_revenue_1d"]
            item[f"est_cost_1d_top{rank}"] = estimate["est_cost_1d"]
            item[f"est_profit_1d_top{rank}"] = estimate["est_profit_1d"]
            item[f"avg_profit_1d_top{rank}"] = estimate["avg_profit_1d"]
            item[f"p25_profit_1d_top{rank}"] = estimate["p25_profit_1d"]
            item[f"p75_profit_1d_top{rank}"] = estimate["p75_profit_1d"]
            item[f"est_profit_per_mbps_top{rank}"] = estimate["est_profit_per_mbps"]
            item[f"avg_profit_per_mbps_top{rank}"] = estimate["avg_profit_per_mbps"]
            item[f"est_profit_scaled_by_bw_top{rank}"] = estimate["est_profit_per_mbps"] * item["effective_bandwidth_mbps"]
        item["est_profit_delta_top1"] = item["est_profit_1d_top1"] - item["current_profit_1d"]
        item["est_revenue_delta_top1"] = item["est_revenue_1d_top1"] - item["current_revenue_1d"]
        item["positive_profit_uplift_top1"] = max(0.0, item["est_profit_delta_top1"])
        should_switch = item["suggested_action"] in SWITCH_ACTIONS
        item["policy_choice"] = "recommend_top1" if should_switch else "keep_current"
        item["policy_revenue_1d"] = item["est_revenue_1d_top1"] if should_switch else item["current_revenue_1d"]
        item["policy_cost_1d"] = item["est_cost_1d_top1"] if should_switch else item["current_cost_1d"]
        item["policy_profit_1d"] = item["est_profit_1d_top1"] if should_switch else item["current_profit_1d"]
        item["policy_profit_delta_1d"] = item["policy_profit_1d"] - item["current_profit_1d"]
        item["positive_policy_profit_uplift_1d"] = max(0.0, item["policy_profit_delta_1d"])
        rows.append(item)
    return pd.DataFrame(rows)


def top_distribution(frame: pd.DataFrame, column: str, name_column: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    grouped = frame.groupby(column, dropna=False).agg(count=("node_id", "nunique")).reset_index()
    if name_column and name_column in frame.columns:
        names = frame.groupby(column, dropna=False)[name_column].agg(v1.latest_nonempty).reset_index()
        grouped = grouped.merge(names, on=column, how="left")
    grouped = grouped.sort_values("count", ascending=False).head(limit)
    rows = []
    for record in grouped.to_dict(orient="records"):
        value = clean(record.get(column))
        rows.append({
            "id": value,
            "name": clean(record.get(name_column)) if name_column else value,
            "count": int(num(record.get("count"))),
        })
    return rows


def build_payload(
    enriched: pd.DataFrame,
    model_summary: dict[str, Any],
    scan_summary: dict[str, Any],
) -> dict[str, Any]:
    current_with_business = enriched[enriched["current_business"].ne("")]
    current_mainstream = current_with_business[current_with_business["current_business_is_mainstream"]]
    positive = enriched["positive_profit_uplift_top1"].clip(lower=0.0)
    policy_positive = enriched["positive_policy_profit_uplift_1d"].clip(lower=0.0)
    review_actions = enriched["suggested_action"].isin(SWITCH_ACTIONS)
    date_window = model_summary.get("date_window", {})
    training_start = model_summary.get("sample_day_min", "") or date_window.get("start", "")
    training_end = model_summary.get("sample_day_max", "") or date_window.get("end", "")
    target = model_summary.get("target", {})
    top1_counts = enriched["business_name_top1"].value_counts()
    top1_share = float(top1_counts.iloc[0] / len(enriched)) if len(enriched) and not top1_counts.empty else 0.0
    is_v3 = str(model_summary.get("version", "")).startswith("v3")
    return {
        "meta": {
            "title": "大节点业务推荐与预计收益报告",
            "model": "V3 单日建设带宽收益率模型" if is_v3 else "1d 单位带宽收益率模型",
            "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "training_window": f"{training_start} 至 {training_end}",
            "current_window": scan_summary.get("current_business_window", {}),
            "target_mode": model_summary.get("target_mode", "") or target.get("objective", ""),
            "rank_score_weights": model_summary.get("rank_score_weights", "") or target.get("rank_weights", ""),
            "estimate_rule": "历史同画像分段+同业务单日收益中位数，样本不足逐级退到全局业务",
            "caveat": (
                "观察性历史估算，不代表因果收益或切换承诺。"
                f"当前 Top1 最高集中度为 {top1_share:.1%}，待处理节点必须人工复核并通过小流量试验。"
            ),
        },
        "summary": {
            "nodes": int(len(enriched)),
            "current_business_nodes": int(len(current_with_business)),
            "current_mainstream_nodes": int(len(current_mainstream)),
            "current_not_best_nodes": int(enriched["current_not_best"].sum()),
            "review_switch_candidates": int(review_actions.sum()),
            "current_profit_1d_sum": float(enriched["current_profit_1d"].sum()),
            "est_profit_1d_sum_top1": float(enriched["est_profit_1d_top1"].sum()),
            "est_profit_delta_1d_sum_top1": float(enriched["est_profit_delta_top1"].sum()),
            "positive_profit_uplift_1d_sum_top1": float(positive.sum()),
            "policy_profit_1d_sum": float(enriched["policy_profit_1d"].sum()),
            "policy_profit_delta_1d_sum": float(enriched["policy_profit_delta_1d"].sum()),
            "positive_policy_profit_uplift_1d_sum": float(policy_positive.sum()),
            "avg_est_profit_per_mbps_top1": float(enriched["est_profit_per_mbps_top1"].mean()),
            "top1_max_share": top1_share,
            "v2_test": model_summary.get("v2_test", {}),
        },
        "action_distribution": top_distribution(enriched, "suggested_action", None, 20),
        "top1_distribution": top_distribution(enriched, "business_top1", "business_name_top1", 20),
        "rows": json.loads(enriched.to_json(orient="records", force_ascii=False)),
    }


def write_data_js(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        "window.BUSINESS_RECOMMENDATION_PROFIT_REPORT = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )


def write_html(path: Path, data_js_name: str) -> None:
    html = HTML_TEMPLATE.replace("__DATA_JS__", data_js_name)
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build business recommendation frontend report with expected revenue.")
    parser.add_argument("--training-pairs", type=Path, default=DEFAULT_TRAINING_PAIRS)
    parser.add_argument("--recommendations", type=Path, default=DEFAULT_RECOMMENDATIONS)
    parser.add_argument("--model-summary", type=Path, default=DEFAULT_MODEL_SUMMARY)
    parser.add_argument("--scan-summary", type=Path, default=DEFAULT_SCAN_SUMMARY)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-data-js", type=Path, default=DEFAULT_OUTPUT_DATA_JS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    training = pd.read_csv(args.training_pairs, dtype="string", low_memory=False).fillna("")
    recommendations = pd.read_csv(args.recommendations, dtype="string", low_memory=False).fillna("")
    lookup = build_estimate_lookup(training)
    enriched = enrich_recommendations(recommendations, lookup)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output_csv, index=False)
    payload = build_payload(enriched, read_json(args.model_summary), read_json(args.scan_summary))
    write_data_js(args.output_data_js, payload)
    write_html(args.output_html, args.output_data_js.name)
    print(json.dumps({
        "html": str(args.output_html),
        "csv": str(args.output_csv),
        "data_js": str(args.output_data_js),
        "rows": int(len(enriched)),
        "current_profit_1d_sum": payload["summary"]["current_profit_1d_sum"],
        "estimated_profit_1d_sum_top1": payload["summary"]["est_profit_1d_sum_top1"],
        "policy_profit_1d_sum": payload["summary"]["policy_profit_1d_sum"],
        "policy_profit_delta_1d_sum": payload["summary"]["policy_profit_delta_1d_sum"],
    }, ensure_ascii=False, indent=2))
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>大节点业务推荐与预计收益报告</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --surface: #ffffff;
      --surface-soft: #eef3f8;
      --text: #17202a;
      --muted: #647181;
      --line: #d9e2ec;
      --blue: #2563eb;
      --blue-soft: #dbeafe;
      --green: #15803d;
      --green-soft: #dcfce7;
      --amber: #b45309;
      --amber-soft: #fef3c7;
      --red: #b91c1c;
      --red-soft: #fee2e2;
      --slate: #334155;
      --slate-soft: #e2e8f0;
      --shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    button, input, select { font: inherit; }
    .app { min-height: 100vh; display: grid; grid-template-rows: auto auto auto 1fr; }
    .topbar {
      background: #111827;
      color: #fff;
      padding: 18px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    h1 { margin: 0; font-size: 22px; font-weight: 760; }
    .subtitle { margin-top: 4px; color: #cbd5e1; font-size: 13px; }
    .top-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
    .btn {
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 8px;
      background: var(--blue);
      color: #fff;
      padding: 8px 12px;
      cursor: pointer;
      white-space: nowrap;
    }
    .btn.secondary { background: transparent; border-color: #64748b; color: #e2e8f0; }
    .btn.light { background: #fff; color: var(--slate); border-color: var(--line); }
    .btn:hover { filter: brightness(0.96); }
    .filters {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(245, 247, 251, 0.97);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
      padding: 14px 28px;
    }
    .caveat {
      margin: 0;
      padding: 10px 28px;
      border-bottom: 1px solid #f5d08a;
      background: #fff7df;
      color: #7c4a03;
      font-size: 13px;
    }
    .filter-grid {
      display: grid;
      grid-template-columns: 2fr repeat(6, minmax(120px, 1fr)) 150px;
      gap: 10px;
      align-items: end;
    }
    .field { display: grid; gap: 5px; min-width: 0; }
    .field label { color: var(--muted); font-size: 12px; }
    .field input, .field select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
    }
    .checkline {
      min-height: 38px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--slate);
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 8px 10px;
      white-space: nowrap;
    }
    main { padding: 22px 28px 36px; display: grid; gap: 16px; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
      min-height: 92px;
    }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { margin-top: 7px; font-size: 24px; font-weight: 760; color: var(--slate); }
    .metric .note { margin-top: 4px; color: var(--muted); font-size: 12px; min-height: 18px; }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
      overflow: hidden;
    }
    .panel-header {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .panel-title { margin: 0; font-size: 15px; font-weight: 720; }
    .panel-meta { color: var(--muted); font-size: 12px; }
    .visual-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .bars { padding: 12px 16px 16px; display: grid; gap: 9px; }
    .bar-row { display: grid; grid-template-columns: minmax(130px, 230px) 1fr 54px; gap: 10px; align-items: center; }
    .bar-label { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; color: var(--slate); }
    .bar-track { height: 10px; background: var(--surface-soft); border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--blue); border-radius: 999px; }
    .bar-value { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
    .table-wrap { overflow: auto; max-height: 68vh; }
    table { width: 100%; border-collapse: collapse; min-width: 1520px; }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f8fafc;
      color: #475569;
      font-size: 12px;
      text-align: left;
      white-space: nowrap;
    }
    td { color: #1f2937; }
    .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .node { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 12px; }
    .muted { color: var(--muted); }
    .business-cell { display: grid; gap: 4px; min-width: 170px; }
    .business-id { color: var(--muted); font-size: 12px; }
    .top3 { display: grid; gap: 5px; min-width: 250px; }
    .top3-line { display: flex; align-items: center; gap: 6px; min-width: 0; }
    .rank {
      width: 20px;
      height: 20px;
      border-radius: 999px;
      display: inline-grid;
      place-items: center;
      font-size: 12px;
      background: var(--slate-soft);
      color: var(--slate);
      flex: 0 0 auto;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      max-width: 220px;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .risk-low { background: var(--green-soft); color: var(--green); }
    .risk-medium, .risk-unknown { background: var(--amber-soft); color: var(--amber); }
    .risk-high { background: var(--red-soft); color: var(--red); }
    .action-keep_top1, .action-keep_current_realized_value { background: var(--green-soft); color: var(--green); }
    .action-observe_current_profitable_high_risk { background: var(--amber-soft); color: var(--amber); }
    .action-review_switch_candidate, .action-review_switch_current_not_mainstream, .action-review_current_in_top3_not_top1 { background: var(--red-soft); color: var(--red); }
    .action-current_business_unknown { background: var(--slate-soft); color: var(--slate); }
    .profit-pos { color: var(--green); }
    .profit-neg { color: var(--red); }
    .footer-row {
      padding: 12px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-top: 1px solid var(--line);
      background: #f8fafc;
    }
    @media (max-width: 1100px) {
      .topbar { align-items: flex-start; flex-direction: column; }
      .top-actions { justify-content: flex-start; }
      .filter-grid { grid-template-columns: 1fr 1fr; }
      .summary-grid { grid-template-columns: 1fr 1fr; }
      .visual-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .topbar, .filters, .caveat, main { padding-left: 14px; padding-right: 14px; }
      .filter-grid, .summary-grid { grid-template-columns: 1fr; }
      h1 { font-size: 20px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div>
        <h1>大节点业务推荐与预计收益报告</h1>
        <div class="subtitle" id="subtitle"></div>
      </div>
      <div class="top-actions">
        <button class="btn secondary" id="resetBtn">重置筛选</button>
        <button class="btn" id="exportBtn">导出当前明细</button>
      </div>
    </header>

    <section class="filters">
      <div class="filter-grid">
        <div class="field">
          <label for="searchInput">搜索</label>
          <input id="searchInput" type="search" placeholder="节点、业务、省份、城市">
        </div>
        <div class="field">
          <label for="provinceSelect">省份</label>
          <select id="provinceSelect"></select>
        </div>
        <div class="field">
          <label for="ispSelect">运营商</label>
          <select id="ispSelect"></select>
        </div>
        <div class="field">
          <label for="businessSelect">推荐业务</label>
          <select id="businessSelect"></select>
        </div>
        <div class="field">
          <label for="scheduleSelect">网络调度类型</label>
          <select id="scheduleSelect"></select>
        </div>
        <div class="field">
          <label for="actionSelect">动作</label>
          <select id="actionSelect"></select>
        </div>
        <div class="field">
          <label for="riskSelect">风险</label>
          <select id="riskSelect"></select>
        </div>
        <label class="checkline">
          <input id="onlyReview" type="checkbox">
          <span>只看待处理</span>
        </label>
      </div>
    </section>

    <p class="caveat" id="caveat"></p>

    <main>
      <section class="summary-grid" id="summaryGrid"></section>

      <section class="visual-grid">
        <div class="panel">
          <div class="panel-header">
            <h2 class="panel-title">推荐 Top1 分布</h2>
            <span class="panel-meta" id="topBusinessMeta"></span>
          </div>
          <div class="bars" id="businessBars"></div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <h2 class="panel-title">处理动作分布</h2>
            <span class="panel-meta" id="actionMeta"></span>
          </div>
          <div class="bars" id="actionBars"></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2 class="panel-title">节点推荐明细</h2>
          <span class="panel-meta" id="tableMeta"></span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>节点</th>
                <th>画像</th>
                <th>当前业务</th>
                <th class="num">当前日收入</th>
                <th class="num">当前日利润</th>
                <th>推荐 Top3</th>
                <th class="num">Top1预计日收入</th>
                <th class="num">Top1预计矿主收益</th>
                <th class="num">Top1预计平台利润</th>
                <th class="num">动作后预计利润</th>
                <th class="num">利润/Mbps</th>
                <th class="num">动作后提升</th>
                <th>样本</th>
                <th>风险</th>
                <th>动作</th>
              </tr>
            </thead>
            <tbody id="tableBody"></tbody>
          </table>
        </div>
        <div class="footer-row">
          <span class="muted" id="pageMeta"></span>
          <button class="btn light" id="moreBtn">加载更多</button>
        </div>
      </section>
    </main>
  </div>

  <script src="__DATA_JS__"></script>
  <script>
    const payload = window.BUSINESS_RECOMMENDATION_PROFIT_REPORT || {meta: {}, summary: {}, rows: []};
    const rows = payload.rows || [];
    const state = { limit: 200, filtered: rows };
    const actionLabels = {
      keep_top1: "保留 Top1",
      keep_current_realized_value: "保留当前收益",
      observe_current_profitable_high_risk: "盈利观察",
      review_switch_candidate: "建议复核切换",
      review_switch_current_not_mainstream: "非主流复核",
      review_current_in_top3_not_top1: "Top3 内复核",
      current_business_unknown: "当前业务未知"
    };
    const riskLabels = { low: "低", medium: "中", high: "高", unknown: "未知" };
    const reviewActions = new Set([
      "review_switch_candidate",
      "review_switch_current_not_mainstream",
      "review_current_in_top3_not_top1",
      "current_business_unknown"
    ]);
    const ids = {
      search: document.getElementById("searchInput"),
      province: document.getElementById("provinceSelect"),
      isp: document.getElementById("ispSelect"),
      business: document.getElementById("businessSelect"),
      schedule: document.getElementById("scheduleSelect"),
      action: document.getElementById("actionSelect"),
      risk: document.getElementById("riskSelect"),
      onlyReview: document.getElementById("onlyReview")
    };

    function fmtInt(value) {
      return Math.round(Number(value || 0)).toLocaleString("zh-CN");
    }
    function fmtMoney(value) {
      const number = Number(value || 0);
      const sign = number < 0 ? "-" : "";
      return sign + Math.abs(number).toLocaleString("zh-CN", {maximumFractionDigits: 2});
    }
    function fmtRate(value) {
      return Number(value || 0).toLocaleString("zh-CN", {maximumFractionDigits: 4});
    }
    function moneyClass(value) {
      return Number(value || 0) >= 0 ? "profit-pos" : "profit-neg";
    }
    function uniqueOptions(key, labelKey) {
      const map = new Map();
      rows.forEach(row => {
        const value = (row[key] || "").toString();
        if (!value) return;
        const label = labelKey ? (row[labelKey] || value).toString() : value;
        map.set(value, label);
      });
      return [...map.entries()].sort((a, b) => a[1].localeCompare(b[1], "zh-CN"));
    }
    function fillSelect(select, options, allLabel) {
      select.innerHTML = "";
      const all = document.createElement("option");
      all.value = "";
      all.textContent = allLabel;
      select.appendChild(all);
      options.forEach(([value, label]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        select.appendChild(option);
      });
    }
    function initFilters() {
      fillSelect(ids.province, uniqueOptions("province"), "全部省份");
      fillSelect(ids.isp, uniqueOptions("isp"), "全部运营商");
      fillSelect(ids.business, uniqueOptions("business_top1", "business_name_top1"), "全部推荐业务");
      fillSelect(ids.schedule, uniqueOptions("network_schedule_type"), "全部调度类型");
      fillSelect(ids.action, uniqueOptions("suggested_action").map(([v]) => [v, actionLabels[v] || v]), "全部动作");
      fillSelect(ids.risk, [["low", "低"], ["medium", "中"], ["high", "高"], ["unknown", "未知"]], "全部风险");
      Object.values(ids).forEach(el => el.addEventListener("input", applyFilters));
    }
    function rowText(row) {
      return [
        row.node_id, row.province, row.city, row.isp, row.current_business_name, row.current_business,
        row.business_name_top1, row.business_top1, row.business_name_top2, row.business_name_top3,
        row.network_schedule_type, row.scheduleisps_text, row.scheduleisps,
        row.analysis_transprovrate, row.join_isbantransprov, row.join_isipv6schedule,
        row.suggested_action
      ].join(" ").toLowerCase();
    }
    function applyFilters() {
      const q = ids.search.value.trim().toLowerCase();
      state.filtered = rows.filter(row => {
        if (q && !rowText(row).includes(q)) return false;
        if (ids.province.value && row.province !== ids.province.value) return false;
        if (ids.isp.value && row.isp !== ids.isp.value) return false;
        if (ids.business.value && row.business_top1 !== ids.business.value) return false;
        if (ids.schedule.value && row.network_schedule_type !== ids.schedule.value) return false;
        if (ids.action.value && row.suggested_action !== ids.action.value) return false;
        if (ids.risk.value && row.risk_level_top1 !== ids.risk.value) return false;
        if (ids.onlyReview.checked && !reviewActions.has(row.suggested_action)) return false;
        return true;
      }).sort((a, b) => Number(b.policy_profit_delta_1d || 0) - Number(a.policy_profit_delta_1d || 0));
      state.limit = 200;
      render();
    }
    function aggregate(key, nameKey, data, limit = 10) {
      const map = new Map();
      data.forEach(row => {
        const value = row[key] || "";
        if (!value) return;
        if (!map.has(value)) map.set(value, {id: value, name: nameKey ? (row[nameKey] || value) : value, count: 0});
        map.get(value).count += 1;
      });
      return [...map.values()].sort((a, b) => b.count - a.count).slice(0, limit);
    }
    function renderBars(el, data) {
      const max = Math.max(1, ...data.map(item => item.count));
      el.innerHTML = data.map(item => `
        <div class="bar-row">
          <div class="bar-label" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, item.count / max * 100)}%"></div></div>
          <div class="bar-value">${fmtInt(item.count)}</div>
        </div>
      `).join("");
    }
    function renderSummary(data) {
      const currentProfit = data.reduce((sum, row) => sum + Number(row.current_profit_1d || 0), 0);
      const policyProfit = data.reduce((sum, row) => sum + Number(row.policy_profit_1d || 0), 0);
      const policyDelta = data.reduce((sum, row) => sum + Number(row.policy_profit_delta_1d || 0), 0);
      const avgUnit = data.length ? data.reduce((sum, row) => sum + Number(row.est_profit_per_mbps_top1 || 0), 0) / data.length : 0;
      const review = data.filter(row => reviewActions.has(row.suggested_action)).length;
      const highRisk = data.filter(row => row.risk_level_top1 === "high").length;
      const cards = [
        ["节点数", fmtInt(data.length), "当前筛选"],
        ["待处理", fmtInt(review), "建议人工复核"],
        ["高风险推荐", fmtInt(highRisk), "Top1 风险"],
        ["当前日利润", fmtMoney(currentProfit), "现网最近 1 天"],
        ["动作后利润", fmtMoney(policyProfit), "保留/复核切换后"],
        ["预计净提升", fmtMoney(policyDelta), "动作后对比当前"],
        ["利润/Mbps", fmtRate(avgUnit), "Top1 平均单位利润"]
      ];
      document.getElementById("summaryGrid").innerHTML = cards.map(card => `
        <div class="metric">
          <div class="label">${card[0]}</div>
          <div class="value">${card[1]}</div>
          <div class="note">${card[2]}</div>
        </div>
      `).join("");
    }
    function renderTop3(row) {
      return [1, 2, 3].map(rank => {
        const name = row[`business_name_top${rank}`] || "";
        const profit = fmtMoney(row[`est_profit_1d_top${rank}`]);
        const risk = row[`risk_level_top${rank}`] || "unknown";
        return `
          <div class="top3-line">
            <span class="rank">${rank}</span>
            <span class="pill risk-${risk}" title="${escapeHtml(row[`business_top${rank}`] || "")}">${escapeHtml(name)}</span>
            <span class="muted">${profit}</span>
          </div>
        `;
      }).join("");
    }
    function renderTable(data) {
      const shown = data.slice(0, state.limit);
      document.getElementById("tableBody").innerHTML = shown.map(row => `
        <tr>
          <td><div class="node">${escapeHtml(row.node_id)}</div></td>
          <td>
            <div>${escapeHtml(row.province)} ${escapeHtml(row.city)} ${escapeHtml(row.isp)}</div>
            <div class="muted">${escapeHtml(row.resourcetype)} · ${escapeHtml(row.bw_bucket)} · ${fmtMoney(row.effective_bandwidth_mbps)} Mbps</div>
            <div class="muted">网络调度 ${escapeHtml(row.network_schedule_type || "未知")} · 目标运营商 ${escapeHtml(row.scheduleisps_text || row.scheduleisps || row.isp || "未知")} · 跨省 ${fmtRate(row.analysis_transprovrate)}% · 禁跨省 ${escapeHtml(row.join_isbantransprov || "否/未知")}</div>
          </td>
          <td>
            <div class="business-cell">
              <span>${escapeHtml(row.current_business_name || "未识别当前业务")}</span>
              <span class="business-id">${escapeHtml(row.current_business || "")}</span>
            </div>
          </td>
          <td class="num">${fmtMoney(row.current_revenue_1d)}</td>
          <td class="num ${moneyClass(row.current_profit_1d)}">${fmtMoney(row.current_profit_1d)}</td>
          <td><div class="top3">${renderTop3(row)}</div></td>
          <td class="num">${fmtMoney(row.est_revenue_1d_top1)}</td>
          <td class="num">${fmtMoney(row.est_cost_1d_top1)}</td>
          <td class="num ${moneyClass(row.est_profit_1d_top1)}">${fmtMoney(row.est_profit_1d_top1)}</td>
          <td class="num ${moneyClass(row.policy_profit_1d)}">${fmtMoney(row.policy_profit_1d)}</td>
          <td class="num ${moneyClass(row.est_profit_per_mbps_top1)}">${fmtRate(row.est_profit_per_mbps_top1)}</td>
          <td class="num ${moneyClass(row.policy_profit_delta_1d)}">${fmtMoney(row.policy_profit_delta_1d)}</td>
          <td>
            <div>${escapeHtml(row.estimate_source_top1)}</div>
            <div class="muted">${fmtInt(row.estimate_support_groups_top1)} 样本 · ${fmtInt(row.estimate_support_nodes_top1)} 节点</div>
          </td>
          <td><span class="pill risk-${row.risk_level_top1 || "unknown"}">${riskLabels[row.risk_level_top1] || row.risk_level_top1 || "未知"}</span></td>
          <td><span class="pill action-${row.suggested_action}">${actionLabels[row.suggested_action] || row.suggested_action}</span></td>
        </tr>
      `).join("");
      document.getElementById("tableMeta").textContent = `${fmtInt(data.length)} 条`;
      document.getElementById("pageMeta").textContent = `已显示 ${fmtInt(shown.length)} / ${fmtInt(data.length)}`;
      document.getElementById("moreBtn").style.display = shown.length < data.length ? "inline-flex" : "none";
    }
    function render() {
      const data = state.filtered;
      renderSummary(data);
      const businessDist = aggregate("business_top1", "business_name_top1", data);
      const actionDist = aggregate("suggested_action", null, data).map(item => ({...item, name: actionLabels[item.id] || item.id}));
      renderBars(document.getElementById("businessBars"), businessDist);
      renderBars(document.getElementById("actionBars"), actionDist);
      document.getElementById("topBusinessMeta").textContent = `Top ${businessDist.length}`;
      document.getElementById("actionMeta").textContent = `Top ${actionDist.length}`;
      renderTable(data);
    }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }
    function exportCsv() {
      const columns = [
        "node_id","province","city","isp","resourcetype","effective_bandwidth_mbps",
        "network_schedule_type","schedule_isp_scope","schedule_province_scope","scheduleisps_text","scheduleisps",
        "analysis_transprovrate","analysis_transprovrate_bucket","join_isbantransprov","join_isipv6schedule",
        "current_business","current_business_name","current_revenue_1d","current_cost_1d","current_profit_1d",
        "business_top1","business_name_top1","est_revenue_1d_top1","est_cost_1d_top1","est_profit_1d_top1","est_profit_per_mbps_top1","est_profit_delta_top1",
        "policy_choice","policy_revenue_1d","policy_cost_1d","policy_profit_1d","policy_profit_delta_1d","risk_level_top1","suggested_action",
        "business_top2","business_name_top2","est_profit_1d_top2","business_top3","business_name_top3","est_profit_1d_top3"
      ];
      const lines = [columns.join(",")];
      state.filtered.forEach(row => {
        lines.push(columns.map(column => `"${String(row[column] ?? "").replace(/"/g, '""')}"`).join(","));
      });
      const blob = new Blob(["\ufeff" + lines.join("\n")], {type: "text/csv;charset=utf-8"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "business_recommendation_profit_filtered.csv";
      a.click();
      URL.revokeObjectURL(url);
    }
    function resetFilters() {
      ids.search.value = "";
      ids.province.value = "";
      ids.isp.value = "";
      ids.business.value = "";
      ids.schedule.value = "";
      ids.action.value = "";
      ids.risk.value = "";
      ids.onlyReview.checked = false;
      applyFilters();
    }
    document.getElementById("subtitle").textContent = `${payload.meta.model || ""} · 训练 ${payload.meta.training_window || ""} · 当前 ${payload.meta.current_window?.start || ""}`;
    document.getElementById("caveat").textContent = payload.meta.caveat || "";
    document.getElementById("moreBtn").addEventListener("click", () => { state.limit += 200; render(); });
    document.getElementById("exportBtn").addEventListener("click", exportCsv);
    document.getElementById("resetBtn").addEventListener("click", resetFilters);
    initFilters();
    applyFilters();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
