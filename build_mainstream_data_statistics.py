#!/usr/bin/env python3
"""Build complete statistics for large-node mainstream business analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v1_recommendation_pipeline as v1


HERE = Path(__file__).resolve().parent
DEFAULT_ALLOWLIST = HERE / "mainstream_business_allowlist.csv"
DEFAULT_OUTCOMES = HERE / "recent_month_large_mainstream_1d/multibusiness_outcomes_large_mainstream_recent_1m.csv"
DEFAULT_ALL_OUTCOMES = HERE / "recent_month_large_1d/multibusiness_outcomes_large_recent_1m.csv"
DEFAULT_CURRENT_SCAN_DIR = HERE / "current_non_idc_large_scan"
DEFAULT_TRAINING_SUMMARY = HERE / "recent_month_large_mainstream_1d/mainstream_large_training_summary.json"
DEFAULT_OUTPUT_DIR = HERE / "mainstream_data_statistics"


def clean(value: Any) -> str:
    return v1.clean_cell(value)


def latest_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise RuntimeError(f"no files match {directory / pattern}")
    return matches[-1]


def bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return frame[column].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def to_number(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    return output


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", low_memory=False).fillna("")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def date_token(frame: pd.DataFrame, preferred_column: str) -> str:
    if preferred_column not in frame.columns or frame.empty:
        return "unknown"
    parsed = pd.to_datetime(frame[preferred_column], errors="coerce")
    if not parsed.notna().any():
        return "unknown"
    return f"{parsed.min().strftime('%Y%m%d')}_{parsed.max().strftime('%Y%m%d')}"


def training_window_text(outcomes: pd.DataFrame, training_summary: dict[str, Any]) -> str:
    outcome_window_days = clean(training_summary.get("outcome_window_days"))
    outcome_grain = clean(training_summary.get("outcome_grain"))
    sample_day_min = clean(training_summary.get("sample_day_min"))
    sample_day_max = clean(training_summary.get("sample_day_max"))
    if outcome_window_days == "1" or outcome_grain == "daily_node_business":
        if not sample_day_min or not sample_day_max:
            parsed = pd.to_datetime(outcomes.get("sample_day", pd.Series(dtype="string")), errors="coerce")
            if parsed.notna().any():
                sample_day_min = parsed.min().strftime("%Y-%m-%d")
                sample_day_max = parsed.max().strftime("%Y-%m-%d")
        return f"sample_day {sample_day_min}..{sample_day_max}, 1d daily node-business outcome"
    start = clean(training_summary.get("business_online_day_start"))
    end = clean(training_summary.get("business_online_day_end"))
    wide_end = clean(training_summary.get("wide_outcome_end_day"))
    if start and end and wide_end:
        return f"business_online_day {start}..{end}, {outcome_window_days or '7'}d outcome through {wide_end}"
    return "training outcome window unknown"


def load_allowlist(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    required = {"business", "business_name", "brand"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"allowlist missing columns: {sorted(missing)}")
    frame["business"] = frame["business"].map(clean)
    frame["business_name"] = frame["business_name"].map(clean)
    frame["brand"] = frame["brand"].map(clean)
    duplicated = frame[frame["business"].duplicated()]["business"].tolist()
    if duplicated:
        raise RuntimeError(f"duplicated business ids in allowlist: {duplicated}")
    return frame


def attach_allowlist(frame: pd.DataFrame, allowlist: pd.DataFrame, business_column: str) -> pd.DataFrame:
    names = allowlist[["business", "business_name", "brand"]].rename(
        columns={
            "business": business_column,
            "business_name": f"{business_column}_allowlist_name",
            "brand": f"{business_column}_brand",
        }
    )
    output = frame.copy()
    output[business_column] = output[business_column].map(clean)
    return output.merge(names, on=business_column, how="left", sort=False)


def prepare_outcomes(outcomes: pd.DataFrame, allowlist: pd.DataFrame) -> pd.DataFrame:
    frame = outcomes.copy()
    frame["business"] = frame["business"].map(clean)
    frame = attach_allowlist(frame, allowlist, "business")
    frame["business_name"] = frame["business_allowlist_name"].fillna("").where(
        frame["business_allowlist_name"].fillna("").ne(""),
        frame.get("business_name", ""),
    )
    frame["brand"] = frame["business_brand"].fillna("")
    numeric = [
        "business_active_days",
        "business_count",
        "business_first_day_count",
        "cum_cost_7d",
        "cum_revenue_7d",
        "outcome_days",
        "outcome_distinct_days",
    ]
    frame = to_number(frame, numeric)
    frame["cum_profit_7d"] = frame["cum_revenue_7d"] - frame["cum_cost_7d"]
    group_fields = v1.outcome_comparison_fields(frame)
    frame["miner_score_norm"] = frame.groupby(group_fields, dropna=False)["cum_cost_7d"].transform(v1.normalize_within_node)
    frame["operator_score_norm"] = frame.groupby(group_fields, dropna=False)["cum_profit_7d"].transform(v1.normalize_within_node)
    frame["combined_score"] = 0.5 * frame["miner_score_norm"] + 0.5 * frame["operator_score_norm"]
    frame["is_combined_best"] = False
    if not frame.empty:
        best_index = frame.groupby(group_fields, dropna=False)["combined_score"].idxmax()
        frame.loc[best_index, "is_combined_best"] = True
    return frame


def money_agg(
    frame: pd.DataFrame,
    group_columns: list[str],
    total_mainstream_nodes: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            row_count=("node_id", "size"),
            node_count=("node_id", "nunique"),
            best_node_count=("is_combined_best", "sum"),
            total_cost_7d=("cum_cost_7d", "sum"),
            total_revenue_7d=("cum_revenue_7d", "sum"),
            total_profit_7d=("cum_profit_7d", "sum"),
            avg_cost_7d=("cum_cost_7d", "mean"),
            avg_revenue_7d=("cum_revenue_7d", "mean"),
            avg_profit_7d=("cum_profit_7d", "mean"),
            median_profit_7d=("cum_profit_7d", "median"),
            avg_combined_score=("combined_score", "mean"),
            avg_miner_score_norm=("miner_score_norm", "mean"),
            avg_operator_score_norm=("operator_score_norm", "mean"),
            avg_outcome_distinct_days=("outcome_distinct_days", "mean"),
            avg_business_active_days=("business_active_days", "mean"),
        )
        .reset_index()
    )
    grouped["profit_margin"] = np.where(
        grouped["total_revenue_7d"].abs() > 1e-9,
        grouped["total_profit_7d"] / grouped["total_revenue_7d"],
        0.0,
    )
    grouped["best_rate_in_observed_nodes"] = grouped["best_node_count"] / grouped["node_count"].clip(lower=1)
    grouped["best_share_all_mainstream_nodes"] = grouped["best_node_count"] / max(total_mainstream_nodes, 1)
    return grouped.sort_values(
        ["best_node_count", "node_count", "avg_combined_score"],
        ascending=[False, False, False],
    )


def prepare_current(recommendations: pd.DataFrame, allowlist: pd.DataFrame) -> pd.DataFrame:
    frame = recommendations.copy()
    allow_ids = set(allowlist["business"])
    for column in ["current_business", "recommended_business", "v2_business_top1", "v2_business_top2", "v2_business_top3"]:
        if column in frame.columns:
            frame[column] = frame[column].map(clean)
    frame = attach_allowlist(frame, allowlist, "current_business")
    frame["current_business_name"] = frame["current_business_allowlist_name"].fillna("").where(
        frame["current_business_allowlist_name"].fillna("").ne(""),
        frame.get("current_business_name", ""),
    )
    frame["current_business_brand"] = frame["current_business_brand"].fillna("")
    frame["current_business_is_mainstream_calc"] = frame["current_business"].isin(allow_ids)
    if "v2_business_top1" in frame.columns:
        frame["recommended_business"] = frame["v2_business_top1"].map(clean)
    if "v2_business_name_top1" in frame.columns:
        frame["recommended_business_name"] = frame["v2_business_name_top1"].map(clean)
    frame = attach_allowlist(frame, allowlist, "recommended_business")
    frame["recommended_business_name"] = frame["recommended_business_allowlist_name"].fillna("").where(
        frame["recommended_business_allowlist_name"].fillna("").ne(""),
        frame.get("recommended_business_name", ""),
    )
    frame["recommended_business_brand"] = frame["recommended_business_brand"].fillna("")
    top_columns = [column for column in ["v2_business_top1", "v2_business_top2", "v2_business_top3"] if column in frame.columns]
    frame["current_eq_top1_calc"] = frame["current_business"].ne("") & frame["current_business"].eq(frame["v2_business_top1"])
    frame["current_in_top3_calc"] = False
    for column in top_columns:
        frame["current_in_top3_calc"] = frame["current_in_top3_calc"] | (
            frame["current_business"].ne("") & frame["current_business"].eq(frame[column])
        )
    numeric = [
        "v2_rank_score_top1",
        "current_business_active_days",
        "current_business_peak95_sum",
        "current_business_peak95_max",
        "current_business_revenue_sum",
        "current_business_cost_sum",
        "current_business_profit_sum",
        "bandwidth",
        "actualbandwidth",
        "bw",
        "corenum",
    ]
    return to_number(frame, numeric)


def current_business_stats(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            node_count=("node_id", "nunique"),
            keep_top1_nodes=("current_eq_top1_calc", "sum"),
            in_top3_nodes=("current_in_top3_calc", "sum"),
            not_top1_nodes=("current_eq_top1_calc", lambda value: int((~value).sum())),
            not_in_top3_nodes=("current_in_top3_calc", lambda value: int((~value).sum())),
            total_revenue_7d=("current_business_revenue_sum", "sum"),
            total_cost_7d=("current_business_cost_sum", "sum"),
            total_profit_7d=("current_business_profit_sum", "sum"),
            avg_revenue_7d=("current_business_revenue_sum", "mean"),
            avg_cost_7d=("current_business_cost_sum", "mean"),
            avg_profit_7d=("current_business_profit_sum", "mean"),
            avg_peak95_max=("current_business_peak95_max", "mean"),
            avg_rank_score_top1=("v2_rank_score_top1", "mean"),
            avg_bandwidth=("bandwidth", "mean"),
            avg_actualbandwidth=("actualbandwidth", "mean"),
        )
        .reset_index()
    )
    grouped["profit_margin"] = np.where(
        grouped["total_revenue_7d"].abs() > 1e-9,
        grouped["total_profit_7d"] / grouped["total_revenue_7d"],
        0.0,
    )
    grouped["keep_top1_rate"] = grouped["keep_top1_nodes"] / grouped["node_count"].clip(lower=1)
    grouped["in_top3_rate"] = grouped["in_top3_nodes"] / grouped["node_count"].clip(lower=1)
    return grouped.sort_values(["node_count", "total_profit_7d"], ascending=[False, False])


def recommendation_stats(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            node_count=("node_id", "nunique"),
            avg_rank_score_top1=("v2_rank_score_top1", "mean"),
            avg_bandwidth=("bandwidth", "mean"),
            avg_actualbandwidth=("actualbandwidth", "mean"),
        )
        .reset_index()
        .sort_values(["node_count", "avg_rank_score_top1"], ascending=[False, False])
    )
    return grouped


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False)
    return str(path)


def write_report(path: Path, summary: dict[str, Any], files: dict[str, str]) -> None:
    current = summary["current_mainstream_scope"]
    training = summary["training_scope"]
    window = summary["current_scope"].get("current_business_window", {})
    lines = [
        "# 主流业务大节点完整数据统计",
        "",
        "## 口径",
        "",
        "- 只统计 `mainstream_business_allowlist.csv` 内业务。",
        "- 非主流当前业务不进入业务分布、收益、切换对比统计，仅作为排除量记录。",
        (
            f"- 训练收益口径：{training.get('window', '')}；现网使用 "
            f"{summary['current_scope']['snapshot_day']} 节点快照和 "
            f"{window.get('start', '')} 至 {window.get('end', '')} 当前业务窗口。"
        ),
        "- 训练文件为兼容旧模型仍保留 `cum_*_7d` 列名；当 `outcome_window_days=1` 时这些列代表单日金额。",
        "- 综合收益同模型口径：单节点内矿主成本归一化 50% + 平台利润归一化 50%。",
        "- 存量纠偏使用当前真实收益保护：当前业务利润和单位利润已优于推荐目标业务现网中位值时，不建议切换。",
        "",
        "## 总览",
        "",
        f"- 主流业务池：{summary['allowlist_businesses']} 个。",
        f"- 近一月大节点全业务收益样本：{training['all_outcome_rows']} 条，主流业务收益样本：{training['mainstream_outcome_rows']} 条。",
        f"- 主流收益样本覆盖节点：{training['mainstream_nodes']} 个，实际出现主流业务：{training['observed_mainstream_businesses']} 个。",
        f"- 现网扫描大节点：{summary['current_scope']['scanned_nodes']} 个。",
        f"- 现网当前主流业务节点：{current['current_mainstream_nodes']} 个。",
        f"- 非主流当前业务节点：{summary['current_scope']['excluded_non_mainstream_current_nodes']} 个，仅排除统计。",
        f"- 当前业务未知节点：{summary['current_scope']['unknown_current_business_nodes']} 个，仅排除统计。",
        f"- 模型判定当前主流业务不是 Top1 的节点：{current['model_not_top1_nodes']} 个。",
        f"- 策略后当前主流业务仍需处理节点：{current['policy_not_best_nodes']} 个。",
        f"- 当前真实收益保护节点：{current['keep_current_realized_value_nodes']} 个。",
        f"- 当前盈利但推荐高风险观察节点：{current['observe_current_profitable_high_risk_nodes']} 个。",
        "",
        "## 核心文件",
        "",
    ]
    for name, value in files.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build statistics for mainstream large-node business data.")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--all-outcomes", type=Path, default=DEFAULT_ALL_OUTCOMES)
    parser.add_argument("--current-recommendations", type=Path)
    parser.add_argument("--current-summary", type=Path)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_recommendations_path = args.current_recommendations or latest_file(
        DEFAULT_CURRENT_SCAN_DIR,
        "current_non_idc_large_recommendations_*.csv",
    )
    current_summary_path = args.current_summary or latest_file(
        DEFAULT_CURRENT_SCAN_DIR,
        "current_non_idc_large_scan_summary_*.json",
    )

    allowlist = load_allowlist(args.allowlist)
    outcomes = prepare_outcomes(read_csv(args.outcomes), allowlist)
    all_outcomes = read_csv(args.all_outcomes)
    current = prepare_current(read_csv(current_recommendations_path), allowlist)
    current_summary = read_json(current_summary_path)
    training_summary = read_json(args.training_summary)
    snapshot_day = current_summary.get("partitions", {}).get("analysis_day", "unknown")
    window = current_summary.get("current_business_window", {})
    current_start = str(window.get("start", "")).replace("-", "") or "unknown"
    current_end = str(window.get("end", "")).replace("-", "") or "unknown"
    training_token = date_token(outcomes, "sample_day")
    if training_token == "unknown":
        training_token = date_token(outcomes, "business_online_day")
    training_window = training_window_text(outcomes, training_summary)

    allow_ids = set(allowlist["business"])
    current_with_business = current[current["current_business"].ne("")].copy()
    current_mainstream = current_with_business[current_with_business["current_business"].isin(allow_ids)].copy()
    current_non_mainstream = current_with_business[~current_with_business["current_business"].isin(allow_ids)].copy()
    current_unknown = current[current["current_business"].eq("")].copy()
    total_mainstream_training_nodes = int(outcomes["node_id"].nunique())
    current_mainstream_policy_not_best = bool_series(current_mainstream, "current_not_best")
    current_mainstream_model_not_top1 = bool_series(current_mainstream, "model_current_not_top1")

    files: dict[str, str] = {}
    files["allowlist_brand_stats"] = write_csv(
        allowlist.groupby("brand", dropna=False)
        .agg(allowlist_businesses=("business", "nunique"))
        .reset_index()
        .sort_values("allowlist_businesses", ascending=False),
        args.output_dir / "allowlist_brand_stats.csv",
    )
    files["training_business_stats"] = write_csv(
        money_agg(outcomes, ["business", "business_name", "brand"], total_mainstream_training_nodes),
        args.output_dir / f"training_mainstream_business_stats_{training_token}.csv",
    )
    files["training_brand_stats"] = write_csv(
        money_agg(outcomes, ["brand"], total_mainstream_training_nodes),
        args.output_dir / f"training_mainstream_brand_stats_{training_token}.csv",
    )
    training_condition_fields = [
        field
        for field in ["province", "isp", "resourcetype", "bw_bucket", "business", "business_name", "brand"]
        if field in outcomes.columns
    ]
    files["training_condition_business_stats"] = write_csv(
        money_agg(outcomes, training_condition_fields, total_mainstream_training_nodes),
        args.output_dir / f"training_mainstream_condition_business_stats_{training_token}.csv",
    )
    files["current_business_stats"] = write_csv(
        current_business_stats(current_mainstream, ["current_business", "current_business_name", "current_business_brand"]),
        args.output_dir / f"current_mainstream_business_stats_{current_start}_{current_end}.csv",
    )
    files["current_brand_stats"] = write_csv(
        current_business_stats(current_mainstream, ["current_business_brand"]),
        args.output_dir / f"current_mainstream_brand_stats_{current_start}_{current_end}.csv",
    )
    switch_frame = current_mainstream[
        current_mainstream["current_not_best"].map(lambda value: str(value).strip().lower() == "true")
    ].copy()
    files["current_switch_pair_stats"] = write_csv(
        current_business_stats(
            switch_frame,
            [
                "suggested_action",
                "current_business",
                "current_business_name",
                "recommended_business",
                "recommended_business_name",
                "recommended_business_brand",
                "v2_risk_level_top1",
            ],
        ),
        args.output_dir / f"current_mainstream_switch_pair_stats_{snapshot_day}.csv",
    )
    files["current_action_stats"] = write_csv(
        current_mainstream.groupby("suggested_action", dropna=False)
        .agg(node_count=("node_id", "nunique"))
        .reset_index()
        .sort_values("node_count", ascending=False),
        args.output_dir / f"current_mainstream_action_stats_{snapshot_day}.csv",
    )
    files["recommended_top1_all_scanned"] = write_csv(
        recommendation_stats(
            current,
            ["recommended_business", "recommended_business_name", "recommended_business_brand", "v2_risk_level_top1"],
        ),
        args.output_dir / f"recommended_top1_all_scanned_large_nodes_{snapshot_day}.csv",
    )
    files["recommended_top1_current_mainstream"] = write_csv(
        recommendation_stats(
            current_mainstream,
            ["recommended_business", "recommended_business_name", "recommended_business_brand", "v2_risk_level_top1"],
        ),
        args.output_dir / f"recommended_top1_current_mainstream_nodes_{snapshot_day}.csv",
    )
    condition_fields = [
        field
        for field in [
            "province",
            "isp",
            "resourcetype",
            "bw_bucket",
            "recommended_business",
            "recommended_business_name",
            "recommended_business_brand",
            "v2_risk_level_top1",
        ]
        if field in current.columns
    ]
    files["condition_recommendation_stats"] = write_csv(
        recommendation_stats(current, condition_fields),
        args.output_dir / f"condition_recommendation_stats_all_scanned_large_nodes_{snapshot_day}.csv",
    )
    files["excluded_non_mainstream_current_business_stats"] = write_csv(
        current_non_mainstream.groupby(["current_business", "current_business_name"], dropna=False)
        .agg(node_count=("node_id", "nunique"))
        .reset_index()
        .sort_values("node_count", ascending=False),
        args.output_dir / f"excluded_non_mainstream_current_business_stats_{snapshot_day}.csv",
    )
    files["excluded_unknown_current_business_nodes"] = write_csv(
        current_unknown[["node_id", "province", "city", "isp", "resourcetype", "bw_bucket"]].copy(),
        args.output_dir / f"excluded_unknown_current_business_nodes_{snapshot_day}.csv",
    )
    missing_training = allowlist[~allowlist["business"].isin(set(outcomes["business"]))].copy()
    files["allowlist_missing_training_support"] = write_csv(
        missing_training,
        args.output_dir / f"allowlist_businesses_missing_training_support_{training_token}.csv",
    )

    validation = {
        "current_stats_non_mainstream_leak_rows": int((~current_mainstream["current_business"].isin(allow_ids)).sum()),
        "training_stats_non_mainstream_leak_rows": int((~outcomes["business"].isin(allow_ids)).sum()),
        "recommended_outside_allowlist": sorted(
            set(current["recommended_business"].map(clean)) - allow_ids - {""}
        ),
    }
    if any(value for key, value in validation.items() if key != "recommended_outside_allowlist") or validation[
        "recommended_outside_allowlist"
    ]:
        raise RuntimeError(f"statistics validation failed: {validation}")

    summary = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "allowlist_businesses": int(len(allowlist)),
        "allowlist_brands": allowlist["brand"].value_counts().to_dict(),
        "training_scope": {
            "window": training_window,
            "outcome_window_days": clean(training_summary.get("outcome_window_days")),
            "outcome_grain": clean(training_summary.get("outcome_grain")),
            "all_outcome_rows": int(len(all_outcomes)),
            "all_outcome_businesses": int(all_outcomes["business"].map(clean).nunique()) if "business" in all_outcomes else 0,
            "mainstream_outcome_rows": int(len(outcomes)),
            "mainstream_nodes": total_mainstream_training_nodes,
            "observed_mainstream_businesses": int(outcomes["business"].nunique()),
            "missing_allowlist_businesses": int(len(missing_training)),
            "excluded_non_mainstream_rows": int(len(all_outcomes) - len(outcomes)),
            "model_summary": training_summary.get("v2_test", {}),
        },
        "current_scope": {
            "snapshot_day": snapshot_day,
            "current_business_window": window,
            "scanned_nodes": int(len(current)),
            "current_business_nodes": int(len(current_with_business)),
            "excluded_non_mainstream_current_nodes": int(len(current_non_mainstream)),
            "unknown_current_business_nodes": int(len(current_unknown)),
        },
        "current_mainstream_scope": {
            "current_mainstream_nodes": int(len(current_mainstream)),
            "keep_top1_nodes": int(current_mainstream["current_eq_top1_calc"].sum()),
            "in_top3_nodes": int(current_mainstream["current_in_top3_calc"].sum()),
            "not_top1_nodes": int((~current_mainstream["current_eq_top1_calc"]).sum()),
            "not_in_top3_nodes": int((~current_mainstream["current_in_top3_calc"]).sum()),
            "model_not_top1_nodes": int(current_mainstream_model_not_top1.sum()),
            "policy_not_best_nodes": int(current_mainstream_policy_not_best.sum()),
            "keep_current_realized_value_nodes": int(
                current_mainstream["suggested_action"].eq("keep_current_realized_value").sum()
            ),
            "observe_current_profitable_high_risk_nodes": int(
                current_mainstream["suggested_action"].eq("observe_current_profitable_high_risk").sum()
            ),
        },
        "validation": validation,
        "files": files,
    }
    summary_path = args.output_dir / f"mainstream_data_statistics_summary_{snapshot_day}.json"
    report_path = args.output_dir / f"mainstream_data_statistics_report_{snapshot_day}.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(report_path, summary, files)
    files["summary_json"] = str(summary_path)
    files["report_md"] = str(report_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
