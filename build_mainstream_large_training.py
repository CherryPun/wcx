#!/usr/bin/env python3
"""Build large-node artifacts with a mainstream business allowlist only."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import v2_ranking_model as v2


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = HERE / "recent_month_large"
DEFAULT_OUTPUT_DIR = HERE / "recent_month_large_mainstream"
DEFAULT_ALLOWLIST = HERE / "mainstream_business_allowlist.csv"
DEFAULT_BUSINESS_MAP = HERE / "v1_business_name_map_enriched.csv"
UNIT_BANDWIDTH_RANK_SCORE_WEIGHTS = "source_rate=0.2,champion=0.1,expected_best_rate=0.2,expected_score=0.5"


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return "" if text in {"nan", "NaN", "None", "<NA>", "\\N", "\\\\N"} else text


def run_command(command: list[str], cwd: Path = HERE) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def write_v2_recommendations_for_all_large_nodes(
    nodes_path: Path,
    model_path: Path,
    output_path: Path,
) -> None:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    nodes = v2.load_nodes(nodes_path)
    if "node_size_type" in nodes.columns:
        nodes = nodes[nodes["node_size_type"].map(v2.clean_cell) == v2.NODE_SIZE_LARGE].copy()
    recommendations = v2.recommendations_for_nodes(nodes, model, top_k=3)
    recommendations.to_csv(output_path, index=False)
    print(json.dumps({"output": str(output_path), "rows": int(len(recommendations))}, ensure_ascii=False, indent=2))


def load_allowlist(path: Path) -> pd.DataFrame:
    allowlist = pd.read_csv(path, dtype=str).fillna("")
    required = {"business", "business_name", "brand"}
    missing = required - set(allowlist.columns)
    if missing:
        raise RuntimeError(f"allowlist missing columns: {sorted(missing)}")
    allowlist["business"] = allowlist["business"].map(clean)
    allowlist["business_name"] = allowlist["business_name"].map(clean)
    if allowlist["business"].duplicated().any():
        duplicated = allowlist.loc[allowlist["business"].duplicated(), "business"].tolist()
        raise RuntimeError(f"duplicated business ids in allowlist: {duplicated}")
    if allowlist["business_name"].duplicated().any():
        duplicated = allowlist.loc[allowlist["business_name"].duplicated(), "business_name"].tolist()
        raise RuntimeError(f"duplicated business names in allowlist: {duplicated}")
    return allowlist


def apply_allowlist_names(frame: pd.DataFrame, allowlist: pd.DataFrame) -> pd.DataFrame:
    names = dict(zip(allowlist["business"], allowlist["business_name"]))
    output = frame.copy()
    output["business"] = output["business"].map(clean)
    output["business_name"] = output["business"].map(lambda value: names.get(value, ""))
    output["business_scope"] = "mainstream"
    return output


def write_filtered_sources(
    source_dir: Path,
    output_dir: Path,
    allowlist: pd.DataFrame,
) -> dict[str, Any]:
    nodes_path = source_dir / "multibusiness_nodes_large_recent_1m.csv"
    outcomes_path = source_dir / "multibusiness_outcomes_large_recent_1m.csv"
    if not nodes_path.exists() or not outcomes_path.exists():
        raise RuntimeError(
            "recent large-node source files are missing; run build_recent_large_training.py first"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(nodes_path, dtype={"node_id": "string"}, low_memory=False)
    outcomes = pd.read_csv(outcomes_path, dtype={"node_id": "string", "business": "string"}, low_memory=False)
    outcomes["business"] = outcomes["business"].map(clean)
    allow_ids = set(allowlist["business"])
    filtered_outcomes = apply_allowlist_names(outcomes[outcomes["business"].isin(allow_ids)], allowlist)
    mainstream_node_ids = set(filtered_outcomes["node_id"].astype(str))

    nodes_output = output_dir / "multibusiness_nodes_large_mainstream_recent_1m.csv"
    outcomes_output = output_dir / "multibusiness_outcomes_large_mainstream_recent_1m.csv"
    training_nodes_output = output_dir / "multibusiness_nodes_large_mainstream_training_recent_1m.csv"
    nodes.to_csv(nodes_output, index=False)
    nodes[nodes["node_id"].astype(str).isin(mainstream_node_ids)].to_csv(training_nodes_output, index=False)
    filtered_outcomes.to_csv(outcomes_output, index=False)

    observed = set(filtered_outcomes["business"])
    missing_observed = allowlist[~allowlist["business"].isin(observed)].copy()
    excluded = outcomes[~outcomes["business"].isin(allow_ids)].copy()
    support = (
        filtered_outcomes.groupby("business", dropna=False)
        .agg(rows=("node_id", "size"), nodes=("node_id", "nunique"))
        .reset_index()
        .merge(allowlist[["business", "business_name", "brand"]], on="business", how="left")
        .sort_values(["rows", "nodes", "business"], ascending=[False, False, True])
    )
    support_path = output_dir / "mainstream_business_support_recent_1m.csv"
    support.to_csv(support_path, index=False)

    outcome_window_days = ""
    if "outcome_window_days" in filtered_outcomes.columns and not filtered_outcomes.empty:
        outcome_window_days = clean(filtered_outcomes["outcome_window_days"].iloc[0])
    outcome_grain = ""
    if "outcome_grain" in filtered_outcomes.columns and not filtered_outcomes.empty:
        outcome_grain = clean(filtered_outcomes["outcome_grain"].iloc[0])
    sample_day_min = ""
    sample_day_max = ""
    if "sample_day" in filtered_outcomes.columns and not filtered_outcomes.empty:
        parsed_sample_day = pd.to_datetime(filtered_outcomes["sample_day"], errors="coerce")
        if parsed_sample_day.notna().any():
            sample_day_min = parsed_sample_day.min().strftime("%Y-%m-%d")
            sample_day_max = parsed_sample_day.max().strftime("%Y-%m-%d")

    return {
        "nodes_path": nodes_output,
        "training_nodes_path": training_nodes_output,
        "outcomes_path": outcomes_output,
        "support_path": support_path,
        "source_nodes": int(len(nodes)),
        "source_outcome_rows": int(len(outcomes)),
        "source_businesses": int(outcomes["business"].nunique()),
        "allowlist_businesses": int(len(allowlist)),
        "observed_allowlist_businesses": int(filtered_outcomes["business"].nunique()),
        "missing_allowlist_businesses": missing_observed[["business", "business_name", "brand"]].to_dict(orient="records"),
        "filtered_nodes_with_mainstream_history": int(len(mainstream_node_ids)),
        "filtered_outcome_rows": int(len(filtered_outcomes)),
        "excluded_non_mainstream_businesses": int(excluded["business"].nunique()),
        "excluded_non_mainstream_rows": int(len(excluded)),
        "outcome_window_days": outcome_window_days,
        "outcome_grain": outcome_grain,
        "sample_day_min": sample_day_min,
        "sample_day_max": sample_day_max,
    }


def build_models(
    output_dir: Path,
    nodes_path: Path,
    training_nodes_path: Path,
    outcomes_path: Path,
    business_map: Path,
    min_business_support: int,
    target_mode: str,
    rank_score_weights: str | None,
) -> dict[str, Path]:
    v1_dir = output_dir / "v1_large_mainstream_outputs"
    v2_dir = output_dir / "v2_large_mainstream_outputs"
    run_command([
        sys.executable,
        "v1_recommendation_pipeline.py",
        "build",
        "--nodes",
        str(training_nodes_path),
        "--outcomes",
        str(outcomes_path),
        "--business-map",
        str(business_map),
        "--output-dir",
        str(v1_dir),
        "--min-business-support",
        str(min_business_support),
        "--top-k",
        "3",
        "--target-mode",
        target_mode,
    ])

    training_pairs = v1_dir / "v1_training_pairs.csv"
    training_pairs_copy = output_dir / "v1_training_pairs_large_mainstream_recent_1m.csv"
    shutil.copyfile(training_pairs, training_pairs_copy)

    v2_command = [
        sys.executable,
        "v2_ranking_model.py",
        "build",
        "--pairs",
        str(training_pairs),
        "--business-map",
        str(business_map),
        "--risk-profile",
        str(v1_dir / "v1_business_risk_profile.csv"),
        "--output-dir",
        str(v2_dir),
        "--min-business-support",
        str(min_business_support),
        "--champion-min-support",
        "1",
        "--risk-min-profile-count",
        "3",
        "--risk-min-best-support",
        "1",
    ]
    if rank_score_weights:
        v2_command.extend(["--rank-score-weights", rank_score_weights])
    v2_command.extend(["--top-k", "3"])
    run_command(v2_command)

    all_recommendations = v2_dir / "v2_node_recommendations.csv"
    write_v2_recommendations_for_all_large_nodes(
        nodes_path,
        v2_dir / "v2_ranking_model.json",
        all_recommendations,
    )

    report_csv = output_dir / "v2_frontend_condition_business_report_large_mainstream_recent_1m.csv"
    report_json = output_dir / "v2_frontend_condition_business_report_large_mainstream_recent_1m.json"
    report_data_js = output_dir / "v2_frontend_condition_business_report_large_mainstream_recent_1m_data.js"
    run_command([
        sys.executable,
        "v2_ranking_model.py",
        "export-condition-report",
        "--nodes",
        str(nodes_path),
        "--model",
        str(v2_dir / "v2_ranking_model.json"),
        "--recommendations",
        str(all_recommendations),
        "--min-nodes",
        "1",
        "--output-csv",
        str(report_csv),
        "--output-json",
        str(report_json),
        "--output-data-js",
        str(report_data_js),
    ])

    html_src = HERE / "v2_frontend_condition_business_report_large.html"
    html_dst = output_dir / "v2_frontend_condition_business_report_large_mainstream_recent_1m.html"
    html = html_src.read_text(encoding="utf-8").replace(
        "v2_frontend_condition_business_report_large_data.js",
        report_data_js.name,
    )
    html_dst.write_text(html, encoding="utf-8")

    run_command([sys.executable, "v1_recommendation_pipeline.py", "check", "--output-dir", str(v1_dir)])
    run_command([sys.executable, "v2_ranking_model.py", "check", "--output-dir", str(v2_dir)])
    if shutil.which("node"):
        run_command(["node", "--check", str(report_data_js)])

    return {
        "v1_dir": v1_dir,
        "v2_dir": v2_dir,
        "training_pairs": training_pairs_copy,
        "recommendations": all_recommendations,
        "report_html": html_dst,
        "report_csv": report_csv,
        "report_json": report_json,
        "report_data_js": report_data_js,
    }


def verify_only_allowlist(path: Path, allowlist: pd.DataFrame, business_columns: list[str]) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    allow_ids = set(allowlist["business"])
    found: set[str] = set()
    violations: set[str] = set()
    for column in business_columns:
        if column not in frame.columns:
            continue
        values = {clean(value) for value in frame[column].tolist() if clean(value)}
        found |= values
        violations |= values - allow_ids
    return {
        "file": str(path),
        "rows": int(len(frame)),
        "businesses_found": len(found),
        "violations": sorted(violations),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train large-node model with mainstream businesses only.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--business-map", type=Path, default=DEFAULT_BUSINESS_MAP)
    parser.add_argument("--min-business-support", type=int, default=1)
    parser.add_argument(
        "--target-mode",
        choices=["absolute", "unit_bandwidth", "hybrid"],
        default="absolute",
        help="Training target passed to V1.",
    )
    parser.add_argument(
        "--rank-score-weights",
        help="Optional V2 ranking weights. Defaults to unit-bandwidth-friendly weights for --target-mode unit_bandwidth.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowlist = load_allowlist(args.allowlist)
    source_info = write_filtered_sources(args.source_dir, args.output_dir, allowlist)
    rank_score_weights = args.rank_score_weights
    if args.target_mode == "unit_bandwidth" and not rank_score_weights:
        rank_score_weights = UNIT_BANDWIDTH_RANK_SCORE_WEIGHTS
    artifacts = build_models(
        args.output_dir,
        source_info["nodes_path"],
        source_info["training_nodes_path"],
        source_info["outcomes_path"],
        args.business_map,
        args.min_business_support,
        args.target_mode,
        rank_score_weights,
    )
    checks = [
        verify_only_allowlist(
            source_info["outcomes_path"],
            allowlist,
            ["business"],
        ),
        verify_only_allowlist(
            artifacts["training_pairs"],
            allowlist,
            ["business"],
        ),
        verify_only_allowlist(
            artifacts["recommendations"],
            allowlist,
            ["v2_business_top1", "v2_business_top2", "v2_business_top3"],
        ),
        verify_only_allowlist(
            artifacts["report_csv"],
            allowlist,
            ["business_top1", "business_top2", "business_top3"],
        ),
    ]
    violations = [item for check in checks for item in check["violations"]]
    if violations:
        raise RuntimeError(f"non-mainstream business leaked into artifacts: {sorted(set(violations))}")

    metrics = json.loads((artifacts["v2_dir"] / "v2_model_metrics.json").read_text(encoding="utf-8"))
    summary = {
        **source_info,
        "min_business_support": args.min_business_support,
        "target_mode": args.target_mode,
        "rank_score_weights": rank_score_weights or "",
        "v2_test": metrics.get("test", {}),
        "checks": checks,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    summary_path = args.output_dir / "mainstream_large_training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
