#!/usr/bin/env python3
"""Build recent one-month large-node recommendation artifacts.

This script keeps the training set focused on 大节点:
1. discover recent large-node candidates from Jarvis node snapshots;
2. query recent node-business outcomes for those candidates;
3. fetch pre-business node attributes and re-apply the repo large-node heuristic;
4. build V1 training pairs, V2 ranking model, and frontend condition report.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

import rebuild_multibusiness_model as rebuild
import v2_ranking_model as v2


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "recent_month_large"
DEFAULT_BUSINESS_MAP = HERE / "v1_business_name_map_enriched.csv"


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def run_command(command: list[str], cwd: Path = HERE) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def query_one_value(sql: str, database_id: int, schema: str, key: str) -> str:
    rows = rebuild.run_sql(sql, database_id=database_id, schema=schema)
    if not rows or rows[0].get(key) in {None, ""}:
        raise RuntimeError(f"SQL did not return {key}: {sql}")
    return str(rows[0][key])


def discover_window(args: argparse.Namespace) -> tuple[dt.date, dt.date, dt.date]:
    window_days = max(int(args.outcome_window_days), 1)
    if args.end_day:
        end_day = dt.date.fromisoformat(args.end_day)
    else:
        max_day = query_one_value(
            """
            SELECT CAST(MAX(day) AS VARCHAR) AS max_day
            FROM node_day_ops_wide_full
            WHERE day >= DATE '2026-01-01'
            """,
            database_id=19,
            schema="test",
            key="max_day",
        )
        end_day = dt.date.fromisoformat(max_day)
    last_full_business_day = end_day - dt.timedelta(days=window_days - 1)
    start_day = (
        dt.date.fromisoformat(args.start_day)
        if args.start_day
        else last_full_business_day - dt.timedelta(days=args.days - 1)
    )
    if start_day > last_full_business_day:
        raise ValueError("start_day must be <= latest fully observable business day")
    return start_day, end_day, last_full_business_day


def discover_attribute_days(end_day: dt.date) -> tuple[str, str]:
    lower_text = (end_day - dt.timedelta(days=14)).strftime("%Y%m%d")
    join_day = query_one_value(
        f"""
        SELECT CAST(MAX(day) AS VARCHAR) AS max_day
        FROM node_join
        WHERE day >= {int(lower_text)}
        """,
        database_id=2,
        schema="jarvis",
        key="max_day",
    )
    analysis_day = query_one_value(
        f"""
        SELECT CAST(MAX(day) AS VARCHAR) AS max_day
        FROM node_analysis_data
        WHERE day >= {rebuild.sql_quote(lower_text)}
        """,
        database_id=2,
        schema="jarvis",
        key="max_day",
    )
    return join_day, analysis_day


def query_large_candidates_from_join(join_day: str, prefix: str | None = None) -> set[str]:
    prefix_filter = (
        f"AND SUBSTR(nj._id, 1, 1) = {rebuild.sql_quote(prefix)}"
        if prefix else ""
    )
    sql = f"""
    WITH ranked AS (
      SELECT
        nj._id AS node_id,
        LOWER(COALESCE(CAST(nj.nodestaticinfo.nominalinfo.deliverytype AS VARCHAR), '')) AS deliverytype,
        LOWER(COALESCE(CAST(nj.nodestaticinfo.nominalinfo.resourcetype AS VARCHAR), '')) AS resourcetype,
        ROW_NUMBER() OVER (
          PARTITION BY nj._id
          ORDER BY nj.hour DESC, nj.nodeinfo.lastreporttime DESC
        ) AS rn
      FROM node_join nj
      WHERE nj.day = {int(join_day)}
        {prefix_filter}
    )
    SELECT node_id
    FROM ranked
    WHERE rn = 1
      AND (
        deliverytype IN ('idc', 'dedicated')
        OR resourcetype = 'dedicated'
      )
    """
    rows = rebuild.run_sql(sql, database_id=2, schema="jarvis")
    return {str(row["node_id"]) for row in rows if row.get("node_id")}


def query_large_candidates_from_analysis(analysis_day: str, prefix: str | None = None) -> set[str]:
    prefix_filter = (
        f"AND SUBSTR(nd.nodeid, 1, 1) = {rebuild.sql_quote(prefix)}"
        if prefix else ""
    )
    sql = f"""
    WITH ranked AS (
      SELECT
        nd.nodeid AS node_id,
        LOWER(COALESCE(CAST(nd.resourcetype AS VARCHAR), '')) AS resourcetype,
        COALESCE(CAST(nd.supplysidedeliverytype AS VARCHAR), '') AS supply_type,
        TRY_CAST(nd.bandwidth AS DOUBLE) AS bandwidth,
        TRY_CAST(nd.actualbandwidth AS DOUBLE) AS actualbandwidth,
        TRY_CAST(nd.corenum AS DOUBLE) AS corenum,
        ROW_NUMBER() OVER (
          PARTITION BY nd.nodeid
          ORDER BY TRY_CAST(nd.hour AS INTEGER) DESC, nd.time DESC
        ) AS rn
      FROM node_analysis_data nd
      WHERE nd.day = {rebuild.sql_quote(analysis_day)}
        {prefix_filter}
    )
    SELECT node_id
    FROM ranked
    WHERE rn = 1
      AND (
        resourcetype = 'dedicated'
        OR supply_type IN ('机房', '专线')
        OR (supply_type = '汇聚' AND GREATEST(COALESCE(bandwidth, 0), COALESCE(actualbandwidth, 0)) >= 100)
        OR (GREATEST(COALESCE(bandwidth, 0), COALESCE(actualbandwidth, 0)) >= 1000 AND COALESCE(corenum, 0) >= 8)
      )
    """
    rows = rebuild.run_sql(sql, database_id=2, schema="jarvis")
    return {str(row["node_id"]) for row in rows if row.get("node_id")}


def discover_large_candidates(output_dir: Path, end_day: dt.date, refresh: bool) -> list[str]:
    output_path = output_dir / "large_candidate_node_ids_recent_1m.json"
    if output_path.exists() and not refresh:
        return json.loads(output_path.read_text(encoding="utf-8"))["node_ids"]
    join_day, analysis_day = discover_attribute_days(end_day)
    prefixes = list("0123456789abcdef")
    join_parts: dict[str, set[str]] = {}
    analysis_parts: dict[str, set[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        join_futures = {
            executor.submit(query_large_candidates_from_join, join_day, prefix): prefix
            for prefix in prefixes
        }
        analysis_futures = {
            executor.submit(query_large_candidates_from_analysis, analysis_day, prefix): prefix
            for prefix in prefixes
        }
        for future in concurrent.futures.as_completed([*join_futures, *analysis_futures]):
            if future in join_futures:
                prefix = join_futures[future]
                join_parts[prefix] = future.result()
                print(f"candidate join prefix {prefix}: {len(join_parts[prefix]):,}")
            else:
                prefix = analysis_futures[future]
                analysis_parts[prefix] = future.result()
                print(f"candidate analysis prefix {prefix}: {len(analysis_parts[prefix]):,}")
    join_nodes = set().union(*join_parts.values()) if join_parts else set()
    analysis_nodes = set().union(*analysis_parts.values()) if analysis_parts else set()
    node_ids = sorted(join_nodes | analysis_nodes)
    payload = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "candidate_strategy": (
            "node_analysis_data uses supply/resource/capacity large-node rules sharded by first node_id hex; "
            "node_join uses explicit idc/dedicated resource rules sharded by first node_id hex."
        ),
        "join_day": join_day,
        "analysis_day": analysis_day,
        "join_candidates": len(join_nodes),
        "analysis_candidates": len(analysis_nodes),
        "join_prefix_counts": {prefix: len(join_parts.get(prefix, set())) for prefix in prefixes},
        "analysis_prefix_counts": {prefix: len(analysis_parts.get(prefix, set())) for prefix in prefixes},
        "node_ids": node_ids,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"large candidates: join={len(join_nodes):,}, "
        f"analysis={len(analysis_nodes):,}, union={len(node_ids):,}"
    )
    return node_ids


def outcome_sql_for_nodes(
    node_ids: list[str],
    start_day: dt.date,
    end_day: dt.date,
    last_full_business_day: dt.date,
    outcome_window_days: int,
) -> str:
    ids = ",\n".join(rebuild.sql_quote(value) for value in node_ids)
    if outcome_window_days == 1:
        return f"""
        WITH daily_business AS (
          SELECT
            nodeId AS node_id,
            customerId AS business,
            day AS sample_day,
            COUNT(*) AS daily_rows,
            MAX(CASE WHEN state = 'online' THEN 1 ELSE 0 END) AS ever_online,
            MIN(COALESCE(CAST(customerName AS VARCHAR), '')) AS business_name,
            COALESCE(SUM(CAST(cost_finalAmount AS DOUBLE)), 0.0) AS cum_cost_7d,
            COALESCE(SUM(CAST(revenue_finalAmount AS DOUBLE)), 0.0) AS cum_revenue_7d,
            COUNT(day) AS outcome_days,
            COUNT(DISTINCT day) AS outcome_distinct_days
          FROM node_day_ops_wide_full
          WHERE day BETWEEN DATE '{start_day.isoformat()}' AND DATE '{end_day.isoformat()}'
            AND customerId > 0
            AND nodeId IN ({ids})
          GROUP BY nodeId, customerId, day
        ), business_active AS (
          SELECT
            node_id,
            business,
            COUNT(DISTINCT sample_day) AS business_active_days
          FROM daily_business
          GROUP BY node_id, business
        ), eligible_nodes AS (
          SELECT
            node_id,
            MIN(sample_day) AS online_day,
            COUNT(DISTINCT business) AS business_count,
            COUNT(DISTINCT sample_day) AS business_first_day_count,
            MAX(ever_online) AS ever_online
          FROM daily_business
          GROUP BY node_id
          HAVING COUNT(DISTINCT business) >= {rebuild.MIN_BUSINESS_COUNT}
             AND COUNT(DISTINCT sample_day) >= {rebuild.MIN_BUSINESS_COUNT}
             AND MAX(ever_online) = 1
        )
        SELECT
          d.node_id,
          CAST(d.business AS VARCHAR) AS business,
          d.sample_day,
          CAST(d.sample_day AS VARCHAR) || ':' || CAST(d.node_id AS VARCHAR) AS outcome_group_id,
          d.sample_day AS business_online_day,
          a.business_active_days,
          e.online_day,
          e.business_count,
          e.business_first_day_count,
          e.ever_online,
          d.business_name,
          d.cum_cost_7d,
          d.cum_revenue_7d,
          d.outcome_days,
          d.outcome_distinct_days,
          1 AS outcome_window_days,
          'daily_node_business' AS outcome_grain
        FROM daily_business d
        JOIN eligible_nodes e ON d.node_id = e.node_id
        LEFT JOIN business_active a
          ON d.node_id = a.node_id
         AND d.business = a.business
        ORDER BY d.node_id, d.sample_day, d.business
        """
    window_end_offset = outcome_window_days - 1
    return f"""
    WITH first_business AS (
      SELECT
        nodeId AS node_id,
        customerId AS business,
        MIN(day) AS first_business_day,
        COUNT(DISTINCT day) AS active_days,
        MAX(CASE WHEN state = 'online' THEN 1 ELSE 0 END) AS ever_online
      FROM node_day_ops_wide_full
      WHERE day BETWEEN DATE '{start_day.isoformat()}' AND DATE '{end_day.isoformat()}'
        AND customerId > 0
        AND nodeId IN ({ids})
      GROUP BY nodeId, customerId
    ), full_window_business AS (
      SELECT *
      FROM first_business
      WHERE first_business_day <= DATE '{last_full_business_day.isoformat()}'
    ), eligible_nodes AS (
      SELECT
        node_id,
        MIN(first_business_day) AS online_day,
        COUNT(*) AS business_count,
        COUNT(DISTINCT first_business_day) AS business_first_day_count,
        MAX(ever_online) AS ever_online
      FROM full_window_business
      GROUP BY node_id
      HAVING COUNT(*) >= {rebuild.MIN_BUSINESS_COUNT}
         AND COUNT(DISTINCT first_business_day) >= {rebuild.MIN_BUSINESS_COUNT}
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
      COUNT(DISTINCT t.day) AS outcome_distinct_days,
      {outcome_window_days} AS outcome_window_days,
      'first_business_window' AS outcome_grain
    FROM full_window_business p
    JOIN eligible_nodes e ON p.node_id = e.node_id
    LEFT JOIN node_day_ops_wide_full t
      ON t.nodeId = p.node_id
     AND t.customerId = p.business
     AND t.day >= p.first_business_day
     AND t.day <= DATE_ADD(p.first_business_day, INTERVAL {window_end_offset} DAY)
    GROUP BY p.node_id, p.business, p.first_business_day, p.active_days,
             e.online_day, e.business_count, e.business_first_day_count, e.ever_online
    ORDER BY p.node_id, p.first_business_day, p.business
    """


def fetch_outcomes(
    candidate_node_ids: list[str],
    output_dir: Path,
    start_day: dt.date,
    end_day: dt.date,
    last_full_business_day: dt.date,
    outcome_window_days: int,
    chunk_size: int,
    workers: int,
    refresh: bool,
) -> pd.DataFrame:
    output_path = output_dir / "multibusiness_outcomes_large_candidates_recent_1m.csv"
    if output_path.exists() and not refresh:
        print(f"reuse cached {output_path}")
        return pd.read_csv(output_path, low_memory=False)

    node_chunks = chunks(candidate_node_ids, chunk_size)

    def fetch_one(index: int, node_chunk: list[str]) -> tuple[int, list[dict[str, Any]]]:
        rows = rebuild.run_sql(
            outcome_sql_for_nodes(
                node_chunk,
                start_day,
                end_day,
                last_full_business_day,
                outcome_window_days,
            ),
            database_id=19,
            schema="test",
        )
        return index, rows

    all_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(fetch_one, index, node_chunk)
            for index, node_chunk in enumerate(node_chunks, 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            index, rows = future.result()
            all_rows.extend(rows)
            print(f"outcome chunk {index}/{len(node_chunks)}: rows={len(rows):,}")

    outcomes = pd.DataFrame(all_rows)
    if outcomes.empty:
        raise RuntimeError("recent large candidate outcomes are empty")
    outcomes.to_csv(output_path, index=False)
    print(f"wrote {output_path}: {len(outcomes):,} rows")
    return outcomes


def fetch_nodes_for_outcomes(
    outcomes: pd.DataFrame,
    output_dir: Path,
    refresh: bool,
    skip_prometheus: bool,
) -> pd.DataFrame:
    output_path = output_dir / "multibusiness_nodes_large_candidates_recent_1m.csv"
    prom_path = output_dir / "multibusiness_prometheus_probe_recent_1m.json"
    if output_path.exists() and not refresh:
        print(f"reuse cached {output_path}")
        return pd.read_csv(output_path, low_memory=False)

    rebuild.OUTPUT_NODES = output_path
    rebuild.OUTPUT_PROM = prom_path
    targets = (
        outcomes[["node_id", "online_day"]]
        .assign(node_id=lambda frame: frame["node_id"].astype(str))
        .drop_duplicates("node_id")
    )
    nodes = rebuild.fetch_attributes(targets)
    if not skip_prometheus:
        node_ids = sorted(targets["node_id"].astype(str).unique())
        prom_probe = rebuild.probe_prometheus(node_ids)
        print(f"prometheus mapping usable={prom_probe.get('mapping_usable')}")
        nodes = rebuild.enrich_nodes_with_prometheus(nodes, node_ids)
    return nodes


def export_large_only_data(
    nodes: pd.DataFrame,
    outcomes: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    nodes_with_type = v2.add_buckets(nodes)
    large_nodes = nodes_with_type[nodes_with_type["node_size_type"] == v2.NODE_SIZE_LARGE].copy()
    large_node_ids = set(large_nodes["node_id"].astype(str))
    large_outcomes = outcomes[outcomes["node_id"].astype(str).isin(large_node_ids)].copy()
    nodes_path = output_dir / "multibusiness_nodes_large_recent_1m.csv"
    outcomes_path = output_dir / "multibusiness_outcomes_large_recent_1m.csv"
    large_nodes.to_csv(nodes_path, index=False)
    large_outcomes.to_csv(outcomes_path, index=False)
    print(
        f"large-only data: nodes={len(large_nodes):,}, "
        f"outcomes={len(large_outcomes):,}, businesses={large_outcomes['business'].nunique():,}"
    )
    if large_nodes.empty or large_outcomes.empty:
        raise RuntimeError("large-only training data is empty after final heuristic filter")
    return nodes_path, outcomes_path


def build_models_and_report(
    output_dir: Path,
    nodes_path: Path,
    outcomes_path: Path,
    business_map: Path,
    min_business_support: int,
) -> dict[str, Path]:
    v1_dir = output_dir / "v1_large_recent_outputs"
    v2_dir = output_dir / "v2_large_recent_outputs"
    run_command([
        sys.executable,
        "v1_recommendation_pipeline.py",
        "build",
        "--nodes",
        str(nodes_path),
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
    ])
    training_pairs = v1_dir / "v1_training_pairs.csv"
    training_pairs_copy = output_dir / "v1_training_pairs_large_recent_1m.csv"
    shutil.copyfile(training_pairs, training_pairs_copy)

    run_command([
        sys.executable,
        "v2_ranking_model.py",
        "build",
        "--pairs",
        str(training_pairs),
        "--business-map",
        str(business_map),
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
        "--top-k",
        "3",
    ])

    report_csv = output_dir / "v2_frontend_condition_business_report_large_recent_1m.csv"
    report_json = output_dir / "v2_frontend_condition_business_report_large_recent_1m.json"
    report_data_js = output_dir / "v2_frontend_condition_business_report_large_recent_1m_data.js"
    run_command([
        sys.executable,
        "v2_ranking_model.py",
        "export-condition-report",
        "--nodes",
        str(nodes_path),
        "--model",
        str(v2_dir / "v2_ranking_model.json"),
        "--recommendations",
        str(v2_dir / "v2_node_recommendations.csv"),
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
    html_dst = output_dir / "v2_frontend_condition_business_report_large_recent_1m.html"
    html = html_src.read_text(encoding="utf-8").replace(
        "v2_frontend_condition_business_report_large_data.js",
        report_data_js.name,
    )
    html_dst.write_text(html, encoding="utf-8")

    run_command([sys.executable, "v2_ranking_model.py", "check", "--output-dir", str(v2_dir)])
    if shutil.which("node"):
        run_command(["node", "--check", str(report_data_js)])

    return {
        "v1_dir": v1_dir,
        "v2_dir": v2_dir,
        "training_pairs": training_pairs_copy,
        "report_html": html_dst,
        "report_csv": report_csv,
        "report_json": report_json,
        "report_data_js": report_data_js,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recent one-month large-node training artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--business-map", type=Path, default=DEFAULT_BUSINESS_MAP)
    parser.add_argument("--days", type=int, default=31)
    parser.add_argument("--start-day", help="First eligible business_online_day, YYYY-MM-DD.")
    parser.add_argument("--end-day", help="Latest wide-table day for outcome data, YYYY-MM-DD.")
    parser.add_argument(
        "--outcome-window-days",
        type=int,
        default=7,
        help="Outcome window size. Use 1 for independent node-business-day training samples.",
    )
    parser.add_argument("--candidate-chunk-size", type=int, default=800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-business-support", type=int, default=2)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-prometheus", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_day, end_day, last_full_business_day = discover_window(args)
    print(
        "recent window: "
        f"business_online_day={start_day.isoformat()}..{last_full_business_day.isoformat()}, "
        f"outcome_data_until={end_day.isoformat()}, "
        f"outcome_window_days={args.outcome_window_days}"
    )
    candidate_node_ids = discover_large_candidates(output_dir, end_day, args.refresh)
    outcomes = fetch_outcomes(
        candidate_node_ids,
        output_dir,
        start_day,
        end_day,
        last_full_business_day,
        args.outcome_window_days,
        args.candidate_chunk_size,
        args.workers,
        args.refresh,
    )
    nodes = fetch_nodes_for_outcomes(outcomes, output_dir, args.refresh, args.skip_prometheus)
    nodes_path, outcomes_path = export_large_only_data(nodes, outcomes, output_dir)
    artifacts = build_models_and_report(
        output_dir,
        nodes_path,
        outcomes_path,
        args.business_map,
        args.min_business_support,
    )
    summary_path = output_dir / "recent_large_training_summary.json"
    metrics = json.loads((artifacts["v2_dir"] / "v2_model_metrics.json").read_text(encoding="utf-8"))
    summary = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "business_online_day_start": start_day.isoformat(),
        "business_online_day_end": last_full_business_day.isoformat(),
        "wide_outcome_end_day": end_day.isoformat(),
        "outcome_window_days": int(args.outcome_window_days),
        "outcome_grain": "daily_node_business" if int(args.outcome_window_days) == 1 else "first_business_window",
        "candidate_nodes": len(candidate_node_ids),
        "large_nodes": int(pd.read_csv(nodes_path, usecols=["node_id"]).shape[0]),
        "large_outcome_rows": int(pd.read_csv(outcomes_path, usecols=["node_id"]).shape[0]),
        "v2_test": metrics.get("test", {}),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "elapsed_sec": round(time.time() - started, 1),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
