#!/usr/bin/env python3
"""Fetch one latest packet-loss benchmark result per node from Superset."""
from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import rebuild_multibusiness_model as rebuild
import v1_recommendation_pipeline as v1


HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = (
    HERE
    / "recent_month_large_mainstream_v3_daily_weighted"
    / "v1_large_mainstream_outputs"
    / "v1_training_pairs.csv"
)
DEFAULT_CURRENT_NODES = (
    HERE
    / "recent_month_large_mainstream_v3_daily_weighted"
    / "current_online_inservice_non_idc_large_nodes_v3.csv"
)
DEFAULT_OUTPUT = HERE / "latest_node_pressure_profiles.csv"


def sql_quote(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def load_node_ids(paths: list[Path]) -> list[str]:
    node_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0).columns
        if "node_id" not in header:
            raise RuntimeError(f"node_id column not found: {path}")
        frame = pd.read_csv(path, usecols=["node_id"], dtype="string", low_memory=False)
        node_ids.update(
            v1.clean_cell(value)
            for value in frame["node_id"]
            if v1.clean_cell(value)
        )
    return sorted(node_ids)


def latest_pressure_sql(node_ids: list[str], snapshot_day: dt.date) -> str:
    ids = ",\n".join(sql_quote(value) for value in node_ids)
    day = int(snapshot_day.strftime("%Y%m%d"))
    return f"""
    SELECT
      node_id,
      pressure_snapshot_day,
      pressure_snapshot_hour,
      pressure_last_report_time,
      CASE
        WHEN pressure_expected_bw_total_mbps > 0 THEN LEAST(
          100.0,
          GREATEST(
            0.0,
            100.0 * packet_loss_benchmark_bw_total_mbps
              / pressure_expected_bw_total_mbps
          )
        )
        ELSE LEAST(
          100.0,
          GREATEST(0.0, raw_summary_packet_loss_satisfaction_pct)
        )
      END AS overall_packet_loss_benchmark_satisfaction_pct,
      packet_loss_benchmark_bw_total_mbps,
      pressure_expected_bw_total_mbps,
      pressure_line_count,
      raw_summary_packet_loss_satisfaction_pct,
      overall_limit_benchmark_satisfaction_pct
    FROM (
      SELECT
        nj._id AS node_id,
        nj.day AS pressure_snapshot_day,
        nj.hour AS pressure_snapshot_hour,
        nj.nodeinfo.lastreporttime AS pressure_last_report_time,
        REDUCE(
          nj.nodeinfo.netbenchresults.lineresult,
          CAST(0 AS DOUBLE),
          (total, line) -> total + COALESCE(line.limitbw, 0),
          total -> total
        ) AS packet_loss_benchmark_bw_total_mbps,
        REDUCE(
          nj.nodeinfo.netbenchresults.lineresult,
          CAST(0 AS DOUBLE),
          (total, line) -> total + COALESCE(line.expectedbw, 0),
          total -> total
        ) AS pressure_expected_bw_total_mbps,
        cardinality(nj.nodeinfo.netbenchresults.lineresult) AS pressure_line_count,
        nj.nodeinfo.netbenchresults.limitbwachieverate
          AS raw_summary_packet_loss_satisfaction_pct,
        nj.nodeinfo.netbenchresults.actualbwachieverate
          AS overall_limit_benchmark_satisfaction_pct,
        ROW_NUMBER() OVER (
          PARTITION BY nj._id
          ORDER BY nj.hour DESC, nj.nodeinfo.lastreporttime DESC
        ) AS rn
      FROM node_join nj
      WHERE nj.day = {day}
        AND nj._id IN ({ids})
        AND nj.nodeinfo.netbenchresults.limitbwachieverate IS NOT NULL
    ) ranked
    WHERE rn = 1
    """


def fetch_latest_pressure(
    node_ids: list[str],
    snapshot_day: dt.date,
    batch_size: int,
    workers: int,
) -> pd.DataFrame:
    batches = [
        node_ids[index:index + batch_size]
        for index in range(0, len(node_ids), batch_size)
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                rebuild.run_sql,
                latest_pressure_sql(batch, snapshot_day),
                2,
                "jarvis",
            ): index
            for index, batch in enumerate(batches, 1)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            batch_rows = future.result()
            rows.extend(batch_rows)
            print(
                f"pressure batch {batch_index}/{len(batches)}: "
                f"rows={len(batch_rows):,}"
            )
    columns = [
        "node_id",
        "pressure_snapshot_day",
        "pressure_snapshot_hour",
        "pressure_last_report_time",
        "overall_packet_loss_benchmark_satisfaction_pct",
        "packet_loss_benchmark_bw_total_mbps",
        "pressure_expected_bw_total_mbps",
        "pressure_line_count",
        "raw_summary_packet_loss_satisfaction_pct",
        "overall_limit_benchmark_satisfaction_pct",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    output = pd.DataFrame(rows)
    output["node_id"] = output["node_id"].map(v1.clean_cell)
    output = output.sort_values(
        ["node_id", "pressure_snapshot_day", "pressure_snapshot_hour", "pressure_last_report_time"]
    ).drop_duplicates("node_id", keep="last")
    return output[columns].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, action="append")
    parser.add_argument("--snapshot-day", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.nodes or [DEFAULT_PAIRS, DEFAULT_CURRENT_NODES]
    node_ids = load_node_ids(paths)
    output = fetch_latest_pressure(
        node_ids,
        args.snapshot_day,
        max(1, args.batch_size),
        max(1, args.workers),
    )
    if output.empty:
        raise RuntimeError(f"no pressure result found for {args.snapshot_day}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    coverage = len(output) / len(node_ids) if node_ids else 0.0
    print(
        f"latest pressure profiles: nodes={len(node_ids):,}, "
        f"matched={len(output):,}, coverage={coverage:.2%}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
