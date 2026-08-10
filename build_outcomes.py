#!/usr/bin/env python3
"""从真实数据构建节点×业务的上线 7 天成本/收入结果。

默认使用宽表中该节点的首次出现日期；设置 USE_SERVICE_TIME=1 时可选用
jarvis.node_join.nodestaticinfo.servicetime。窗口为 [online_day, online_day + 6]。
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient


BASE = "http://superset.yzh-logverse.k8s.qiniu.io"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "outcomes_raw.csv")

JARVIS_DATABASE_ID = 2
OUTCOME_DATABASE_ID = 19
JARVIS_SNAPSHOT_DAY = 20260805
USE_SERVICE_TIME = os.environ.get("USE_SERVICE_TIME", "0") == "1"
WINDOW_START = dt.date(2026, 4, 1)
WINDOW_END = dt.date(2026, 7, 28)
SAMPLE_SUFFIXES = tuple(f"{value:02x}" for value in range(13))  # 13/256 = 5.08%
SERVICE_BATCH_SIZE = 1000
OUTCOME_BATCH_SIZE = 1000
OUTCOME_QUERY_END = WINDOW_END + dt.timedelta(days=6)
OUTCOME_WORKERS = int(os.environ.get("OUTCOME_WORKERS", "4"))


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def normalize_day(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return None


def normalize_service_day(value: Any) -> str | None:
    return normalize_day(value)


def load_service_days(node_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Return serviceTime for sampled nodes without scanning all Jarvis nodes."""
    client = SupersetSQLClient(base_url=BASE, database_id=JARVIS_DATABASE_ID)
    service_days: dict[str, tuple[str, str]] = {}
    service_expr = "TRY(from_iso8601_timestamp(nodestaticinfo.servicetime))"

    for start in range(0, len(node_ids), SERVICE_BATCH_SIZE):
        batch = node_ids[start : start + SERVICE_BATCH_SIZE]
        ids = ",\n".join(sql_quote(node_id) for node_id in batch)
        sql = f"""
        SELECT
          _id AS node_id,
          MIN({service_expr}) AS service_time
        FROM jarvis.node_join
        WHERE day = {JARVIS_SNAPSHOT_DAY}
          AND nodestaticinfo.servicetime IS NOT NULL
          AND _id IN ({ids})
        GROUP BY _id
        """
        rows = client.execute_sql(sql=sql, schema="jarvis")
        for row in rows:
            online_day = normalize_service_day(row.get("service_time"))
            if online_day and WINDOW_START.isoformat() <= online_day <= WINDOW_END.isoformat():
                service_days[str(row["node_id"])] = (online_day, "serviceTime")
        print(f"serviceTime batch {start}-{start + len(batch)}: {len(rows)} rows")
    print(f"serviceTime candidates: {len(service_days):,} nodes")
    return service_days


def load_first_seen_days() -> dict[str, str]:
    """Load the same deterministic sample's first appearance fallback from the wide table."""
    client = SupersetSQLClient(base_url=BASE, database_id=OUTCOME_DATABASE_ID)
    first_seen: dict[str, str] = {}
    suffixes = ", ".join(sql_quote(value) for value in SAMPLE_SUFFIXES)
    for suffix in SAMPLE_SUFFIXES:
        sql = f"""
        SELECT nodeId AS node_id, MIN(day) AS first_day
        FROM test.node_day_ops_wide_full
        WHERE day BETWEEN DATE '{WINDOW_START.isoformat()}' AND DATE '{WINDOW_END.isoformat()}'
          AND RIGHT(nodeId, 2) IN ({suffixes})
          AND RIGHT(nodeId, 2) = {sql_quote(suffix)}
        GROUP BY nodeId
        HAVING MIN(day) >= DATE '{WINDOW_START.isoformat()}'
           AND MIN(day) <= DATE '{WINDOW_END.isoformat()}'
        """
        rows = client.execute_sql(sql=sql, schema="test")
        for row in rows:
            online_day = normalize_day(row.get("first_day"))
            if online_day:
                first_seen[str(row["node_id"])] = online_day
        print(f"first-appearance suffix {suffix}: {len(rows)} rows")
    print(f"first-appearance candidates: {len(first_seen):,} nodes")
    return first_seen


def load_online_nodes() -> dict[str, tuple[str, str]]:
    first_seen = load_first_seen_days()
    service_days = load_service_days(sorted(first_seen)) if USE_SERVICE_TIME else {}
    online_nodes = dict(service_days)
    fallback_count = 0
    for node_id, online_day in first_seen.items():
        if node_id not in online_nodes:
            online_nodes[node_id] = (online_day, "first_appearance")
            fallback_count += 1
    print(f"fallback first-appearance nodes: {fallback_count:,}")
    print(f"sampled online nodes: {len(online_nodes):,}")
    return online_nodes


def fetch_outcomes(online_nodes: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    nodes_by_online_day: dict[str, list[str]] = {}
    for node_id, (online_day, _source) in online_nodes.items():
        nodes_by_online_day.setdefault(online_day, []).append(node_id)

    jobs: list[tuple[int, str, list[str]]] = []
    batch_number = 0
    for online_day, node_ids in sorted(nodes_by_online_day.items()):
        for start in range(0, len(node_ids), OUTCOME_BATCH_SIZE):
            jobs.append((batch_number, online_day, node_ids[start : start + OUTCOME_BATCH_SIZE]))
            batch_number += 1

    def run_batch(job: tuple[int, str, list[str]]) -> tuple[int, str, int, list[dict[str, Any]]]:
        number, online_day, batch = job
        window_end = dt.date.fromisoformat(online_day) + dt.timedelta(days=6)
        ids = ",\n".join(sql_quote(node_id) for node_id in batch)
        client = SupersetSQLClient(
            base_url=BASE,
            database_id=OUTCOME_DATABASE_ID,
            query_limit=3_000_000,
        )
        sql = f"""
        SELECT node_id, business,
               SUM(cost_value) AS cum_cost_7d,
               SUM(revenue_value) AS cum_revenue_7d
        FROM (
          SELECT
            t.nodeId AS node_id,
            t.customerId AS business,
            COALESCE(CAST(t.cost_finalAmount AS DOUBLE), 0.0) AS cost_value,
            COALESCE(CAST(t.revenue_finalAmount AS DOUBLE), 0.0) AS revenue_value
          FROM test.node_day_ops_wide_full t
          WHERE t.day BETWEEN DATE '{online_day}' AND DATE '{window_end.isoformat()}'
            AND t.nodeId IN ({ids})
        ) window_rows
        GROUP BY node_id, business
        """
        rows = client.execute_sql(sql=sql, schema="test")
        output: list[dict[str, Any]] = []
        for row in rows:
            node_id = str(row["node_id"])
            _online_day, source = online_nodes[node_id]
            output.append({
                "node_id": node_id,
                "business": str(row["business"]),
                "cum_cost_7d": float(row.get("cum_cost_7d") or 0.0),
                "cum_revenue_7d": float(row.get("cum_revenue_7d") or 0.0),
                "online_day": online_day,
                "online_source": source,
            })
        return number, online_day, len(batch), output

    results: dict[int, tuple[str, int, list[dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=OUTCOME_WORKERS) as executor:
        future_map = {executor.submit(run_batch, job): job[0] for job in jobs}
        for future in as_completed(future_map):
            number, online_day, node_count, rows = future.result()
            results[number] = (online_day, node_count, rows)
            print(
                f"outcome batch {number}: online_day={online_day}, "
                f"nodes={node_count}, rows={len(rows)}"
            )

    all_rows: list[dict[str, Any]] = []
    for number in sorted(results):
        _online_day, _node_count, rows = results[number]
        all_rows.extend(rows)
    return all_rows


def build_outcomes() -> None:
    online_nodes = load_online_nodes()
    rows = fetch_outcomes(online_nodes)
    if not rows:
        raise RuntimeError("No outcome rows were returned; refusing to overwrite the output.")

    fields = [
        "node_id", "business", "cum_cost_7d", "cum_revenue_7d",
        "online_day", "online_source",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    print(f"total outcome rows: {len(rows):,}")
    print(f"distinct nodes: {len({row['node_id'] for row in rows}):,}")
    print(f"distinct businesses: {len({row['business'] for row in rows}):,}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    build_outcomes()
