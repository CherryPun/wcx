#!/usr/bin/env python3
"""按节点上线日当天的 node_join 最新小时快照抽取节点自有属性。"""
from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, "/Users/nany/Downloads/superset-sql-query-skill/common")
from superset_api import SupersetSQLClient


BASE = "http://superset.yzh-logverse.k8s.qiniu.io"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTCOMES_CSV = os.path.join(HERE, "outcomes_raw.csv")
NODES_CSV = os.path.join(HERE, "nodes.csv")
NODE_BATCH_SIZE = 1500
ATTRIBUTE_WORKERS = int(os.environ.get("ATTRIBUTE_WORKERS", "4"))


FEATURE_COLUMNS = [
    "node_id", "online_day", "attribute_snapshot_day", "attribute_snapshot_hour",
    "vendorid", "deliverytype", "resourcetype", "dialtype", "nattype",
    "scheduleisps", "regsource", "customermode", "province", "isp", "city",
    "device_type", "arch_type", "isvm", "qoskiller_status", "bw",
]


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def read_node_days() -> dict[str, str]:
    node_days: dict[str, str] = {}
    with open(OUTCOMES_CSV, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node_id = str(row.get("node_id") or "").strip()
            online_day = (row.get("online_day") or row.get("online_time") or "")[:10]
            if node_id and online_day:
                node_days[node_id] = online_day
    return node_days


def fetch_attributes(node_days: dict[str, str]) -> list[dict[str, Any]]:
    nodes_by_day: dict[str, list[str]] = {}
    for node_id, online_day in node_days.items():
        nodes_by_day.setdefault(online_day, []).append(node_id)

    jobs: list[tuple[int, str, list[str]]] = []
    batch_number = 0
    for online_day, node_ids in sorted(nodes_by_day.items()):
        for start in range(0, len(node_ids), NODE_BATCH_SIZE):
            jobs.append((batch_number, online_day, node_ids[start : start + NODE_BATCH_SIZE]))
            batch_number += 1

    def run_batch(job: tuple[int, str, list[str]]) -> tuple[int, str, int, list[dict[str, Any]]]:
        number, online_day, batch = job
        snapshot_day = int(online_day.replace("-", ""))
        ids = ",\n".join(sql_quote(node_id) for node_id in batch)
        client = SupersetSQLClient(base_url=BASE, database_id=2, query_limit=3_000_000)
        sql = f"""
        SELECT
          node_id, DATE '{online_day}' AS online_day,
          attribute_snapshot_day, attribute_snapshot_hour,
          vendorid, deliverytype, resourcetype, dialtype, nattype,
          scheduleisps, regsource, customermode, province, isp, city,
          device_type, arch_type, isvm, qoskiller_status, bw
        FROM (
          SELECT
            nj._id AS node_id,
            nj.day AS attribute_snapshot_day,
            nj.hour AS attribute_snapshot_hour,
            nj.nodestaticinfo.vendorid AS vendorid,
            nj.nodestaticinfo.nominalinfo.deliverytype AS deliverytype,
            nj.nodestaticinfo.nominalinfo.resourcetype AS resourcetype,
            nj.nodestaticinfo.nominalinfo.dialtype AS dialtype,
            nj.nodestaticinfo.nominalinfo.nattype AS nattype,
            element_at(nj.nodestaticinfo.nominalinfo.scheduleisps, 1) AS scheduleisps,
            nj.nodestaticinfo.regsource AS regsource,
            nj.nodestaticinfo.customermode AS customermode,
            nj.nodestaticinfo.nominalinfo.province AS province,
            nj.nodestaticinfo.nominalinfo.isp AS isp,
            nj.nodestaticinfo.nominalinfo.city AS city,
            nj.nodeinfo.devicetype AS device_type,
            nj.nodestaticinfo.nominalinfo.nodearchtype AS arch_type,
            CAST(nj.nodestaticinfo.nominalinfo.iscloudvm AS VARCHAR) AS isvm,
            nj.nodestaticinfo.nodeqoskillerconfig.status AS qoskiller_status,
            nj.nodeinfo.bw AS bw,
            ROW_NUMBER() OVER (
              PARTITION BY nj._id
              ORDER BY nj.hour DESC, nj.nodeinfo.lastreporttime DESC
            ) AS rn
          FROM jarvis.node_join nj
          WHERE nj.day = {snapshot_day}
            AND nj._id IN ({ids})
        ) ranked
        WHERE rn = 1
        """
        rows = client.execute_sql(sql=sql, schema="jarvis")
        return number, online_day, len(batch), rows

    results: dict[int, tuple[str, int, list[dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=ATTRIBUTE_WORKERS) as executor:
        future_map = {executor.submit(run_batch, job): job[0] for job in jobs}
        for future in as_completed(future_map):
            number, online_day, node_count, part = future.result()
            results[number] = (online_day, node_count, part)
            print(
                f"attribute batch {number}: online_day={online_day}, "
                f"nodes={node_count}, rows={len(part)}"
            )

    rows: list[dict[str, Any]] = []
    for number in sorted(results):
        _online_day, _node_count, part = results[number]
        rows.extend(part)
    return rows


def write_attributes(rows: list[dict[str, Any]], expected_nodes: int) -> None:
    if not rows:
        raise RuntimeError("No node attributes were returned; refusing to overwrite the output.")
    with open(NODES_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            output = {"node_id": row.get("node_id")}
            for column in FEATURE_COLUMNS[1:]:
                value = row.get(column)
                output[column] = "" if value is None else str(value)
            writer.writerow(output)
    match_rate = len({row.get("node_id") for row in rows}) / max(expected_nodes, 1)
    print(f"attribute rows: {len(rows):,}; as-of match rate: {match_rate:.1%}")
    print(f"wrote {NODES_CSV}")


def main() -> None:
    node_days = read_node_days()
    print(f"distinct nodes to fetch: {len(node_days):,}")
    rows = fetch_attributes(node_days)
    write_attributes(rows, len(node_days))


if __name__ == "__main__":
    main()
