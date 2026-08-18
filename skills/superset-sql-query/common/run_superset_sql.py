#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from superset_api import (
    DEFAULT_BASE_URL,
    DEFAULT_DATABASE_ID,
    DEFAULT_PROVIDER,
    DEFAULT_QUERY_LIMIT,
    DEFAULT_SCHEMA,
    DEFAULT_TIMEOUT,
    SupersetSQLClient,
    env_or_die,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute ad-hoc SQL against Superset SQL Lab.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sql", help="Inline SQL string.")
    group.add_argument("--sql-file", help="Path to a SQL file. Use - to read from stdin.")
    parser.add_argument("--output-json", help="Optional path to write the full result set as JSON.")
    parser.add_argument("--output-csv", help="Optional path to write the full result set as CSV.")
    parser.add_argument("--preview", type=int, default=10, help="How many rows to print in stdout preview.")
    parser.add_argument("--tab", default="临时SQL执行", help="Superset SQL Lab tab title.")
    parser.add_argument("--sql-editor-id", default="adhoc", help="Superset SQL editor id.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Superset base URL.")
    parser.add_argument("--database-id", type=int, default=DEFAULT_DATABASE_ID, help="Superset database id.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Superset schema.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Superset auth provider.")
    parser.add_argument("--query-limit", type=int, default=DEFAULT_QUERY_LIMIT, help="Superset SQLLab query limit.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds.")
    parser.add_argument("--username", help="Optional Superset username override.")
    parser.add_argument("--password", help="Optional Superset password override.")
    return parser.parse_args()


def read_sql(args: argparse.Namespace) -> str:
    if args.sql:
        return args.sql
    if args.sql_file:
        if args.sql_file == "-":
            return sys.stdin.read()
        return Path(args.sql_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --sql, --sql-file, or pipe SQL via stdin.")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    sql = read_sql(args).strip()
    if not sql:
        raise SystemExit("SQL is empty.")

    client = SupersetSQLClient(
        base_url=args.base_url,
        database_id=args.database_id,
        schema=args.schema,
        provider=args.provider,
        query_limit=args.query_limit,
        timeout=args.timeout,
        username=args.username or env_or_die("SUPERSET_USERNAME"),
        password=args.password or env_or_die("SUPERSET_PASSWORD"),
    )
    rows = client.execute_sql(
        sql=sql,
        query_limit=args.query_limit,
        tab=args.tab,
        sql_editor_id=args.sql_editor_id,
    )

    json_path = None
    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    csv_path = None
    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, rows)

    preview_size = max(args.preview, 0)
    payload = {
        "row_count": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "preview_rows": rows[:preview_size],
        "output_json": str(json_path) if json_path else None,
        "output_csv": str(csv_path) if csv_path else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
