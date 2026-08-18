#!/usr/bin/env python3
"""Query VictoriaMetrics/Prometheus and flatten results for operations work."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = os.environ.get(
    "VM_PROM_BASE_URL",
    "https://vm-select.mvm.qiniu.io/select/293:0/prometheus/api/v1",
)
BEIJING_TZ = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")
UTC = dt.timezone.utc
HIGH_CARDINALITY_METRICS = ("niulink_agent_flow_upstream",)
SAFE_LABEL_HINTS = ("node_id", "switch_id", "idc_id", "province", "isp")
SUBCOMMANDS = {"instant", "range", "labels", "label-values", "series"}
GLOBAL_FLAGS = {"--allow-wide-query", "--sort-desc"}
GLOBAL_OPTIONS = {
    "--base-url",
    "--timeout",
    "--output",
    "--preview",
    "--node-file",
    "--node-column",
    "--node-placeholder",
    "--node-chunk-size",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", help="Write flattened rows to .csv or .json.")
    parser.add_argument("--preview", type=int, default=20, help="Rows to print to stdout; use 0 to disable.")
    parser.add_argument("--allow-wide-query", action="store_true", help="Allow broad high-cardinality queries.")
    parser.add_argument("--sort-desc", action="store_true", help="Sort preview/result rows by numeric value descending.")
    parser.add_argument("--node-file", help="Text or CSV file containing node IDs for chunked queries.")
    parser.add_argument("--node-column", help="CSV column to read when --node-file is a CSV.")
    parser.add_argument("--node-placeholder", default="__NODE_REGEX__")
    parser.add_argument("--node-chunk-size", type=int, default=80)

    subparsers = parser.add_subparsers(dest="command", required=True)

    instant = subparsers.add_parser("instant", help="Run /query.")
    instant.add_argument("--query", required=True)
    instant.add_argument("--time", help="UTC or offset-aware evaluation time.")
    instant.add_argument("--beijing-time", help="Asia/Shanghai evaluation time.")

    range_parser = subparsers.add_parser("range", help="Run /query_range.")
    range_parser.add_argument("--query", required=True)
    range_parser.add_argument("--start", help="UTC or offset-aware range start.")
    range_parser.add_argument("--end", help="UTC or offset-aware range end.")
    range_parser.add_argument("--beijing-start", help="Asia/Shanghai range start.")
    range_parser.add_argument("--beijing-end", help="Asia/Shanghai range end.")
    range_parser.add_argument("--step", required=True, help="Prometheus step, e.g. 30s, 1m, 300.")

    labels = subparsers.add_parser("labels", help="Run /labels.")
    labels.add_argument("--match", action="append", default=[], help="Optional match[] selector; repeatable.")
    labels.add_argument("--start")
    labels.add_argument("--end")
    labels.add_argument("--beijing-start")
    labels.add_argument("--beijing-end")

    values = subparsers.add_parser("label-values", help="Run /label/<label>/values.")
    values.add_argument("label")
    values.add_argument("--match", action="append", default=[], help="Optional match[] selector; repeatable.")
    values.add_argument("--start")
    values.add_argument("--end")
    values.add_argument("--beijing-start")
    values.add_argument("--beijing-end")

    series = subparsers.add_parser("series", help="Run /series.")
    series.add_argument("--match", action="append", default=[], required=True, help="match[] selector; repeatable.")
    series.add_argument("--start")
    series.add_argument("--end")
    series.add_argument("--beijing-start")
    series.add_argument("--beijing-end")

    return parser.parse_args(normalize_argv(argv or sys.argv[1:]))


def normalize_argv(argv: list[str]) -> list[str]:
    command_index = next((index for index, item in enumerate(argv) if item in SUBCOMMANDS), None)
    if command_index is None:
        return argv
    before = argv[:command_index]
    command = argv[command_index]
    after = argv[command_index + 1 :]
    moved: list[str] = []
    rest: list[str] = []
    index = 0
    while index < len(after):
        token = after[index]
        option = token.split("=", 1)[0]
        if option in GLOBAL_FLAGS:
            moved.append(token)
            index += 1
        elif option in GLOBAL_OPTIONS:
            moved.append(token)
            if "=" not in token:
                if index + 1 >= len(after):
                    rest.append(token)
                    index += 1
                    continue
                moved.append(after[index + 1])
                index += 2
            else:
                index += 1
        else:
            rest.append(token)
            index += 1
    return before + moved + [command] + rest


def parse_time(value: str | None, assume_tz: dt.tzinfo) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    if re.fullmatch(r"\d+(\.\d+)?", stripped):
        return stripped
    normalized = stripped.replace("Z", "+00:00")
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                parsed = dt.datetime.strptime(normalized, fmt)
                break
            except ValueError:
                parsed = None
        if parsed is None:
            raise SystemExit(f"Cannot parse time: {value}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=assume_tz)
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def fmt_time(ts: float, tz: dt.tzinfo) -> str:
    return dt.datetime.fromtimestamp(float(ts), UTC).astimezone(tz).isoformat(timespec="seconds")


def api_post(base_url: str, endpoint: str, params: list[tuple[str, str]], timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
    data = urllib.parse.urlencode(params).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "codex-prometheus-vm-query"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed for {url}: {exc}") from exc
    if payload.get("status") != "success":
        raise SystemExit(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def check_query_safety(query: str, allow: bool) -> None:
    if allow:
        return
    for metric in HIGH_CARDINALITY_METRICS:
        if metric in query and not any(hint in query for hint in SAFE_LABEL_HINTS):
            raise SystemExit(
                f"Refusing broad query on high-cardinality metric {metric}. "
                "Add node_id/switch_id/idc_id/province/isp filter or use --allow-wide-query."
            )


def read_node_ids(path: str, column: str | None) -> list[str]:
    text_path = Path(path)
    if not text_path.exists():
        raise SystemExit(f"node file not found: {path}")
    if column:
        with text_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or column not in reader.fieldnames:
                raise SystemExit(f"CSV column {column!r} not found in {path}")
            values = [row.get(column, "").strip() for row in reader]
    else:
        text = text_path.read_text()
        values = re.split(r"[\s,]+", text)
    seen: set[str] = set()
    nodes = []
    for value in values:
        node_id = value.strip()
        if not node_id or node_id.lower() == "node_id":
            continue
        if node_id not in seen:
            seen.add(node_id)
            nodes.append(node_id)
    if not nodes:
        raise SystemExit(f"no node IDs found in {path}")
    return nodes


def chunks(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise SystemExit("--node-chunk-size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def regex_for_nodes(nodes: list[str]) -> str:
    return "|".join(re.escape(node) for node in nodes)


def query_params(args: argparse.Namespace, query: str) -> tuple[str, list[tuple[str, str]]]:
    if args.command == "instant":
        when = parse_time(args.beijing_time, BEIJING_TZ) if args.beijing_time else parse_time(args.time, UTC)
        params = [("query", query)]
        if when:
            params.append(("time", when))
        return "query", params
    if args.command == "range":
        start = parse_time(args.beijing_start, BEIJING_TZ) if args.beijing_start else parse_time(args.start, UTC)
        end = parse_time(args.beijing_end, BEIJING_TZ) if args.beijing_end else parse_time(args.end, UTC)
        if not start or not end:
            raise SystemExit("range requires --start/--end or --beijing-start/--beijing-end")
        return "query_range", [("query", query), ("start", start), ("end", end), ("step", args.step)]
    raise AssertionError(args.command)


def time_params(args: argparse.Namespace) -> list[tuple[str, str]]:
    start = parse_time(getattr(args, "beijing_start", None), BEIJING_TZ) if getattr(args, "beijing_start", None) else parse_time(getattr(args, "start", None), UTC)
    end = parse_time(getattr(args, "beijing_end", None), BEIJING_TZ) if getattr(args, "beijing_end", None) else parse_time(getattr(args, "end", None), UTC)
    params: list[tuple[str, str]] = []
    if start:
        params.append(("start", start))
    if end:
        params.append(("end", end))
    return params


def run_query_command(args: argparse.Namespace) -> list[dict[str, Any]]:
    query = args.query
    check_query_safety(query, args.allow_wide_query)
    queries: list[tuple[str, str]] = []
    if args.node_file:
        if args.node_placeholder not in query:
            raise SystemExit(f"--node-placeholder {args.node_placeholder!r} not found in query")
        nodes = read_node_ids(args.node_file, args.node_column)
        for index, chunk in enumerate(chunks(nodes, args.node_chunk_size), start=1):
            replaced = query.replace(args.node_placeholder, regex_for_nodes(chunk))
            queries.append((f"chunk_{index}", replaced))
    else:
        queries.append(("query", query))

    rows: list[dict[str, Any]] = []
    for batch_name, batch_query in queries:
        endpoint, params = query_params(args, batch_query)
        payload = api_post(args.base_url, endpoint, params, args.timeout)
        batch_rows = flatten_query_result(payload, batch_name)
        rows.extend(batch_rows)
    return rows


def run_metadata_command(args: argparse.Namespace) -> list[dict[str, Any]]:
    params = time_params(args)
    for matcher in getattr(args, "match", []) or []:
        for metric in HIGH_CARDINALITY_METRICS:
            if metric in matcher and not args.allow_wide_query and not any(hint in matcher for hint in SAFE_LABEL_HINTS):
                raise SystemExit(
                    f"Refusing broad metadata lookup on high-cardinality metric {metric}. "
                    "Add a narrow matcher or use --allow-wide-query."
                )
        params.append(("match[]", matcher))
    if args.command == "labels":
        payload = api_post(args.base_url, "labels", params, args.timeout)
        return [{"label": value} for value in payload.get("data", [])]
    if args.command == "label-values":
        safe_label = urllib.parse.quote(args.label, safe="")
        payload = api_post(args.base_url, f"label/{safe_label}/values", params, args.timeout)
        return [{args.label: value} for value in payload.get("data", [])]
    if args.command == "series":
        payload = api_post(args.base_url, "series", params, args.timeout)
        return [dict(item) for item in payload.get("data", [])]
    raise AssertionError(args.command)


def flatten_query_result(payload: dict[str, Any], batch_name: str) -> list[dict[str, Any]]:
    data = payload.get("data", {})
    result_type = data.get("resultType")
    result = data.get("result", [])
    rows: list[dict[str, Any]] = []
    if result_type == "vector":
        for item in result:
            row = row_from_metric(item.get("metric", {}), item.get("value"), batch_name, result_type)
            if row:
                rows.append(row)
    elif result_type == "matrix":
        for item in result:
            metric = item.get("metric", {})
            for value in item.get("values", []):
                row = row_from_metric(metric, value, batch_name, result_type)
                if row:
                    rows.append(row)
    elif result_type in {"scalar", "string"}:
        row = row_from_metric({}, result, batch_name, result_type)
        if row:
            rows.append(row)
    else:
        for item in result:
            rows.append({"batch": batch_name, "result_type": result_type, "raw": json.dumps(item, ensure_ascii=False)})
    return rows


def row_from_metric(metric: dict[str, Any], sample: Any, batch_name: str, result_type: str) -> dict[str, Any] | None:
    if not isinstance(sample, list) or len(sample) < 2:
        return None
    timestamp = float(sample[0])
    raw_value = sample[1]
    try:
        value: float | str = float(raw_value)
    except (TypeError, ValueError):
        value = str(raw_value)
    row: dict[str, Any] = {
        "batch": batch_name,
        "result_type": result_type,
        "timestamp": timestamp,
        "time_utc": fmt_time(timestamp, UTC),
        "time_bj": fmt_time(timestamp, BEIJING_TZ),
        "value": value,
    }
    row.update(metric)
    return row


def sort_rows(rows: list[dict[str, Any]], sort_desc: bool) -> list[dict[str, Any]]:
    if not sort_desc:
        return rows
    return sorted(rows, key=lambda row: row.get("value") if isinstance(row.get("value"), (int, float)) else float("-inf"), reverse=True)


def write_output(rows: list[dict[str, Any]], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        return
    if path.suffix.lower() != ".csv":
        raise SystemExit("--output suffix must be .csv or .json")
    columns = collect_columns(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["batch", "result_type", "timestamp", "time_utc", "time_bj", "value", "__name__", "node_id", "node", "customer_id", "customer_zh", "province", "isp"]
    seen = set()
    columns = []
    for column in preferred:
        if any(column in row for row in rows):
            columns.append(column)
            seen.add(column)
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def print_preview(rows: list[dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        return
    if not rows:
        print("No data.")
        return
    preview = rows[:limit]
    columns = collect_columns(preview)
    print("\t".join(columns))
    for row in preview:
        print("\t".join(format_cell(row.get(column, "")) for column in columns))
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows not shown")


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> int:
    args = parse_args()
    if args.command in {"instant", "range"}:
        rows = run_query_command(args)
    else:
        rows = run_metadata_command(args)
    rows = sort_rows(rows, args.sort_desc)
    if args.output:
        write_output(rows, args.output)
        print(f"Wrote {len(rows)} rows to {args.output}")
    print_preview(rows, args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
