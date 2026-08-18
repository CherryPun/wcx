#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = ROOT / "common"
import sys

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from superset_api import SupersetSQLClient, env_or_die


CN_TZ = timezone(timedelta(hours=8))
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "references" / "full-schema.md"


@dataclass
class ScalarType:
    text: str


@dataclass
class RowType:
    fields: list[tuple[str, Any]]


@dataclass
class ArrayType:
    item_type: Any


@dataclass
class MapType:
    key_type: Any
    value_type: Any


class TypeParser:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        self.pos = 0

    def parse(self) -> Any:
        parsed = self.parse_type()
        self.skip_ws()
        return parsed

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def startswith(self, token: str) -> bool:
        return self.text.startswith(token, self.pos)

    def consume(self, token: str) -> None:
        if not self.startswith(token):
            raise ValueError(f"Expected {token!r} at position {self.pos} in {self.text!r}")
        self.pos += len(token)

    def parse_identifier(self) -> str:
        self.skip_ws()
        start = self.pos
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch.isalnum() or ch in {"_", "$"}:
                self.pos += 1
            else:
                break
        if self.pos == start:
            raise ValueError(f"Expected identifier at position {self.pos} in {self.text!r}")
        return self.text[start:self.pos]

    def parse_scalar(self) -> ScalarType:
        start = self.pos
        depth = 0
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
            self.pos += 1
        return ScalarType(self.text[start:self.pos].strip())

    def parse_type(self) -> Any:
        self.skip_ws()
        if self.startswith("row("):
            self.consume("row(")
            fields: list[tuple[str, Any]] = []
            while True:
                self.skip_ws()
                if self.pos < len(self.text) and self.text[self.pos] == ")":
                    self.pos += 1
                    break
                field_name = self.parse_identifier()
                self.skip_ws()
                field_type = self.parse_type()
                fields.append((field_name, field_type))
                self.skip_ws()
                if self.pos < len(self.text) and self.text[self.pos] == ",":
                    self.pos += 1
                    continue
                if self.pos < len(self.text) and self.text[self.pos] == ")":
                    self.pos += 1
                    break
            return RowType(fields)
        if self.startswith("array("):
            self.consume("array(")
            item_type = self.parse_type()
            self.skip_ws()
            self.consume(")")
            return ArrayType(item_type)
        if self.startswith("map("):
            self.consume("map(")
            key_type = self.parse_type()
            self.skip_ws()
            self.consume(",")
            value_type = self.parse_type()
            self.skip_ws()
            self.consume(")")
            return MapType(key_type, value_type)
        return self.parse_scalar()


def render_type(type_node: Any) -> str:
    if isinstance(type_node, ScalarType):
        return type_node.text
    if isinstance(type_node, RowType):
        return "row(...)"
    if isinstance(type_node, ArrayType):
        return f"array({render_type(type_node.item_type)})"
    if isinstance(type_node, MapType):
        return f"map({render_type(type_node.key_type)}, {render_type(type_node.value_type)})"
    return str(type_node)


def flatten_paths(prefix: str, type_node: Any) -> list[tuple[str, str]]:
    if isinstance(type_node, ScalarType):
        return [(prefix, type_node.text)]
    if isinstance(type_node, RowType):
        flattened: list[tuple[str, str]] = []
        for field_name, field_type in type_node.fields:
            flattened.extend(flatten_paths(f"{prefix}.{field_name}", field_type))
        return flattened
    if isinstance(type_node, ArrayType):
        flattened = [(prefix, render_type(type_node))]
        if isinstance(type_node.item_type, RowType):
            for field_name, field_type in type_node.item_type.fields:
                flattened.extend(flatten_paths(f"{prefix}[].{field_name}", field_type))
        return flattened
    if isinstance(type_node, MapType):
        flattened = [(prefix, render_type(type_node))]
        if isinstance(type_node.value_type, RowType):
            for field_name, field_type in type_node.value_type.fields:
                flattened.extend(flatten_paths(f"{prefix}{{}}.{field_name}", field_type))
        return flattened
    return [(prefix, str(type_node))]


def query_columns() -> list[dict[str, Any]]:
    sql = """
SELECT
  table_name,
  ordinal_position,
  column_name,
  data_type
FROM information_schema.columns
WHERE table_schema = 'jarvis'
ORDER BY table_name, ordinal_position
""".strip()
    client = SupersetSQLClient(
        username=env_or_die("SUPERSET_USERNAME"),
        password=env_or_die("SUPERSET_PASSWORD"),
    )
    return client.execute_sql(sql=sql, query_limit=5000, tab="全量Schema字典", sql_editor_id="schema_ref")


def build_markdown(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["table_name"]), []).append(row)

    now_label = datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Full Schema")
    lines.append("")
    lines.append(f"生成时间：`{now_label}` `Asia/Shanghai`")
    lines.append("")
    lines.append("这份文档按 `jarvis` schema 自动生成，目标是补全当前可见表的字段清单。")
    lines.append("")
    lines.append("## Tables Overview")
    lines.append("")
    lines.append("| table | columns | note |")
    lines.append("|---|---:|---|")
    for table_name in sorted(grouped):
        table_rows = grouped[table_name]
        has_nested = any(str(row["data_type"]).startswith("row(") for row in table_rows)
        note = "contains nested row fields" if has_nested else "flat columns"
        lines.append(f"| `{table_name}` | {len(table_rows)} | {note} |")

    for table_name in sorted(grouped):
        table_rows = grouped[table_name]
        lines.append("")
        lines.append(f"## `{table_name}`")
        lines.append("")
        lines.append(f"列数：`{len(table_rows)}`")
        lines.append("")
        lines.append("### Direct Columns")
        lines.append("")
        for row in table_rows:
            column_name = str(row["column_name"])
            data_type = str(row["data_type"])
            lines.append(f"- `{column_name}`: `{data_type}`")

        nested_entries: list[tuple[str, str]] = []
        for row in table_rows:
            column_name = str(row["column_name"])
            data_type = str(row["data_type"])
            if data_type.startswith("row("):
                try:
                    parsed = TypeParser(data_type).parse()
                    nested_entries.extend(flatten_paths(column_name, parsed))
                except Exception:
                    nested_entries.append((column_name, data_type))

        if nested_entries:
            lines.append("")
            lines.append("### Expanded Nested Paths")
            lines.append("")
            for path, dtype in nested_entries:
                lines.append(f"- `{path}`: `{dtype}`")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `node_analysis_data.day/hour` 是 `varchar`")
    lines.append("- `node_join.day/hour` 是 `integer`")
    lines.append("- 常见关联键：`node_analysis_data.nodeid = node_join._id`")
    lines.append("- 需要按天 join 时，注意做类型转换")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a full jarvis schema markdown reference.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = query_columns()
    output_path.write_text(build_markdown(rows), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
