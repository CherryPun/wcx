#!/usr/bin/env python3
"""Look up modeled nodes and precomputed V1 business recommendations."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RECOMMENDATIONS = HERE / "multibusiness_recommendations.csv"
DEFAULT_NODES = HERE / "multibusiness_nodes.csv"
DEFAULT_OUTCOMES = HERE / "multibusiness_outcomes.csv"
DEFAULT_FIELD_CONTRACT = HERE / "node_profile_v1_fields.json"

SUMMARY_FIELDS = [
    "node_id",
    "province",
    "city",
    "isp",
    "nattype",
    "dialtype",
    "bandwidth",
    "bw",
    "actualbandwidth",
    "recommendation_status",
    "combined_top1",
    "combined_top3",
    "miner_best_provisional",
    "operator_best_provisional",
    "feasible_business_count",
    "blocked_business_count",
    "split",
]

FIELD_ALIASES = {
    "节点": "node_id",
    "节点id": "node_id",
    "节点ID": "node_id",
    "node": "node_id",
    "省": "province",
    "省份": "province",
    "城市": "city",
    "市": "city",
    "运营商": "isp",
    "nat": "nattype",
    "NAT": "nattype",
    "拨号": "dialtype",
    "资源类型": "resourcetype",
    "硬件": "hardwaretype",
    "带宽": "bw",
    "名义带宽": "bw",
    "实际带宽": "actualbandwidth",
    "实测带宽": "actualbandwidth",
    "状态": "recommendation_status",
    "推荐状态": "recommendation_status",
    "矿主推荐": "miner_best_provisional",
    "运营推荐": "operator_best_provisional",
    "综合推荐": "combined_top1",
}

OPERATORS = (">=", "<=", "!=", "~", ">", "<", "=")


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_field(field: str) -> str:
    field = field.strip()
    return FIELD_ALIASES.get(field, field)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def maybe_float(value: Any) -> float | None:
    text = clean_cell(value)
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_condition(text: str) -> tuple[str, str, str]:
    text = text.strip()
    for operator in OPERATORS:
        if operator in text:
            field, value = text.split(operator, 1)
            field = normalize_field(field)
            value = clean_cell(value)
            if not field or value == "":
                raise ValueError(f"invalid condition: {text}")
            return field, operator, value
    raise ValueError(
        f"invalid condition: {text}. Use forms like isp=移动, province=安徽, bandwidth>=1000, recommendation_status=needs_manual_check."
    )


def matches_condition(row: dict[str, str], condition: tuple[str, str, str]) -> bool:
    field, operator, expected = condition
    actual = clean_cell(row.get(field))
    if operator == "=":
        return actual == expected
    if operator == "!=":
        return actual != expected
    if operator == "~":
        return expected in actual

    actual_number = maybe_float(actual)
    expected_number = maybe_float(expected)
    if actual_number is not None and expected_number is not None:
        left: float | str = actual_number
        right: float | str = expected_number
    else:
        left = actual
        right = expected
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    raise ValueError(f"unsupported operator: {operator}")


def read_first_by_node(path: Path, node_id: str) -> dict[str, str] | None:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if clean_cell(row.get("node_id")) == node_id:
                return {key: clean_cell(value) for key, value in row.items()}
    return None


def load_field_contract(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_recommendation_index(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    output: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {key: clean_cell(value) for key, value in row.items()}
            node_id = cleaned.get("node_id", "")
            if node_id:
                output[node_id] = cleaned
    return output


def read_outcomes(path: Path, node_id: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if clean_cell(row.get("node_id")) == node_id:
                rows.append({key: clean_cell(value) for key, value in row.items()})
    return rows


def parse_top3(value: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in clean_cell(value).split(";"):
        if not item or ":" not in item:
            continue
        business, score = item.split(":", 1)
        output.append({"business": clean_cell(business), "score": to_float(score)})
    return output


def combine_top3(
    miner_top3: list[dict[str, Any]],
    operator_top3: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    scores: dict[str, dict[str, float]] = {}
    for item in miner_top3:
        business = clean_cell(item.get("business"))
        if business:
            scores.setdefault(business, {"miner_score": 0.0, "operator_score": 0.0})
            scores[business]["miner_score"] = float(item.get("score", 0.0))
    for item in operator_top3:
        business = clean_cell(item.get("business"))
        if business:
            scores.setdefault(business, {"miner_score": 0.0, "operator_score": 0.0})
            scores[business]["operator_score"] = float(item.get("score", 0.0))

    combined = []
    for business, score in scores.items():
        combined_score = 0.5 * score["miner_score"] + 0.5 * score["operator_score"]
        combined.append({
            "business": business,
            "combined_score": combined_score,
            "miner_score": score["miner_score"],
            "operator_score": score["operator_score"],
        })
    return sorted(combined, key=lambda row: (-row["combined_score"], row["business"]))[:top_k]


def combined_top3_text(row: dict[str, str]) -> str:
    combined = combine_top3(
        parse_top3(row.get("miner_top3", "")),
        parse_top3(row.get("operator_top3", "")),
    )
    return ";".join(f"{item['business']}:{item['combined_score']:.6g}" for item in combined)


def summarize_outcomes(rows: list[dict[str, str]], limit: int) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        cost = to_float(row.get("cum_cost_7d"))
        revenue = to_float(row.get("cum_revenue_7d"))
        enriched.append({
            "business": clean_cell(row.get("business")),
            "business_name": clean_cell(row.get("business_name")),
            "cum_cost_7d": cost,
            "cum_revenue_7d": revenue,
            "cum_profit_7d": revenue - cost,
            "outcome_distinct_days": clean_cell(row.get("outcome_distinct_days")),
        })
    return {
        "count": len(enriched),
        "top_cost": sorted(enriched, key=lambda row: (-row["cum_cost_7d"], row["business"]))[:limit],
        "top_profit": sorted(enriched, key=lambda row: (-row["cum_profit_7d"], row["business"]))[:limit],
    }


def compact_blocked_reasons(value: str, limit: int) -> list[str]:
    if limit <= 0:
        return []
    reasons = [part for part in clean_cell(value).split(";") if part]
    return reasons[:limit]


def format_field_value(field: dict[str, Any], value: str) -> str:
    text = clean_cell(value)
    if text == "":
        return ""
    field_type = clean_cell(field.get("type"))
    if field_type.startswith("numeric"):
        number = maybe_float(text)
        if number is None:
            return text
        return f"{number:.6g}"
    return text


def profile_from_contract(
    node: dict[str, str] | None,
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    if node is None:
        return {}
    if not contract:
        return {
            "legacy": {
                "label": "节点字段",
                "fields": [
                    {"field": key, "label": key, "value": value, "missing": value == ""}
                    for key, value in node.items()
                    if value != ""
                ],
                "missing_fields": [],
            }
        }

    profile: dict[str, Any] = {}
    for group in contract.get("groups", []):
        group_name = clean_cell(group.get("name"))
        entries = []
        missing = []
        for field in group.get("fields", []):
            field_name = clean_cell(field.get("field"))
            if field_name not in node:
                continue
            value = format_field_value(field, node.get(field_name, ""))
            item = {
                "field": field_name,
                "label": clean_cell(field.get("label")) or field_name,
                "value": value,
                "missing": value == "",
                "display": bool(field.get("display", True)),
                "new_node_model": field.get("new_node_model"),
                "existing_node_model": field.get("existing_node_model"),
                "missing_policy": clean_cell(field.get("missing_policy")),
            }
            entries.append(item)
            if value == "" and item["missing_policy"] not in {"allow_missing", "mostly_missing_display_only"}:
                missing.append(field_name)
        profile[group_name] = {
            "label": clean_cell(group.get("description")),
            "fields": entries,
            "missing_fields": missing,
        }
    return profile


def risk_summary(profile: dict[str, Any], recommendation: dict[str, Any] | None) -> dict[str, Any]:
    missing = []
    observed = []
    for group_name in ["network_shape_optional", "network_quality_risk"]:
        group = profile.get(group_name, {})
        missing.extend(group.get("missing_fields", []))
        for field in group.get("fields", []):
            if field.get("value") != "":
                observed.append({
                    "field": field["field"],
                    "label": field["label"],
                    "value": field["value"],
                })
    return {
        "missing_risk_fields": sorted(set(missing)),
        "observed_risk_values": observed,
        "blocked_reasons_sample": (recommendation or {}).get("blocked_reasons_sample", []),
    }


def merged_search_rows(
    nodes_path: Path,
    recommendations_path: Path,
    conditions: list[tuple[str, str, str]],
    limit: int,
) -> tuple[list[dict[str, str]], int]:
    recommendations = read_recommendation_index(recommendations_path)
    rows: list[dict[str, str]] = []
    total = 0
    with nodes_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for node_row in reader:
            node = {key: clean_cell(value) for key, value in node_row.items()}
            recommendation = recommendations.get(node.get("node_id", ""), {})
            merged = {**node, **recommendation}
            if all(matches_condition(merged, condition) for condition in conditions):
                total += 1
                if len(rows) < limit:
                    rows.append(merged)
    return rows, total


def summary_row(row: dict[str, str]) -> dict[str, str]:
    enriched = dict(row)
    combined = combined_top3_text(row)
    enriched["combined_top3"] = combined
    enriched["combined_top1"] = combined.split(":", 1)[0] if combined else ""
    return {field: clean_cell(enriched.get(field)) for field in SUMMARY_FIELDS}


def build_search_payload(args: argparse.Namespace, conditions: list[tuple[str, str, str]]) -> dict[str, Any]:
    if not args.nodes.exists():
        raise FileNotFoundError(args.nodes)
    rows, total = merged_search_rows(args.nodes, args.recommendations, conditions, args.limit)
    summaries = [summary_row(row) for row in rows]
    return {
        "found": total > 0,
        "conditions": [
            {"field": field, "operator": operator, "value": value}
            for field, operator, value in conditions
        ],
        "total_matches": total,
        "returned": len(summaries),
        "rows": summaries,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    node_id = clean_cell(args.node_id)
    recommendation = read_first_by_node(args.recommendations, node_id)
    node = read_first_by_node(args.nodes, node_id)
    outcomes = summarize_outcomes(read_outcomes(args.outcomes, node_id), args.top)
    contract = load_field_contract(args.field_contract)

    if recommendation is None and node is None and outcomes["count"] == 0:
        return {
            "node_id": node_id,
            "found": False,
            "message": "node_id was not found in current local artifacts",
        }

    profile = profile_from_contract(node, contract)
    recommendation_summary: dict[str, Any] | None = None
    if recommendation:
        miner_top3 = parse_top3(recommendation.get("miner_top3", ""))
        operator_top3 = parse_top3(recommendation.get("operator_top3", ""))
        combined_top3 = combine_top3(miner_top3, operator_top3)
        recommendation_summary = {
            "status": recommendation.get("recommendation_status", ""),
            "official_miner_best": recommendation.get("miner_best", ""),
            "official_operator_best": recommendation.get("operator_best", ""),
            "provisional_miner_best": recommendation.get("miner_best_provisional", ""),
            "provisional_operator_best": recommendation.get("operator_best_provisional", ""),
            "feasible_business_count": recommendation.get("feasible_business_count", ""),
            "blocked_business_count": recommendation.get("blocked_business_count", ""),
            "combined_top3": combined_top3,
            "combined_top1": combined_top3[0]["business"] if combined_top3 else "",
            "combined_score_note": "computed from available miner_top3/operator_top3 scores only",
            "miner_top3": miner_top3,
            "operator_top3": operator_top3,
            "split": recommendation.get("split", ""),
            "blocked_reasons_sample": compact_blocked_reasons(
                recommendation.get("blocked_business_reasons", ""),
                args.blocked_limit,
            ),
        }

    return {
        "node_id": node_id,
        "found": True,
        "profile": profile,
        "risk": risk_summary(profile, recommendation_summary),
        "recommendation": recommendation_summary,
        "historical_outcomes": outcomes,
    }


def print_search(payload: dict[str, Any]) -> None:
    conditions = [
        f"{item['field']}{item['operator']}{item['value']}"
        for item in payload["conditions"]
    ]
    print("conditions: " + ", ".join(conditions))
    print(f"total_matches: {payload['total_matches']}")
    print(f"returned: {payload['returned']}")
    rows = payload.get("rows") or []
    if not rows:
        return
    fields = [field for field in SUMMARY_FIELDS if any(row.get(field) for row in rows)]
    widths = {
        field: min(
            max(len(field), *(len(clean_cell(row.get(field))) for row in rows)),
            32,
        )
        for field in fields
    }

    def short(value: str, width: int) -> str:
        value = clean_cell(value)
        if len(value) <= width:
            return value
        return value[: max(width - 1, 0)] + "…"

    print("")
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  ".join(short(row.get(field, ""), widths[field]).ljust(widths[field]) for field in fields))


def print_text(payload: dict[str, Any]) -> None:
    print(f"node_id: {payload['node_id']}")
    if not payload.get("found"):
        print(payload["message"])
        return

    profile = payload.get("profile") or {}
    if profile:
        sections = [
            ("identity", "基础身份"),
            ("common_profile", "共性画像"),
            ("network_shape_optional", "网络形态"),
            ("network_quality_risk", "网络质量/风险"),
        ]
        for group_name, title in sections:
            group = profile.get(group_name)
            if not group:
                continue
            visible = [field for field in group.get("fields", []) if field.get("display") and field.get("value") != ""]
            if not visible:
                continue
            print(f"\n[{title}]")
            for field in visible:
                print(f"{field['label']}({field['field']}): {field['value']}")
            missing = group.get("missing_fields", [])
            if missing:
                shown = ", ".join(missing[:8])
                suffix = " ..." if len(missing) > 8 else ""
                print(f"信息不足: {shown}{suffix}")
    else:
        print("\n[node]\nnot found in multibusiness_nodes.csv")

    recommendation = payload.get("recommendation")
    if recommendation:
        print("\n[recommendation]")
        print(f"status: {recommendation['status']}")
        print(f"split: {recommendation['split']}")
        print(f"feasible_business_count: {recommendation['feasible_business_count']}")
        print(f"blocked_business_count: {recommendation['blocked_business_count']}")
        official_miner = recommendation["official_miner_best"] or "(empty)"
        official_operator = recommendation["official_operator_best"] or "(empty)"
        print(f"official_miner_best: {official_miner}")
        print(f"official_operator_best: {official_operator}")
        print(f"provisional_miner_best: {recommendation['provisional_miner_best'] or '(empty)'}")
        print(f"provisional_operator_best: {recommendation['provisional_operator_best'] or '(empty)'}")
        print("combined_top3:")
        for row in recommendation["combined_top3"]:
            print(
                f"  {row['business']}: combined={row['combined_score']:.6g}, "
                f"miner={row['miner_score']:.6g}, operator={row['operator_score']:.6g}"
            )
        if recommendation["combined_top3"]:
            print(f"combined_score_note: {recommendation['combined_score_note']}")
        print("miner_top3:")
        for row in recommendation["miner_top3"]:
            print(f"  {row['business']}: {row['score']:.6g}")
        print("operator_top3:")
        for row in recommendation["operator_top3"]:
            print(f"  {row['business']}: {row['score']:.6g}")
        if recommendation["blocked_reasons_sample"]:
            print("blocked_reasons_sample:")
            for reason in recommendation["blocked_reasons_sample"]:
                print(f"  {reason}")
    else:
        print("\n[recommendation]\nnot found in multibusiness_recommendations.csv")

    outcomes = payload.get("historical_outcomes") or {}
    print("\n[historical_outcomes]")
    print(f"observed_business_count: {outcomes.get('count', 0)}")
    for label, rows in [("top_cost", outcomes.get("top_cost", [])), ("top_profit", outcomes.get("top_profit", []))]:
        print(f"{label}:")
        for row in rows:
            name = f" ({row['business_name']})" if row.get("business_name") else ""
            print(
                f"  {row['business']}{name}: "
                f"cost={row['cum_cost_7d']:.4g}, "
                f"revenue={row['cum_revenue_7d']:.4g}, "
                f"profit={row['cum_profit_7d']:.4g}, "
                f"days={row['outcome_distinct_days']}"
            )

    risk = payload.get("risk") or {}
    if risk:
        print("\n[risk_summary]")
        missing = risk.get("missing_risk_fields") or []
        print(f"missing_risk_field_count: {len(missing)}")
        if missing:
            print("missing_risk_fields: " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
        observed = risk.get("observed_risk_values") or []
        if observed:
            print("observed_risk_values:")
            for row in observed[:12]:
                print(f"  {row['label']}({row['field']}): {row['value']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Look up modeled nodes in local multibusiness recommendation artifacts.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help=(
            "Either one node_id, or conditions such as isp=移动 province=安徽 "
            "bandwidth>=1000 recommendation_status=needs_manual_check. "
            "Use ~=contains for substring matching."
        ),
    )
    parser.add_argument("--node-id", help="Node ID to look up.")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        help="Add a filter condition. Can be repeated, e.g. --where isp=移动 --where province=安徽.",
    )
    parser.add_argument("--recommendations", type=Path, default=DEFAULT_RECOMMENDATIONS)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--field-contract", type=Path, default=DEFAULT_FIELD_CONTRACT)
    parser.add_argument("--top", type=int, default=3, help="Number of historical outcomes to show.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to return for condition queries.")
    parser.add_argument(
        "--blocked-limit",
        type=int,
        default=3,
        help="Number of blocked business reasons to show as a sample.",
    )
    parser.add_argument("--csv", type=Path, help="Write condition-query results to this CSV file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def resolve_query(args: argparse.Namespace) -> tuple[str | None, list[tuple[str, str, str]]]:
    if args.node_id:
        if args.query:
            raise ValueError("use either --node-id or positional query terms, not both")
        return clean_cell(args.node_id), [parse_condition(item) for item in args.where]

    terms = list(args.where) + list(args.query)
    if not terms:
        raise ValueError("provide a node_id or at least one condition")

    if len(terms) == 1 and not any(operator in terms[0] for operator in OPERATORS):
        return clean_cell(terms[0]), []
    return None, [parse_condition(item) for item in terms]


def write_search_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(SUMMARY_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field)) for field in fields})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    node_id, conditions = resolve_query(args)
    if node_id and not conditions:
        args.node_id = node_id
        payload = build_payload(args)
        if args.csv:
            raise ValueError("--csv is only supported for condition queries")
        is_found = bool(payload.get("found"))
    else:
        if node_id:
            conditions.insert(0, ("node_id", "=", node_id))
        payload = build_search_payload(args, conditions)
        if args.csv:
            write_search_csv(args.csv, payload.get("rows") or [])
        is_found = bool(payload.get("found"))

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif "rows" in payload:
        print_search(payload)
        if args.csv:
            print(f"\nwrote: {args.csv}")
    else:
        print_text(payload)
    return 0 if is_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
