#!/usr/bin/env python3
"""Scan current online in-service non-IDC large nodes for business mismatches."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

import build_v3_daily_business_training as v3
import rebuild_multibusiness_model as rebuild
import v1_recommendation_pipeline as v1
import v2_ranking_model as v2


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "current_non_idc_large_scan_network_scope"
DEFAULT_MODEL = (
    HERE
    / "recent_month_large_mainstream_1d_unit_bw_network_scope"
    / "v2_large_mainstream_outputs"
    / "v2_ranking_model.json"
)
DEFAULT_ALLOWLIST = HERE / "mainstream_business_allowlist.csv"
DEFAULT_BUSINESS_MAP = HERE / "v1_business_name_map_enriched.csv"


def clean(value: Any) -> str:
    return v1.clean_cell(value)


def parse_yyyymmdd(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y%m%d").date()


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def load_business_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {
        clean(row.business): clean(row.business_name)
        for row in frame.itertuples(index=False)
        if clean(row.business) and clean(row.business_name)
    }


def load_allowlist(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["business"] = frame["business"].map(clean)
    frame["business_name"] = frame["business_name"].map(clean)
    return frame


def latest_partitions() -> dict[str, str]:
    wide = rebuild.run_sql(
        "SELECT CAST(MAX(day) AS VARCHAR) AS max_day FROM node_day_ops_wide_full WHERE day >= DATE '2026-08-01'",
        database_id=19,
        schema="test",
    )[0]["max_day"]
    join = rebuild.run_sql(
        "SELECT CAST(MAX(day) AS VARCHAR) AS max_day FROM node_join WHERE day >= 20260801",
        database_id=2,
        schema="jarvis",
    )[0]["max_day"]
    analysis = rebuild.run_sql(
        "SELECT CAST(MAX(day) AS VARCHAR) AS max_day FROM node_analysis_data WHERE day >= '20260801'",
        database_id=2,
        schema="jarvis",
    )[0]["max_day"]
    dial = rebuild.run_sql(
        "SELECT CAST(MAX(day) AS VARCHAR) AS max_day FROM dial_acc WHERE day >= 20260801",
        database_id=2,
        schema="jarvis",
    )[0]["max_day"]
    return {
        "wide_day": str(wide),
        "join_day": str(join),
        "analysis_day": str(analysis),
        "dial_day": str(dial),
    }


def current_candidate_sql(analysis_day: str, prefix: str) -> str:
    return f"""
    WITH ranked AS (
      SELECT
        nd.nodeid AS node_id,
        nd.state AS state,
        nd.stage AS stage,
        COALESCE(CAST(nd.supplysidedeliverytype AS VARCHAR), '') AS supply_type,
        LOWER(COALESCE(CAST(nd.nodedeliverytype AS VARCHAR), '')) AS node_delivery,
        LOWER(COALESCE(CAST(nd.resourcetype AS VARCHAR), '')) AS resourcetype,
        TRY_CAST(nd.bandwidth AS DOUBLE) AS bandwidth,
        TRY_CAST(nd.actualbandwidth AS DOUBLE) AS actualbandwidth,
        TRY_CAST(nd.corenum AS DOUBLE) AS corenum,
        ROW_NUMBER() OVER (
          PARTITION BY nd.nodeid
          ORDER BY TRY_CAST(nd.hour AS INTEGER) DESC, nd.time DESC
        ) AS rn
      FROM node_analysis_data nd
      WHERE nd.day = {rebuild.sql_quote(analysis_day)}
        AND SUBSTR(nd.nodeid, 1, 1) = {rebuild.sql_quote(prefix)}
    )
    SELECT node_id
    FROM ranked
    WHERE rn = 1
      AND state = 'online'
      AND stage = 'inService'
      AND COALESCE(supply_type, '') NOT IN ('机房', '小盒子')
      AND node_delivery NOT IN ('droid', 'h618', 'aml', 'n1', 'openwrt')
      AND (
        supply_type IN ('汇聚', '专线')
        OR resourcetype IN ('aggregation', 'dedicated')
        OR GREATEST(COALESCE(bandwidth, 0), COALESCE(actualbandwidth, 0)) >= 100
        OR (
          GREATEST(COALESCE(bandwidth, 0), COALESCE(actualbandwidth, 0)) >= 1000
          AND COALESCE(corenum, 0) >= 8
        )
      )
    """


def discover_current_candidates(analysis_day: str, output_dir: Path, refresh: bool) -> list[str]:
    path = output_dir / f"current_non_idc_large_candidate_node_ids_{analysis_day}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))["node_ids"]

    prefixes = list("0123456789abcdef")
    parts: dict[str, list[str]] = {}

    def fetch(prefix: str) -> tuple[str, list[str]]:
        rows = rebuild.run_sql(current_candidate_sql(analysis_day, prefix), database_id=2, schema="jarvis")
        return prefix, sorted(clean(row.get("node_id")) for row in rows if clean(row.get("node_id")))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, prefix) for prefix in prefixes]
        for future in concurrent.futures.as_completed(futures):
            prefix, node_ids = future.result()
            parts[prefix] = node_ids
            print(f"candidate prefix {prefix}: {len(node_ids):,}")

    node_ids = sorted({node_id for values in parts.values() for node_id in values})
    payload = {
        "analysis_day": analysis_day,
        "candidate_rule": (
            "node_analysis_data latest snapshot: state=online, stage=inService, "
            "supplysidedeliverytype not in 机房/小盒子, and large-node capacity/resource rule."
        ),
        "prefix_counts": {prefix: len(parts.get(prefix, [])) for prefix in prefixes},
        "node_ids": node_ids,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return node_ids


def fetch_current_node_attributes(
    node_ids: list[str],
    analysis_day: str,
    output_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    path = output_dir / f"current_non_idc_large_node_attrs_{analysis_day}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"node_id": "string"}, low_memory=False)

    target_day = parse_yyyymmdd(analysis_day)
    online_day = (target_day + dt.timedelta(days=1)).isoformat()
    targets = pd.DataFrame({"node_id": node_ids, "online_day": online_day})
    old_output_nodes = rebuild.OUTPUT_NODES
    try:
        rebuild.OUTPUT_NODES = path
        nodes = rebuild.fetch_attributes(targets)
    finally:
        rebuild.OUTPUT_NODES = old_output_nodes
    return nodes


def keep_current_non_idc_large(nodes: pd.DataFrame) -> pd.DataFrame:
    frame = v2.add_buckets(nodes)
    for column in [
        "node_state",
        "node_status",
        "node_analysis_stage",
        "node_stage",
        "analysis_supply_side_delivery_type",
        "deliverytype",
    ]:
        if column not in frame.columns:
            frame[column] = ""
    online = (
        frame["node_state"].map(clean).eq("online")
        | frame["node_status"].map(clean).eq("online")
    )
    in_service = (
        frame["node_analysis_stage"].map(clean).eq("inService")
        | frame["node_stage"].map(clean).eq("inService")
    )
    non_idc = (
        frame["analysis_supply_side_delivery_type"].map(clean).ne("机房")
        & frame["analysis_supply_side_delivery_type"].map(clean).ne("小盒子")
        & frame["deliverytype"].map(lambda value: clean(value).lower()).ne("smallbox")
    )
    large = frame["node_size_type"].map(clean).eq(v2.NODE_SIZE_LARGE)
    return frame[online & in_service & non_idc & large].copy()


def current_business_sql(node_ids: list[str], start_day: dt.date, end_day: dt.date) -> str:
    ids = ",\n".join(rebuild.sql_quote(node_id) for node_id in node_ids)
    return f"""
    SELECT
      nodeId,
      customerId,
      customerName,
      day,
      state,
      stage,
      buildBandwidth,
      province,
      city,
      isp,
      natType,
      resourceType,
      deliveryType,
      transProvRate,
      scheduleISPs,
      vendorSuggestCustomersName,
      virtualCustomersName,
      peak95,
      revenue_finalAmount,
      cost_finalAmount,
      profit_profitAmount
    FROM node_day_ops_wide_full
    WHERE day BETWEEN DATE '{start_day.isoformat()}' AND DATE '{end_day.isoformat()}'
      AND customerId > 0
      AND nodeId IN ({ids})
    """


def derive_current_business_v3(
    raw: pd.DataFrame,
    allowlist: pd.DataFrame,
    business_name_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve current business from one canonical active business per node-day."""
    _, audit, annotated = v3.build_daily_facts(raw, allowlist, business_name_map)
    if audit.empty:
        return pd.DataFrame(), audit
    accepted_statuses = {
        "clean",
        "excluded_unattributed_financial_rows",
        "excluded_zero_financial_outcome",
        "excluded_missing_build_bandwidth",
        "excluded_inconsistent_build_bandwidth",
        "excluded_invalid_transprov_rate",
        "excluded_inconsistent_transprov_rate",
    }
    current = audit[
        audit["status"].isin(accepted_statuses)
        & audit["selected_business"].fillna("").map(clean).ne("")
    ].copy()
    if current.empty:
        return pd.DataFrame(), audit
    current = current.sort_values("sample_day").drop_duplicates("node_id", keep="last")
    peak = annotated[annotated["is_attributed"]].copy()
    peak["peak95"] = pd.to_numeric(peak.get("peak95"), errors="coerce")
    peak = peak.groupby(["node_id", "sample_day"], dropna=False)["peak95"].max().rename("current_business_peak95_max")
    current = current.merge(peak, left_on=["node_id", "sample_day"], right_index=True, how="left")
    resolved = pd.DataFrame({
        "node_id": current["node_id"].map(clean),
        "current_business": current["selected_business"].map(clean),
        "current_business_name": current["selected_business_name"].fillna("").map(clean),
        "current_business_day": current["sample_day"].map(clean),
        "current_business_active_days": 1,
        "current_business_revenue_sum": current["attributed_revenue_finalAmount"],
        "current_business_cost_sum": current["attributed_cost_finalAmount"],
        "current_business_profit_sum": (
            current["attributed_revenue_finalAmount"] - current["attributed_cost_finalAmount"]
        ),
        "current_business_peak95_max": current["current_business_peak95_max"],
        "current_business_source": "wide_daily_online_inservice_single_canonical_business",
        "current_business_confidence": "authoritative_daily_active_snapshot",
        "current_business_sampling_status": current["status"],
        "current_business_sampling_reason": current["reason"],
    })
    return resolved, audit


def fetch_current_business_wide(
    node_ids: list[str],
    wide_day: str,
    output_dir: Path,
    lookback_days: int,
    chunk_size: int,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dt.date, dt.date]:
    end_day = dt.date.fromisoformat(wide_day)
    start_day = end_day - dt.timedelta(days=lookback_days - 1)
    raw_path = output_dir / f"current_business_wide_raw_{start_day.strftime('%Y%m%d')}_{end_day.strftime('%Y%m%d')}.csv"
    current_path = output_dir / f"current_business_from_wide_{start_day.strftime('%Y%m%d')}_{end_day.strftime('%Y%m%d')}.csv"
    if raw_path.exists() and current_path.exists() and not refresh:
        return (
            pd.read_csv(raw_path, dtype="string", low_memory=False),
            pd.read_csv(current_path, dtype="string", low_memory=False),
            start_day,
            end_day,
        )

    all_rows: list[dict[str, Any]] = []
    node_chunks = chunks(node_ids, chunk_size)

    def fetch(index: int, batch: list[str]) -> tuple[int, list[dict[str, Any]]]:
        rows = rebuild.run_sql(current_business_sql(batch, start_day, end_day), database_id=19, schema="test")
        return index, rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, index, batch) for index, batch in enumerate(node_chunks, 1)]
        for future in concurrent.futures.as_completed(futures):
            index, rows = future.result()
            all_rows.extend(rows)
            print(f"current business chunk {index}/{len(node_chunks)}: rows={len(rows):,}")

    raw = pd.DataFrame(all_rows)
    raw.to_csv(raw_path, index=False)
    current = v1.derive_current_business_from_wide(
        raw,
        lookback_days=0,
        min_active_days=1 if lookback_days <= 1 else 2,
        online_only=True,
    )
    current.to_csv(current_path, index=False)
    return raw, current, start_day, end_day


def add_business_name_fallback(current: pd.DataFrame, business_name_map: dict[str, str]) -> pd.DataFrame:
    output = current.copy()
    if output.empty:
        return output
    output["current_business"] = output["current_business"].map(clean)
    output["current_business_name"] = output["current_business_name"].fillna("").map(clean)
    missing = output["current_business_name"].isin({"", "未映射业务"})
    output.loc[missing, "current_business_name"] = output.loc[missing, "current_business"].map(
        business_name_map
    ).fillna("")
    still_missing = output["current_business"].ne("") & output["current_business_name"].eq("")
    output.loc[still_missing, "current_business_name"] = "未映射业务"
    return output


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def add_current_value_policy(rec: pd.DataFrame) -> pd.DataFrame:
    """Protect profitable current businesses from model-only churn."""
    output = rec.copy()
    for column in [
        "current_business_revenue_sum",
        "current_business_cost_sum",
        "current_business_profit_sum",
        "current_business_peak95_max",
        "v2_rank_score_top1",
    ]:
        output[column] = numeric_series(output, column)

    peak95_mbps = output["current_business_peak95_max"] / 1_000_000.0
    output["current_peak95_mbps"] = peak95_mbps
    output["current_revenue_per_peak95_mbps"] = (
        output["current_business_revenue_sum"] / peak95_mbps.where(peak95_mbps > 0)
    )
    output["current_profit_per_peak95_mbps"] = (
        output["current_business_profit_sum"] / peak95_mbps.where(peak95_mbps > 0)
    )
    output["current_profit_margin"] = (
        output["current_business_profit_sum"]
        / output["current_business_revenue_sum"].where(output["current_business_revenue_sum"].abs() > 1e-9)
    )

    mainstream = output[
        output["current_business"].ne("")
        & output["current_business_is_mainstream"].astype(bool)
    ].copy()
    peer = (
        mainstream.groupby("current_business", dropna=False)
        .agg(
            recommended_peer_nodes=("node_id", "nunique"),
            recommended_peer_median_revenue_7d=("current_business_revenue_sum", "median"),
            recommended_peer_median_cost_7d=("current_business_cost_sum", "median"),
            recommended_peer_median_profit_7d=("current_business_profit_sum", "median"),
            recommended_peer_median_profit_per_peak95_mbps=("current_profit_per_peak95_mbps", "median"),
            recommended_peer_p75_profit_7d=("current_business_profit_sum", lambda value: value.quantile(0.75)),
            recommended_peer_p75_profit_per_peak95_mbps=("current_profit_per_peak95_mbps", lambda value: value.quantile(0.75)),
        )
        .reset_index()
        .rename(columns={"current_business": "recommended_business"})
    )
    output = output.merge(peer, on="recommended_business", how="left", sort=False)
    output["current_vs_recommended_peer_median_profit_delta"] = (
        output["current_business_profit_sum"] - output["recommended_peer_median_profit_7d"]
    )
    output["current_vs_recommended_peer_median_unit_profit_delta"] = (
        output["current_profit_per_peak95_mbps"]
        - output["recommended_peer_median_profit_per_peak95_mbps"]
    )

    beats_profit = (
        output["recommended_peer_median_profit_7d"].notna()
        & (output["current_business_profit_sum"] >= output["recommended_peer_median_profit_7d"])
    )
    beats_unit_profit = (
        output["recommended_peer_median_profit_per_peak95_mbps"].notna()
        & output["current_profit_per_peak95_mbps"].notna()
        & (
            output["current_profit_per_peak95_mbps"]
            >= output["recommended_peer_median_profit_per_peak95_mbps"]
        )
    )
    profitable = output["current_business_profit_sum"] > 0
    healthy_margin = output["current_profit_margin"].fillna(0.0) >= 0.03
    output["current_realized_beats_recommended_peer"] = (
        output["current_business"].ne("")
        & output["current_business_is_mainstream"].astype(bool)
        & output["recommended_business"].ne("")
        & output["current_business"].ne(output["recommended_business"])
        & profitable
        & healthy_margin
        & beats_profit
        & (beats_unit_profit | output["current_profit_per_peak95_mbps"].isna())
    )
    output["current_value_policy_reason"] = ""
    protected = output["current_realized_beats_recommended_peer"]
    output.loc[protected, "current_value_policy_reason"] = (
        "keep current: realized profit and unit profit beat recommended business peer median"
    )
    high_risk_profitable = (
        output["current_business"].ne("")
        & output["current_business_is_mainstream"].astype(bool)
        & output["recommended_business"].ne("")
        & output["current_business"].ne(output["recommended_business"])
        & profitable
        & output["v2_risk_level_top1"].fillna("").eq("high")
    )
    observe = high_risk_profitable & ~protected
    output["current_profitable_high_risk_observe"] = observe
    output.loc[observe, "current_value_policy_reason"] = (
        "observe: current business is profitable and recommended top1 risk is high"
    )
    return output


def score_and_compare(
    nodes: pd.DataFrame,
    current: pd.DataFrame,
    model_path: Path,
    allowlist: pd.DataFrame,
) -> pd.DataFrame:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    recommendations = v2.recommendations_for_nodes(nodes, model, top_k=3)
    node_context = nodes.copy()
    node_context["node_id"] = node_context["node_id"].map(clean)
    node_context = node_context.drop_duplicates("node_id", keep="last")
    rec = recommendations.merge(current, on="node_id", how="left", sort=False)
    rec = rec.merge(node_context, on="node_id", how="left", sort=False, suffixes=("", "_node"))
    allow_ids = set(allowlist["business"].map(clean))
    for index in range(1, 4):
        for column in [f"v2_business_top{index}", f"v2_business_name_top{index}"]:
            if column not in rec.columns:
                rec[column] = ""
    rec["current_business"] = rec["current_business"].fillna("").map(clean)
    rec["current_business_name"] = rec["current_business_name"].fillna("").map(clean)
    rec.loc[rec["current_business"].eq(""), "current_business_name"] = "未识别当前业务"
    rec.loc[
        rec["current_business"].ne("") & rec["current_business_name"].eq(""),
        "current_business_name",
    ] = "未映射业务"
    rec["current_business_is_mainstream"] = rec["current_business"].isin(allow_ids)
    rec["recommended_business"] = rec["v2_business_top1"].map(clean)
    rec["recommended_business_name"] = rec["v2_business_name_top1"].fillna("").map(clean)
    rec["current_eq_top1"] = rec["current_business"].ne("") & rec["current_business"].eq(rec["recommended_business"])
    rec["current_in_top3"] = rec.apply(
        lambda row: clean(row["current_business"]) in {
            clean(row["v2_business_top1"]),
            clean(row["v2_business_top2"]),
            clean(row["v2_business_top3"]),
        },
        axis=1,
    )
    rec["model_current_not_top1"] = rec["current_business"].eq("") | ~rec["current_eq_top1"]
    rec = add_current_value_policy(rec)

    def action(row: pd.Series) -> str:
        current_business = clean(row.get("current_business"))
        if not current_business:
            return "current_business_unknown"
        if bool(row.get("current_eq_top1")):
            return "keep_top1"
        if not bool(row.get("current_business_is_mainstream")):
            return "review_switch_current_not_mainstream"
        if bool(row.get("current_realized_beats_recommended_peer")):
            return "keep_current_realized_value"
        if bool(row.get("current_profitable_high_risk_observe")):
            return "observe_current_profitable_high_risk"
        if bool(row.get("current_in_top3")):
            return "review_current_in_top3_not_top1"
        return "review_switch_candidate"

    rec["suggested_action"] = rec.apply(action, axis=1)
    priority = {
        "review_switch_current_not_mainstream": 1,
        "review_switch_candidate": 2,
        "review_current_in_top3_not_top1": 3,
        "current_business_unknown": 4,
        "observe_current_profitable_high_risk": 5,
        "keep_current_realized_value": 8,
        "keep_top1": 9,
    }
    rec["action_priority"] = rec["suggested_action"].map(priority).fillna(99).astype(int)
    rec["current_not_best"] = rec["suggested_action"].isin(
        {
            "review_switch_current_not_mainstream",
            "review_switch_candidate",
            "review_current_in_top3_not_top1",
        }
    )
    rec["recommended_top3_detail"] = rec.apply(
        lambda row: ";".join(
            f"{clean(row.get(f'v2_business_top{index}'))}:{clean(row.get(f'v2_business_name_top{index}'))}:"
            f"{clean(row.get(f'v2_risk_level_top{index}'))}"
            for index in range(1, 4)
            if clean(row.get(f"v2_business_top{index}"))
        ),
        axis=1,
    )
    return rec.sort_values(
        ["action_priority", "v2_risk_level_top1", "v2_rank_score_top1", "node_id"],
        ascending=[True, True, False, True],
    )


def write_report(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# 在线服务中非机房大节点主流业务重匹配扫描",
        "",
        "## 口径",
        "",
        f"- 节点快照日：`{summary['partitions']['analysis_day']}`；宽表当前业务窗口：`{summary['current_business_window']['start']}` 至 `{summary['current_business_window']['end']}`。",
        "- 节点范围：`node_analysis_data.state=online`、`stage=inService`、`supplysidedeliverytype != 机房/小盒子`，并经项目 `node_size_type=large_node` 二次过滤。",
        "- 推荐范围：只允许 `mainstream_business_allowlist.csv` 中的主流业务进入模型候选。",
        "",
        "## 结果",
        "",
        f"- 候选节点：{summary['candidate_nodes']:,}",
        f"- 最终扫描节点：{summary['scanned_nodes']:,}",
        f"- 当前业务可识别节点：{summary['current_business_nodes']:,}",
        f"- 策略后仍需处理节点：{summary['current_not_best_nodes']:,}",
        f"- 模型判定当前非 Top1 节点：{summary.get('model_current_not_top1_nodes', 0):,}",
        f"- 当前业务不在主流业务池：{summary['current_not_mainstream_nodes']:,}",
        "",
        "## 动作分布",
        "",
    ]
    for action, count in summary["action_counts"].items():
        lines.append(f"- `{action}`：{count:,}")
    validation = summary.get("validation", {})
    if validation:
        lines.extend([
            "",
            "## 校验",
            "",
            f"- 推荐业务越出主流池：{len(validation.get('recommended_outside_allowlist', []))}",
            f"- 非在线记录：{validation.get('bad_online_rows', 0):,}",
            f"- 非服务中记录：{validation.get('bad_inservice_rows', 0):,}",
            f"- 机房/小盒子记录：{validation.get('idc_or_smallbox_rows', 0):,}",
            f"- 非大节点记录：{validation.get('not_large_node_rows', 0):,}",
        ])
    top_pairs = summary.get("top_switch_pairs", [])[:10]
    if top_pairs:
        lines.extend(["", "## Top 迁移方向", ""])
        for row in top_pairs:
            lines.append(
                "- "
                f"{row.get('current_business_name', '')}({row.get('current_business', '')}) -> "
                f"{row.get('recommended_business_name', '')}({row.get('recommended_business', '')})："
                f"{int(row.get('node_count', 0)):,} 节点，风险 `{row.get('v2_risk_level_top1', '')}`"
            )
    lines.extend([
        "",
        "## 文件",
        "",
        f"- 明细：`{summary['artifacts']['all_recommendations']}`",
        f"- 不匹配候选：`{summary['artifacts']['mismatch_candidates']}`",
        f"- 当前业务：`{summary['artifacts']['current_business']}`",
    ])
    if "switch_pair_summary" in summary["artifacts"]:
        lines.append(f"- 迁移方向汇总：`{summary['artifacts']['switch_pair_summary']}`")
    if "priority_top300" in summary["artifacts"]:
        lines.append(f"- Top300 优先候选：`{summary['artifacts']['priority_top300']}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan current online in-service non-IDC large nodes.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--business-map", type=Path, default=DEFAULT_BUSINESS_MAP)
    parser.add_argument("--current-lookback-days", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partitions = latest_partitions()
    print(json.dumps(partitions, ensure_ascii=False, indent=2))
    allowlist = load_allowlist(args.allowlist)
    business_name_map = load_business_map(args.business_map)

    candidate_ids = discover_current_candidates(partitions["analysis_day"], output_dir, args.refresh)
    raw_nodes = fetch_current_node_attributes(candidate_ids, partitions["analysis_day"], output_dir, args.refresh)
    nodes = keep_current_non_idc_large(raw_nodes)
    nodes_path = output_dir / f"current_online_inservice_non_idc_large_nodes_{partitions['analysis_day']}.csv"
    nodes.to_csv(nodes_path, index=False)
    print(f"current nodes after final filter: {len(nodes):,}")

    node_ids = sorted(nodes["node_id"].map(clean).unique())
    current_raw, _, current_start, current_end = fetch_current_business_wide(
        node_ids,
        partitions["wide_day"],
        output_dir,
        args.current_lookback_days,
        args.chunk_size,
        args.refresh,
    )
    current, current_audit = derive_current_business_v3(current_raw, allowlist, business_name_map)
    current = add_business_name_fallback(current, business_name_map)
    current_path = output_dir / (
        f"current_business_from_wide_{current_start.strftime('%Y%m%d')}_"
        f"{current_end.strftime('%Y%m%d')}.csv"
    )
    current.to_csv(current_path, index=False)
    current_audit_path = output_dir / (
        f"current_business_sampling_audit_{current_start.strftime('%Y%m%d')}_"
        f"{current_end.strftime('%Y%m%d')}.csv"
    )
    current_audit.to_csv(current_audit_path, index=False)

    recommendations = score_and_compare(nodes, current, args.model, allowlist)
    all_path = output_dir / f"current_non_idc_large_recommendations_{partitions['analysis_day']}.csv"
    mismatch_path = output_dir / f"current_non_idc_large_mismatch_candidates_{partitions['analysis_day']}.csv"
    columns_first = [
        "node_id",
        "suggested_action",
        "current_not_best",
        "current_business",
        "current_business_name",
        "current_business_is_mainstream",
        "current_business_confidence",
        "current_business_day",
        "recommended_business",
        "recommended_business_name",
        "model_current_not_top1",
        "current_realized_beats_recommended_peer",
        "current_value_policy_reason",
        "current_business_profit_sum",
        "current_profit_per_peak95_mbps",
        "recommended_peer_median_profit_7d",
        "recommended_peer_median_profit_per_peak95_mbps",
        "current_vs_recommended_peer_median_profit_delta",
        "current_vs_recommended_peer_median_unit_profit_delta",
        "v2_rank_score_top1",
        "v2_risk_level_top1",
        "v2_risk_reasons_top1",
        "v2_business_top2",
        "v2_business_name_top2",
        "v2_business_top3",
        "v2_business_name_top3",
        "recommended_top3_detail",
        "province",
        "city",
        "isp",
        "network_schedule_type",
        "schedule_isp_scope",
        "schedule_province_scope",
        "scheduleisps_text",
        "scheduleisps",
        "analysis_transprovrate",
        "analysis_supply_side_delivery_type",
        "resourcetype",
        "deliverytype",
        "nattype",
        "dialtype",
        "bw",
        "bandwidth",
        "actualbandwidth",
        "corenum",
        "node_state",
        "node_status",
        "node_analysis_stage",
        "node_stage",
    ]
    ordered_columns = [column for column in columns_first if column in recommendations.columns]
    ordered_columns.extend(column for column in recommendations.columns if column not in ordered_columns)
    recommendations[ordered_columns].to_csv(all_path, index=False)
    mismatch = recommendations[
        recommendations["current_not_best"]
        & recommendations["current_business"].ne("")
    ].copy()
    mismatch[ordered_columns].to_csv(mismatch_path, index=False)

    pair = mismatch.copy()
    for column in [
        "v2_rank_score_top1",
        "bandwidth",
        "actualbandwidth",
        "current_business_revenue_sum",
        "current_business_cost_sum",
        "current_business_profit_sum",
    ]:
        if column in pair.columns:
            pair[column] = pd.to_numeric(pair[column], errors="coerce")
    aggregations: dict[str, tuple[str, str]] = {"node_count": ("node_id", "nunique")}
    for source, target in [
        ("v2_rank_score_top1", "avg_rank_score"),
        ("bandwidth", "avg_bandwidth"),
        ("actualbandwidth", "avg_actualbandwidth"),
        ("current_business_revenue_sum", "avg_revenue_7d"),
        ("current_business_cost_sum", "avg_cost_7d"),
        ("current_business_profit_sum", "avg_profit_7d"),
    ]:
        if source in pair.columns:
            aggregations[target] = (source, "mean")
    switch_pair_summary = (
        pair.groupby(
            [
                "suggested_action",
                "current_business",
                "current_business_name",
                "recommended_business",
                "recommended_business_name",
                "v2_risk_level_top1",
            ],
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
        .sort_values(["node_count", "avg_rank_score"], ascending=[False, False])
    )
    switch_pair_path = output_dir / f"current_non_idc_large_switch_pair_summary_{partitions['analysis_day']}.csv"
    switch_pair_summary.to_csv(switch_pair_path, index=False)

    risk_order = {"low": 1, "medium": 2, "high": 3, "": 9}
    priority = mismatch.copy()
    if "v2_rank_score_top1" in priority.columns:
        priority["v2_rank_score_top1"] = pd.to_numeric(priority["v2_rank_score_top1"], errors="coerce")
    priority["risk_order"] = priority.get("v2_risk_level_top1", "").map(risk_order).fillna(9)
    priority = priority.sort_values(
        ["action_priority", "risk_order", "v2_rank_score_top1"],
        ascending=[True, True, False],
    ).head(300)
    priority_path = output_dir / f"current_non_idc_large_priority_top300_{partitions['analysis_day']}.csv"
    priority.to_csv(priority_path, index=False)

    action_counts = recommendations["suggested_action"].value_counts().to_dict()
    allow_ids = set(allowlist["business"].map(clean))
    recommended_ids: set[str] = set()
    for index in range(1, 4):
        column = f"v2_business_top{index}"
        if column in recommendations.columns:
            recommended_ids.update(clean(value) for value in recommendations[column] if clean(value))
    online_ok = recommendations["node_state"].map(clean).eq("online") | recommendations["node_status"].map(clean).eq("online")
    in_service_ok = (
        recommendations["node_analysis_stage"].map(clean).eq("inService")
        | recommendations["node_stage"].map(clean).eq("inService")
    )
    idc_bad = recommendations["analysis_supply_side_delivery_type"].map(clean).isin({"机房", "小盒子"})
    large_ok = recommendations["node_size_type"].map(clean).eq(v2.NODE_SIZE_LARGE)
    summary = {
        "partitions": partitions,
        "current_business_window": {
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
            "lookback_days": args.current_lookback_days,
        },
        "candidate_nodes": len(candidate_ids),
        "scanned_nodes": int(len(nodes)),
        "current_business_nodes": int(current["node_id"].nunique()) if not current.empty else 0,
        "current_business_sampling_status_counts": {
            str(key): int(value)
            for key, value in current_audit["status"].value_counts().to_dict().items()
        },
        "current_not_best_nodes": int((recommendations["current_not_best"] & recommendations["current_business"].ne("")).sum()),
        "model_current_not_top1_nodes": int((recommendations["model_current_not_top1"] & recommendations["current_business"].ne("")).sum()),
        "current_not_mainstream_nodes": int((~recommendations["current_business_is_mainstream"] & recommendations["current_business"].ne("")).sum()),
        "current_value_policy_protected_nodes": int(
            recommendations["suggested_action"].eq("keep_current_realized_value").sum()
        ),
        "current_profitable_high_risk_observe_nodes": int(
            recommendations["suggested_action"].eq("observe_current_profitable_high_risk").sum()
        ),
        "action_counts": action_counts,
        "top1_distribution": recommendations["recommended_business_name"].value_counts().to_dict(),
        "current_business_distribution": recommendations["current_business_name"].value_counts().head(30).to_dict(),
        "network_schedule_distribution": recommendations["network_schedule_type"].value_counts().to_dict(),
        "missing_current_business_name_ids": recommendations[
            recommendations["current_business"].ne("")
            & recommendations["current_business_name"].eq("未映射业务")
        ]["current_business"].value_counts().to_dict(),
        "validation": {
            "recommended_outside_allowlist": sorted(recommended_ids - allow_ids),
            "bad_online_rows": int((~online_ok).sum()),
            "bad_inservice_rows": int((~in_service_ok).sum()),
            "idc_or_smallbox_rows": int(idc_bad.sum()),
            "not_large_node_rows": int((~large_ok).sum()),
        },
        "top_switch_pairs": switch_pair_summary.head(20).to_dict(orient="records"),
        "artifacts": {
            "nodes": str(nodes_path),
            "current_business": str(current_path),
            "current_business_sampling_audit": str(current_audit_path),
            "all_recommendations": str(all_path),
            "mismatch_candidates": str(mismatch_path),
            "switch_pair_summary": str(switch_pair_path),
            "priority_top300": str(priority_path),
        },
    }
    summary_path = output_dir / f"current_non_idc_large_scan_summary_{partitions['analysis_day']}.json"
    report_path = output_dir / f"current_non_idc_large_scan_report_{partitions['analysis_day']}.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(summary, report_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
