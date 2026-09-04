#!/usr/bin/env python3
"""Build V3 daily business facts and a large-node recommendation model.

V3 fixes two data-grain problems in the earlier pipeline:
- one node/day is assigned from an online + inService business snapshot;
- Qiniu virtual customer IDs are one logical business named 七牛CDN-ZJ月95.

The daily reward uses cost_finalAmount as miner income and
revenue_finalAmount - cost_finalAmount as platform profit. Both are divided by
the node-day buildBandwidth value from node_day_ops_wide_full.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import build_mainstream_large_training as mainstream
import rebuild_multibusiness_model as rebuild
import v1_recommendation_pipeline as v1
import v2_ranking_model as v2


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = HERE / "recent_month_large_1d"
DEFAULT_OUTPUT_DIR = HERE / "recent_month_large_mainstream_v3_daily"
DEFAULT_ALLOWLIST = HERE / "mainstream_business_allowlist.csv"
DEFAULT_BUSINESS_MAP = HERE / "v1_business_name_map_enriched.csv"
DEFAULT_VIRTUAL_BINDING_DIR = HERE / "business_binding_audit"
DEFAULT_CURRENT_NODES = (
    HERE
    / "current_non_idc_large_scan_network_scope"
    / "current_online_inservice_non_idc_large_nodes_20260902.csv"
)
SUPERSET_RESULT_CAP = 10_000
QINIU_BUSINESS_ID = "10000280"
QINIU_BUSINESS_NAME = "七牛CDN-ZJ月95"
REWARD_ONLY_RANK_WEIGHTS = (
    "source_rate=0,champion=0,expected_best_rate=0,expected_score=1"
)
DAILY_PROFILE_FIELDS = {
    "province": "daily_province",
    "city": "daily_city",
    "isp": "daily_isp",
    "natType": "daily_nattype",
    "resourceType": "daily_resourcetype",
    "deliveryType": "daily_deliverytype",
}


def clean(value: Any) -> str:
    return v1.clean_cell(value)


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def load_allowlist(path: Path) -> pd.DataFrame:
    return mainstream.load_allowlist(path)


def qiniu_ids(allowlist: pd.DataFrame) -> set[str]:
    return set(allowlist.loc[allowlist["brand"].eq("七牛"), "business"].map(clean))


def canonical_business(business: Any, qiniu_business_ids: set[str]) -> str:
    value = clean(business)
    return QINIU_BUSINESS_ID if value in qiniu_business_ids else value


def load_virtual_bindings(path: Path | None) -> dict[str, set[str]]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"virtual_business", "business"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"virtual binding file missing columns: {sorted(missing)}")
    output: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        virtual_business = clean(row.virtual_business)
        business = clean(row.business)
        if virtual_business and business:
            output.setdefault(virtual_business, set()).add(business)
    return output


def latest_virtual_binding_path(directory: Path = DEFAULT_VIRTUAL_BINDING_DIR) -> Path | None:
    candidates = sorted(directory.glob("niulink_virtual_business_bindings_*.csv"))
    return candidates[-1] if candidates else None


def apply_virtual_business_bindings(
    frame: pd.DataFrame,
    allow_ids: set[str],
    qiniu_business_ids: set[str],
    virtual_bindings: dict[str, set[str]],
) -> pd.DataFrame:
    """Resolve an active virtual ID only when node-day evidence is unique."""
    output = frame.copy()
    output["canonical_business"] = output["source_business"].map(
        lambda value: canonical_business(value, qiniu_business_ids)
    )
    output["is_mainstream"] = output["source_business"].isin(allow_ids)
    output["is_known_virtual"] = output["source_business"].isin(virtual_bindings)
    output["virtual_resolution_status"] = ""
    output["virtual_resolved_business"] = ""
    if not virtual_bindings:
        return output

    output["_has_financial"] = (
        output["cost_finalAmount"].fillna(0.0).abs().gt(1e-12)
        | output["revenue_finalAmount"].fillna(0.0).abs().gt(1e-12)
    )
    keys = ["node_id", "sample_day"]
    active_virtual = output[
        output["is_active"]
        & output["is_known_virtual"]
        & ~output["is_mainstream"]
    ]
    for key_values, virtual_rows in active_virtual.groupby(keys, sort=False, dropna=False):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        day_mask = pd.Series(True, index=output.index)
        for key, value in zip(keys, key_values):
            day_mask &= output[key].eq(value)
        day_rows = output[day_mask]
        for virtual_business, source_rows in virtual_rows.groupby("source_business", sort=False):
            associated_sources = virtual_bindings.get(clean(virtual_business), set())
            associated_canonical = {
                canonical_business(value, qiniu_business_ids)
                for value in associated_sources
                if value in allow_ids
            }
            financial_candidates = {
                canonical_business(value, qiniu_business_ids)
                for value in day_rows.loc[
                    day_rows["_has_financial"]
                    & day_rows["source_business"].isin(associated_sources),
                    "source_business",
                ]
            }
            active_candidates = {
                canonical_business(value, qiniu_business_ids)
                for value in day_rows.loc[
                    day_rows["is_active"]
                    & day_rows["source_business"].isin(associated_sources),
                    "source_business",
                ]
            }
            suggested_sources: set[str] = set()
            for value in source_rows.get("vendorSuggestCustomersName", pd.Series(dtype=object)):
                suggested_sources.update(parse_customer_ids(value))
            suggested_candidates = {
                canonical_business(value, qiniu_business_ids)
                for value in suggested_sources & associated_sources
                if value in allow_ids
            }

            evidence = financial_candidates or active_candidates or suggested_candidates
            if len(evidence) == 1:
                resolved = next(iter(evidence))
            elif not evidence and len(associated_canonical) == 1:
                resolved = next(iter(associated_canonical))
            else:
                resolved = ""

            virtual_mask = day_mask & output["source_business"].eq(virtual_business)
            if resolved:
                output.loc[virtual_mask, "canonical_business"] = resolved
                output.loc[virtual_mask, "is_mainstream"] = True
                output.loc[virtual_mask, "virtual_resolution_status"] = "resolved"
                output.loc[virtual_mask, "virtual_resolved_business"] = resolved
            else:
                output.loc[virtual_mask, "virtual_resolution_status"] = "ambiguous"
    output = output.drop(columns=["_has_financial"])
    return output


def add_consecutive_sample_weights(facts: pd.DataFrame) -> pd.DataFrame:
    """Give every contiguous node/business run one unit of total weight."""
    if facts.empty:
        return facts
    output = facts.copy()
    output["_original_order"] = range(len(output))
    output["_sample_date"] = pd.to_datetime(output["sample_day"], errors="coerce")
    output = output.sort_values(["node_id", "_sample_date", "business", "_original_order"])
    previous_business = output.groupby("node_id", sort=False)["business"].shift()
    day_gap = output.groupby("node_id", sort=False)["_sample_date"].diff().dt.days
    new_run = previous_business.ne(output["business"]) | day_gap.ne(1) | day_gap.isna()
    output["consecutive_business_run"] = new_run.groupby(output["node_id"]).cumsum().astype(int)
    run_fields = ["node_id", "business", "consecutive_business_run"]
    output["consecutive_valid_days"] = output.groupby(run_fields)["sample_day"].transform("size")
    output["sample_weight"] = 1.0 / output["consecutive_valid_days"].astype(float)
    return (
        output.sort_values("_original_order")
        .drop(columns=["_original_order", "_sample_date"])
        .reset_index(drop=True)
    )


def parse_customer_ids(value: Any) -> set[str]:
    text = clean(value)
    if not text:
        return set()
    return set(re.findall(r"(?<!\d)(\d{7,})(?=\s*:|\D|$)", text))


def normalize_schedule_isps(value: Any, local_isp: Any = "") -> str:
    return v2.canonical_schedule_isps({"isp": local_isp, "scheduleisps": value})


def normalize_transprov_rate(value: Any) -> float | None:
    """An empty value means local province; only 0 and 100 are valid."""
    text = clean(value)
    if not text:
        return 0.0
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if abs(parsed) < 1e-9:
        return 0.0
    if abs(parsed - 100.0) < 1e-9:
        return 100.0
    return None


def most_common_nonempty(values: pd.Series) -> str:
    cleaned = [clean(value) for value in values if clean(value)]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def raw_sql(node_ids: list[str], start_day: str, end_day: str) -> str:
    ids = ",\n".join(rebuild.sql_quote(value) for value in node_ids)
    return f"""
    WITH active_days AS (
      SELECT DISTINCT nodeId, day
      FROM node_day_ops_wide_full
      WHERE day BETWEEN DATE '{start_day}' AND DATE '{end_day}'
        AND nodeId IN ({ids})
        AND customerId > 0
        AND LOWER(COALESCE(state, '')) = 'online'
        AND LOWER(COALESCE(stage, '')) = 'inservice'
    )
    SELECT
      t.nodeId,
      t.day,
      CAST(t.customerId AS VARCHAR) AS customerId,
      t.customerName,
      t.state,
      t.stage,
      CAST(t.buildBandwidth AS DOUBLE) AS buildBandwidth,
      t.province,
      t.city,
      t.isp,
      t.natType,
      t.resourceType,
      t.deliveryType,
      CAST(t.cost_finalAmount AS DOUBLE) AS cost_finalAmount,
      CAST(t.revenue_finalAmount AS DOUBLE) AS revenue_finalAmount,
      CAST(t.peak95 AS DOUBLE) AS peak95,
      CAST(t.transProvRate AS DOUBLE) AS transProvRate,
      CAST(t.scheduleISPs AS VARCHAR) AS scheduleISPs,
      CAST(t.vendorSuggestCustomersName AS VARCHAR) AS vendorSuggestCustomersName,
      CAST(t.virtualCustomersName AS VARCHAR) AS virtualCustomersName,
      CAST(t.snapshotTime AS VARCHAR) AS snapshotTime,
      CAST(t.updatedTime AS VARCHAR) AS updatedTime
    FROM node_day_ops_wide_full t
    JOIN active_days a ON t.nodeId = a.nodeId AND t.day = a.day
    WHERE t.customerId > 0
    ORDER BY t.nodeId, t.day, t.customerId
    """


def fetch_raw_rows(
    candidate_ids: list[str],
    start_day: str,
    end_day: str,
    output_path: Path,
    chunk_size: int,
    workers: int,
    refresh: bool,
) -> pd.DataFrame:
    if output_path.exists() and not refresh:
        print(f"reuse cached {output_path}")
        return pd.read_csv(output_path, dtype={"nodeId": "string", "customerId": "string"}, low_memory=False)

    batches = chunks(candidate_ids, chunk_size)
    parts_dir = output_path.parent / f"{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def fetch_complete(batch: list[str], label: str) -> list[dict[str, Any]]:
        rows = rebuild.run_sql(raw_sql(batch, start_day, end_day), database_id=19, schema="test")
        if len(rows) < SUPERSET_RESULT_CAP:
            return rows
        if len(batch) <= 1:
            raise RuntimeError(
                f"Superset result cap reached for one node in {label}: rows={len(rows):,}; "
                "split this node by date before training"
            )
        midpoint = len(batch) // 2
        print(f"raw {label} reached {len(rows):,}; split {len(batch)} nodes into {midpoint}+{len(batch) - midpoint}")
        return (
            fetch_complete(batch[:midpoint], f"{label}a")
            + fetch_complete(batch[midpoint:], f"{label}b")
        )

    def fetch_one(index: int, batch: list[str]) -> tuple[int, Path, int]:
        part_path = parts_dir / f"part_{index:04d}.csv"
        if part_path.exists() and not refresh:
            part = pd.read_csv(part_path, low_memory=False)
            return index, part_path, len(part)
        rows = fetch_complete(batch, f"batch-{index}")
        pd.DataFrame(rows).to_csv(part_path, index=False)
        return index, part_path, len(rows)

    part_paths: dict[int, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, index, batch) for index, batch in enumerate(batches, 1)]
        for future in concurrent.futures.as_completed(futures):
            index, part_path, row_count = future.result()
            part_paths[index] = part_path
            print(f"raw node-day chunk {index}/{len(batches)}: rows={row_count:,}")

    frames = [
        pd.read_csv(part_paths[index], dtype={"nodeId": "string", "customerId": "string"}, low_memory=False)
        for index in sorted(part_paths)
    ]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    frame.to_csv(output_path, index=False)
    return frame


def _build_daily_facts_reference(
    raw: pd.DataFrame,
    allowlist: pd.DataFrame,
    business_name_map: dict[str, str],
    virtual_bindings: dict[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = raw.copy()
    rename = {"nodeId": "node_id", "day": "sample_day", "customerId": "source_business"}
    frame = frame.rename(columns=rename)
    for column in ["node_id", "sample_day", "source_business", "state", "stage"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].map(clean)
    for column in ["buildBandwidth", "cost_finalAmount", "revenue_finalAmount", "peak95", "transProvRate"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    allow_ids = set(allowlist["business"].map(clean))
    qiniu_business_ids = qiniu_ids(allowlist)
    allow_names = dict(zip(allowlist["business"].map(clean), allowlist["business_name"].map(clean)))
    allow_names[QINIU_BUSINESS_ID] = QINIU_BUSINESS_NAME
    frame["is_active"] = (
        frame["state"].str.lower().eq("online")
        & frame["stage"].str.lower().eq("inservice")
    )
    frame = apply_virtual_business_bindings(
        frame, allow_ids, qiniu_business_ids, virtual_bindings or {}
    )

    facts: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    raw_status: list[str] = []
    for (node_id, sample_day), group in frame.groupby(["node_id", "sample_day"], sort=False, dropna=False):
        active = group[group["is_active"]]
        active_mainstream = active[active["is_mainstream"]]
        active_mainstream_businesses = sorted(set(active_mainstream["canonical_business"]) - {""})
        active_nonmainstream = sorted(set(active.loc[~active["is_mainstream"], "source_business"]) - {""})
        status = "clean"
        reason = ""
        selected = ""
        if not active_mainstream_businesses:
            status = "excluded_no_active_mainstream_business"
            reason = "no online+inService mainstream business row"
        elif len(active_mainstream_businesses) > 1:
            status = "excluded_multiple_active_mainstream_businesses"
            reason = "multiple canonical mainstream businesses are online+inService on the same node-day"
        elif active_nonmainstream:
            status = "excluded_active_nonmainstream_overlap"
            reason = "mainstream and non-mainstream businesses are both online+inService on the same node-day"
        else:
            selected = active_mainstream_businesses[0]

        selected_active = active_mainstream[
            active_mainstream["canonical_business"].eq(selected)
        ].copy()
        schedule_values = selected_active.apply(
            lambda row: normalize_schedule_isps(row.get("scheduleISPs"), row.get("isp")),
            axis=1,
        )
        daily_schedule = most_common_nonempty(schedule_values)
        transprov_values = [
            normalize_transprov_rate(value)
            for value in selected_active.get("transProvRate", pd.Series(dtype=float))
        ]
        invalid_transprov_count = sum(value is None for value in transprov_values)
        valid_transprov_rates = sorted({value for value in transprov_values if value is not None})
        daily_transprov = valid_transprov_rates[0] if len(valid_transprov_rates) == 1 else pd.NA
        if status == "clean" and invalid_transprov_count:
            status = "excluded_invalid_transprov_rate"
            reason = "transProvRate must be empty/0 for local province or 100 for cross-province"
        elif status == "clean" and len(valid_transprov_rates) > 1:
            status = "excluded_inconsistent_transprov_rate"
            reason = "both local-province and cross-province transProvRate values exist on the same node-day"

        positive_bw = sorted({float(value) for value in group["buildBandwidth"].dropna() if float(value) > 0})
        if status == "clean" and not positive_bw:
            status = "excluded_missing_build_bandwidth"
            reason = "buildBandwidth is missing or non-positive"
        elif status == "clean" and len(positive_bw) > 1:
            status = "excluded_inconsistent_build_bandwidth"
            reason = "multiple buildBandwidth values exist on the same node-day"

        bound_ids: set[str] = set()
        if selected == QINIU_BUSINESS_ID:
            for value in active_mainstream.get("vendorSuggestCustomersName", pd.Series(dtype=object)):
                bound_ids.update(parse_customer_ids(value))
            attributed = group[
                group["source_business"].isin(qiniu_business_ids | bound_ids)
            ]
        elif selected:
            attributed = group[group["canonical_business"].eq(selected)]
        else:
            attributed = group.iloc[0:0]

        financial = (
            group["cost_finalAmount"].fillna(0.0).abs().gt(1e-12)
            | group["revenue_finalAmount"].fillna(0.0).abs().gt(1e-12)
        )
        unattributed = group.loc[financial & ~group.index.isin(attributed.index)]
        cost = float(attributed["cost_finalAmount"].fillna(0.0).sum())
        revenue = float(attributed["revenue_finalAmount"].fillna(0.0).sum())
        unattributed_cost = float(unattributed["cost_finalAmount"].fillna(0.0).sum())
        unattributed_revenue = float(unattributed["revenue_finalAmount"].fillna(0.0).sum())
        if status == "clean" and not unattributed.empty:
            status = "excluded_unattributed_financial_rows"
            reason = "non-zero financial rows cannot be assigned to the selected daily business"
        if status == "clean" and abs(cost) < 1e-12 and abs(revenue) < 1e-12:
            status = "excluded_zero_financial_outcome"
            reason = "selected business has zero miner income and zero platform revenue"

        active_ids = sorted(set(active["source_business"]) - {""})
        audit = {
            "node_id": node_id,
            "sample_day": sample_day,
            "status": status,
            "reason": reason,
            "selected_business": selected,
            "selected_business_name": allow_names.get(selected, business_name_map.get(selected, "")),
            "active_business_ids": "|".join(active_ids),
            "active_mainstream_canonical_ids": "|".join(active_mainstream_businesses),
            "active_nonmainstream_ids": "|".join(active_nonmainstream),
            "qiniu_bound_business_ids": "|".join(sorted(bound_ids)),
            "financial_business_ids": "|".join(sorted(set(group.loc[financial, "source_business"]) - {""})),
            "unattributed_financial_business_ids": "|".join(
                sorted(set(unattributed["source_business"]) - {""})
            ),
            "raw_rows": int(len(group)),
            "attributed_rows": int(len(attributed)),
            "build_bandwidth_values": "|".join(f"{value:g}" for value in positive_bw),
            "transprov_values": "|".join(f"{value:g}" for value in valid_transprov_rates),
            "invalid_transprov_rows": invalid_transprov_count,
            "attributed_cost_finalAmount": cost,
            "attributed_revenue_finalAmount": revenue,
            "unattributed_cost_finalAmount": unattributed_cost,
            "unattributed_revenue_finalAmount": unattributed_revenue,
        }
        audits.append(audit)
        raw_status.extend([status] * len(group))
        if status != "clean":
            continue

        daily_profile: dict[str, Any] = {}
        for source, target in DAILY_PROFILE_FIELDS.items():
            daily_profile[target] = most_common_nonempty(selected_active.get(source, pd.Series(dtype=object)))
        facts.append({
            "node_id": node_id,
            "business": selected,
            "sample_day": sample_day,
            "outcome_group_id": f"{sample_day}:{node_id}",
            "business_online_day": sample_day,
            "business_name": allow_names.get(selected, business_name_map.get(selected, "")),
            "cum_cost_7d": cost,
            "cum_revenue_7d": revenue,
            "outcome_days": 1,
            "outcome_distinct_days": 1,
            "outcome_window_days": 1,
            "outcome_grain": "daily_single_active_business",
            "buildBandwidth": positive_bw[0],
            "daily_scheduleisps": daily_schedule,
            "daily_transprovrate": daily_transprov,
            "source_active_business_ids": "|".join(active_ids),
            "qiniu_bound_business_ids": "|".join(sorted(bound_ids)),
            **daily_profile,
        })

    audit_frame = pd.DataFrame(audits)
    frame["node_day_sampling_status"] = raw_status
    facts_frame = pd.DataFrame(facts)
    if facts_frame.empty:
        return facts_frame, audit_frame, frame

    facts_frame["sample_day"] = pd.to_datetime(facts_frame["sample_day"], errors="coerce").dt.strftime("%Y-%m-%d")
    facts_frame["business_active_days"] = facts_frame.groupby(["node_id", "business"])["sample_day"].transform("nunique")
    facts_frame["online_day"] = facts_frame.groupby("node_id")["sample_day"].transform("min")
    facts_frame["business_count"] = facts_frame.groupby("node_id")["business"].transform("nunique")
    facts_frame["business_first_day_count"] = facts_frame.groupby("node_id")["sample_day"].transform("nunique")
    facts_frame["ever_online"] = 1
    return add_consecutive_sample_weights(facts_frame), audit_frame, frame


def build_daily_facts(
    raw: pd.DataFrame,
    allowlist: pd.DataFrame,
    business_name_map: dict[str, str],
    virtual_bindings: dict[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Vectorized implementation of the V3 node-day sampling contract."""
    frame = raw.copy().rename(columns={
        "nodeId": "node_id",
        "day": "sample_day",
        "customerId": "source_business",
    })
    keys = ["node_id", "sample_day"]
    text_columns = [
        *keys, "source_business", "state", "stage", "vendorSuggestCustomersName",
        "scheduleISPs", *DAILY_PROFILE_FIELDS,
    ]
    for column in text_columns:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").map(clean)
    numeric_columns = [
        "buildBandwidth", "cost_finalAmount", "revenue_finalAmount", "peak95", "transProvRate",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    allow_ids = set(allowlist["business"].map(clean))
    qiniu_business_ids = qiniu_ids(allowlist)
    allow_names = dict(zip(allowlist["business"].map(clean), allowlist["business_name"].map(clean)))
    allow_names[QINIU_BUSINESS_ID] = QINIU_BUSINESS_NAME
    frame["is_active"] = (
        frame["state"].str.lower().eq("online")
        & frame["stage"].str.lower().eq("inservice")
    )
    frame = apply_virtual_business_bindings(
        frame, allow_ids, qiniu_business_ids, virtual_bindings or {}
    )

    summary = frame[keys].drop_duplicates().set_index(keys)
    active = frame[frame["is_active"]]
    active_mainstream = active[active["is_mainstream"]]
    active_nonmainstream = active[~active["is_mainstream"]]

    def joined_distinct(source: pd.DataFrame, column: str) -> pd.Series:
        values = source[keys + [column]].copy()
        values = values[values[column].ne("")].drop_duplicates(keys + [column])
        return values.groupby(keys, sort=False)[column].agg("|".join)

    summary["active_business_ids"] = joined_distinct(active, "source_business")
    summary["active_mainstream_canonical_ids"] = joined_distinct(
        active_mainstream, "canonical_business"
    )
    summary["active_nonmainstream_ids"] = joined_distinct(
        active_nonmainstream, "source_business"
    )
    summary["active_mainstream_count"] = active_mainstream.groupby(keys)[
        "canonical_business"
    ].nunique()
    summary["active_nonmainstream_count"] = active_nonmainstream.groupby(keys)[
        "source_business"
    ].nunique()
    unresolved_virtual = active[active["virtual_resolution_status"].eq("ambiguous")]
    summary["ambiguous_virtual_count"] = unresolved_virtual.groupby(keys)[
        "source_business"
    ].nunique()
    summary["resolved_virtual_business_ids"] = joined_distinct(
        active[active["virtual_resolution_status"].eq("resolved")],
        "source_business",
    )
    summary["ambiguous_virtual_business_ids"] = joined_distinct(
        unresolved_virtual,
        "source_business",
    )
    summary["selected_business"] = active_mainstream.groupby(keys)["canonical_business"].min()

    positive_bw = frame[frame["buildBandwidth"].gt(0)][keys + ["buildBandwidth"]]
    summary["build_bandwidth_count"] = positive_bw.groupby(keys)["buildBandwidth"].nunique()
    summary["buildBandwidth"] = positive_bw.groupby(keys)["buildBandwidth"].min()
    bw_values = positive_bw.drop_duplicates(keys + ["buildBandwidth"]).copy()
    bw_values["_bw_text"] = bw_values["buildBandwidth"].map(lambda value: f"{float(value):g}")
    summary["build_bandwidth_values"] = bw_values.groupby(keys, sort=False)["_bw_text"].agg("|".join)

    selected = summary["selected_business"].rename("_selected_business")
    frame = frame.merge(selected, left_on=keys, right_index=True, how="left", sort=False)
    qiniu_active = frame[
        frame["is_active"]
        & frame["canonical_business"].eq(QINIU_BUSINESS_ID)
    ]
    bound_records: list[dict[str, str]] = []
    for row in qiniu_active[keys + ["vendorSuggestCustomersName"]].itertuples(index=False):
        for business in parse_customer_ids(row.vendorSuggestCustomersName):
            bound_records.append({"node_id": row.node_id, "sample_day": row.sample_day, "source_business": business})
    bound = pd.DataFrame(bound_records, columns=keys + ["source_business"]).drop_duplicates()
    if bound.empty:
        bound_keys: set[str] = set()
        bound_ids = pd.Series(dtype=object)
    else:
        bound["_bound_key"] = bound["node_id"] + "|" + bound["sample_day"] + "|" + bound["source_business"]
        bound_keys = set(bound["_bound_key"])
        bound_ids = bound.groupby(keys, sort=False)["source_business"].agg("|".join)
    summary["qiniu_bound_business_ids"] = bound_ids

    row_bound_key = frame["node_id"] + "|" + frame["sample_day"] + "|" + frame["source_business"]
    selected_qiniu = frame["_selected_business"].eq(QINIU_BUSINESS_ID)
    direct_attributed = (
        frame["_selected_business"].ne("")
        & ~selected_qiniu
        & frame["canonical_business"].eq(frame["_selected_business"])
    )
    qiniu_attributed = selected_qiniu & (
        frame["canonical_business"].eq(QINIU_BUSINESS_ID)
        | row_bound_key.isin(bound_keys)
    )
    frame["is_attributed"] = direct_attributed | qiniu_attributed
    frame["is_financial"] = (
        frame["cost_finalAmount"].fillna(0.0).abs().gt(1e-12)
        | frame["revenue_finalAmount"].fillna(0.0).abs().gt(1e-12)
    )
    frame["is_unattributed_financial"] = frame["is_financial"] & ~frame["is_attributed"]
    frame["_attributed_cost"] = frame["cost_finalAmount"].fillna(0.0).where(frame["is_attributed"], 0.0)
    frame["_attributed_revenue"] = frame["revenue_finalAmount"].fillna(0.0).where(frame["is_attributed"], 0.0)
    frame["_unattributed_cost"] = frame["cost_finalAmount"].fillna(0.0).where(frame["is_unattributed_financial"], 0.0)
    frame["_unattributed_revenue"] = frame["revenue_finalAmount"].fillna(0.0).where(frame["is_unattributed_financial"], 0.0)
    financial_summary = frame.groupby(keys, sort=False).agg(
        raw_rows=("source_business", "size"),
        attributed_rows=("is_attributed", "sum"),
        unattributed_financial_rows=("is_unattributed_financial", "sum"),
        attributed_cost_finalAmount=("_attributed_cost", "sum"),
        attributed_revenue_finalAmount=("_attributed_revenue", "sum"),
        unattributed_cost_finalAmount=("_unattributed_cost", "sum"),
        unattributed_revenue_finalAmount=("_unattributed_revenue", "sum"),
    )
    summary = summary.join(financial_summary)
    summary["financial_business_ids"] = joined_distinct(
        frame[frame["is_financial"]], "source_business"
    )
    summary["unattributed_financial_business_ids"] = joined_distinct(
        frame[frame["is_unattributed_financial"]], "source_business"
    )

    selected_active = frame[
        frame["is_active"]
        & frame["is_mainstream"]
        & frame["canonical_business"].eq(frame["_selected_business"])
    ].copy()
    for source, target in DAILY_PROFILE_FIELDS.items():
        profile = selected_active[keys + [source]].copy()
        profile[source] = profile[source].replace("", pd.NA)
        summary[target] = profile.groupby(keys, sort=False)[source].first()
    selected_active["_schedule"] = selected_active.apply(
        lambda row: normalize_schedule_isps(row.get("scheduleISPs"), row.get("isp")),
        axis=1,
    )
    selected_active["_transprov_effective"] = selected_active["transProvRate"].map(
        normalize_transprov_rate
    )
    selected_active["_transprov_invalid"] = selected_active["_transprov_effective"].isna()
    summary["daily_scheduleisps"] = selected_active.groupby(keys, sort=False)["_schedule"].first()
    summary["daily_transprovrate"] = selected_active.groupby(keys, sort=False)[
        "_transprov_effective"
    ].first()
    summary["invalid_transprov_rows"] = selected_active.groupby(keys, sort=False)[
        "_transprov_invalid"
    ].sum()
    summary["transprov_value_count"] = selected_active.groupby(keys, sort=False)[
        "_transprov_effective"
    ].nunique()

    summary = summary.reset_index()
    for column in [
        "active_mainstream_count", "active_nonmainstream_count", "build_bandwidth_count",
        "unattributed_financial_rows", "attributed_rows", "invalid_transprov_rows",
        "transprov_value_count", "ambiguous_virtual_count",
    ]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)
    summary["status"] = "clean"
    summary["reason"] = ""

    def exclude(mask: pd.Series, status: str, reason: str) -> None:
        eligible = summary["status"].eq("clean") & mask
        summary.loc[eligible, "status"] = status
        summary.loc[eligible, "reason"] = reason

    exclude(
        summary["ambiguous_virtual_count"].gt(0),
        "excluded_ambiguous_virtual_business_binding",
        "active virtual business cannot be uniquely resolved to one mainstream real business",
    )
    exclude(
        summary["active_mainstream_count"].eq(0),
        "excluded_no_active_mainstream_business",
        "no online+inService mainstream business row",
    )
    exclude(
        summary["active_mainstream_count"].gt(1),
        "excluded_multiple_active_mainstream_businesses",
        "multiple canonical mainstream businesses are online+inService on the same node-day",
    )
    exclude(
        summary["active_nonmainstream_count"].gt(0),
        "excluded_active_nonmainstream_overlap",
        "mainstream and non-mainstream businesses are both online+inService on the same node-day",
    )
    exclude(
        summary["build_bandwidth_count"].eq(0),
        "excluded_missing_build_bandwidth",
        "buildBandwidth is missing or non-positive",
    )
    exclude(
        summary["build_bandwidth_count"].gt(1),
        "excluded_inconsistent_build_bandwidth",
        "multiple buildBandwidth values exist on the same node-day",
    )
    exclude(
        summary["invalid_transprov_rows"].gt(0),
        "excluded_invalid_transprov_rate",
        "transProvRate must be empty/0 for local province or 100 for cross-province",
    )
    exclude(
        summary["transprov_value_count"].gt(1),
        "excluded_inconsistent_transprov_rate",
        "both local-province and cross-province transProvRate values exist on the same node-day",
    )
    exclude(
        summary["unattributed_financial_rows"].gt(0),
        "excluded_unattributed_financial_rows",
        "non-zero financial rows cannot be assigned to the selected daily business",
    )
    exclude(
        summary["attributed_cost_finalAmount"].fillna(0).abs().lt(1e-12)
        & summary["attributed_revenue_finalAmount"].fillna(0).abs().lt(1e-12),
        "excluded_zero_financial_outcome",
        "selected business has zero miner income and zero platform revenue",
    )
    summary["selected_business_name"] = summary["selected_business"].map(
        lambda value: allow_names.get(clean(value), business_name_map.get(clean(value), ""))
    )
    audit = summary.copy()

    clean_summary = summary[summary["status"].eq("clean")].copy()
    facts = pd.DataFrame({
        "node_id": clean_summary["node_id"],
        "business": clean_summary["selected_business"],
        "sample_day": clean_summary["sample_day"],
        "outcome_group_id": clean_summary["sample_day"] + ":" + clean_summary["node_id"],
        "business_online_day": clean_summary["sample_day"],
        "business_name": clean_summary["selected_business_name"],
        "cum_cost_7d": clean_summary["attributed_cost_finalAmount"],
        "cum_revenue_7d": clean_summary["attributed_revenue_finalAmount"],
        "outcome_days": 1,
        "outcome_distinct_days": 1,
        "outcome_window_days": 1,
        "outcome_grain": "daily_single_active_business",
        "buildBandwidth": clean_summary["buildBandwidth"],
        "daily_scheduleisps": clean_summary["daily_scheduleisps"],
        "daily_transprovrate": clean_summary["daily_transprovrate"],
        "source_active_business_ids": clean_summary["active_business_ids"],
        "qiniu_bound_business_ids": clean_summary["qiniu_bound_business_ids"],
        "resolved_virtual_business_ids": clean_summary["resolved_virtual_business_ids"],
        **{target: clean_summary[target] for target in DAILY_PROFILE_FIELDS.values()},
    })
    if not facts.empty:
        facts["sample_day"] = pd.to_datetime(facts["sample_day"], errors="coerce").dt.strftime("%Y-%m-%d")
        facts["business_active_days"] = facts.groupby(["node_id", "business"])["sample_day"].transform("nunique")
        facts["online_day"] = facts.groupby("node_id")["sample_day"].transform("min")
        facts["business_count"] = facts.groupby("node_id")["business"].transform("nunique")
        facts["business_first_day_count"] = facts.groupby("node_id")["sample_day"].transform("nunique")
        facts["ever_online"] = 1

    raw_annotated = frame.merge(
        audit[keys + ["status", "reason"]].rename(columns={"status": "node_day_sampling_status"}),
        on=keys,
        how="left",
        sort=False,
    )
    helper_columns = [column for column in raw_annotated.columns if column.startswith("_")]
    raw_annotated = raw_annotated.drop(columns=helper_columns)
    return add_consecutive_sample_weights(facts), audit, raw_annotated


def build_training_nodes(facts: pd.DataFrame) -> pd.DataFrame:
    latest = facts.sort_values("sample_day").drop_duplicates("node_id", keep="last").copy()
    rename = {
        "daily_province": "province",
        "daily_city": "city",
        "daily_isp": "isp",
        "daily_nattype": "nattype",
        "daily_resourcetype": "resourcetype",
        "daily_deliverytype": "deliverytype",
        "daily_scheduleisps": "scheduleisps",
        "daily_transprovrate": "analysis_transprovrate",
        "buildBandwidth": "bw",
    }
    latest = latest.rename(columns=rename)
    columns = [
        "node_id", "province", "city", "isp", "nattype", "resourcetype",
        "deliverytype", "bw", "scheduleisps", "analysis_transprovrate",
    ]
    output = latest[[column for column in columns if column in latest.columns]].copy()
    output["node_size_type"] = "large_node"
    return output


def load_business_names(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {
        clean(row.business): clean(row.business_name)
        for row in frame.itertuples(index=False)
        if clean(row.business) and clean(row.business_name)
    }


def run_profit_report(output_dir: Path, artifacts: dict[str, Path], summary_path: Path) -> Path:
    output_html = output_dir / "business_recommendation_profit_report_v3_daily.html"
    command = [
        sys.executable,
        "build_business_recommendation_profit_report.py",
        "--training-pairs", str(artifacts["training_pairs"]),
        "--recommendations", str(artifacts["recommendations"]),
        "--model-summary", str(summary_path),
        "--output-html", str(output_html),
        "--output-csv", str(output_dir / "business_recommendation_profit_report_v3_daily.csv"),
        "--output-data-js", str(output_dir / "business_recommendation_profit_report_v3_daily_data.js"),
    ]
    subprocess.run(command, cwd=HERE, check=True)
    return output_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean one-business-per-node-day V3 training data.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--business-map", type=Path, default=DEFAULT_BUSINESS_MAP)
    parser.add_argument(
        "--virtual-bindings",
        type=Path,
        help="NiuLink virtual binding CSV; defaults to the latest local audited snapshot.",
    )
    parser.add_argument("--current-nodes", type=Path, default=DEFAULT_CURRENT_NODES)
    parser.add_argument("--start-day")
    parser.add_argument("--end-day")
    parser.add_argument(
        "--raw-input",
        type=Path,
        help="Reuse an existing raw node-day CSV instead of querying Superset.",
    )
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-business-support", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_outcomes = pd.read_csv(
        args.source_dir / "multibusiness_outcomes_large_recent_1m.csv",
        usecols=["sample_day"],
        low_memory=False,
    )
    sample_days = pd.to_datetime(source_outcomes["sample_day"], errors="coerce").dropna()
    start_day = args.start_day or sample_days.min().strftime("%Y-%m-%d")
    end_day = args.end_day or sample_days.max().strftime("%Y-%m-%d")
    candidate_payload = json.loads(
        (args.source_dir / "large_candidate_node_ids_recent_1m.json").read_text(encoding="utf-8")
    )
    candidate_ids = sorted({clean(value) for value in candidate_payload["node_ids"] if clean(value)})
    if args.raw_input:
        raw_path = args.raw_input.resolve()
        raw = pd.read_csv(
            raw_path,
            dtype={"nodeId": "string", "customerId": "string"},
            low_memory=False,
        )
        print(f"reuse explicit raw input {raw_path}")
    else:
        raw_path = args.output_dir / f"node_day_business_raw_{start_day.replace('-', '')}_{end_day.replace('-', '')}.csv"
        raw = fetch_raw_rows(
            candidate_ids,
            start_day,
            end_day,
            raw_path,
            args.chunk_size,
            args.workers,
            args.refresh,
        )

    allowlist = load_allowlist(args.allowlist)
    names = load_business_names(args.business_map)
    virtual_binding_path = args.virtual_bindings or latest_virtual_binding_path()
    virtual_bindings = load_virtual_bindings(virtual_binding_path)
    facts, audit, raw_annotated = build_daily_facts(raw, allowlist, names, virtual_bindings)
    outcomes_path = args.output_dir / "multibusiness_outcomes_large_mainstream_v3_daily.csv"
    audit_path = args.output_dir / "node_day_sampling_audit_v3.csv"
    audit_raw_path = args.output_dir / "node_day_sampling_excluded_raw_v3.csv"
    facts.to_csv(outcomes_path, index=False)
    audit.to_csv(audit_path, index=False)
    excluded_keys = set(
        zip(
            audit.loc[audit["status"].ne("clean"), "node_id"],
            audit.loc[audit["status"].ne("clean"), "sample_day"],
        )
    )
    excluded_raw = raw_annotated[
        raw_annotated.apply(lambda row: (row["node_id"], row["sample_day"]) in excluded_keys, axis=1)
    ]
    excluded_raw.to_csv(audit_raw_path, index=False)

    training_nodes = build_training_nodes(facts)
    training_nodes_path = args.output_dir / "multibusiness_nodes_large_mainstream_training_v3_daily.csv"
    training_nodes.to_csv(training_nodes_path, index=False)
    if args.current_nodes.exists():
        report_nodes = pd.read_csv(args.current_nodes, dtype={"node_id": "string"}, low_memory=False)
    else:
        report_nodes = training_nodes
    report_nodes_path = args.output_dir / "current_online_inservice_non_idc_large_nodes_v3.csv"
    report_nodes.to_csv(report_nodes_path, index=False)

    artifacts = mainstream.build_models(
        args.output_dir,
        report_nodes_path,
        training_nodes_path,
        outcomes_path,
        args.business_map,
        args.min_business_support,
        "unit_bandwidth",
        REWARD_ONLY_RANK_WEIGHTS,
    )

    status_counts = audit["status"].value_counts().to_dict()
    business_support = (
        facts.groupby(["business", "business_name"], dropna=False)
        .agg(
            node_days=("node_id", "size"),
            effective_run_support=("sample_weight", "sum"),
            nodes=("node_id", "nunique"),
        )
        .reset_index()
        .sort_values(["node_days", "nodes"], ascending=False)
    )
    support_path = args.output_dir / "business_support_v3_daily.csv"
    business_support.to_csv(support_path, index=False)
    metrics = json.loads((artifacts["v2_dir"] / "v2_model_metrics.json").read_text(encoding="utf-8"))
    summary = {
        "version": "v3.1_daily_weighted_virtual_bindings",
        "date_window": {"start": start_day, "end": end_day},
        "candidate_large_nodes": len(candidate_ids),
        "raw_rows": len(raw),
        "raw_node_days": int(audit.shape[0]),
        "clean_node_days": len(facts),
        "clean_effective_run_support": float(facts["sample_weight"].sum()),
        "clean_nodes": int(facts["node_id"].nunique()),
        "clean_businesses": int(facts["business"].nunique()),
        "sampling_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "business_rules": {
            "active": "state=online AND stage=inService",
            "qiniu_canonical_business": QINIU_BUSINESS_ID,
            "qiniu_canonical_name": QINIU_BUSINESS_NAME,
            "virtual_binding_source": str(virtual_binding_path or ""),
            "virtual_businesses_loaded": len(virtual_bindings),
            "virtual_resolution_policy": (
                "resolve by unique financial, active-real, or vendor-suggestion evidence; "
                "exclude the node-day when resolution is ambiguous"
            ),
            "multiple_active_business_policy": "exclude whole node-day and write audit rows",
            "non_mainstream_policy": "not a candidate; active overlap excludes the node-day",
            "delivery_type_policy": "dedicated and aggregation are both eligible; no hard gate",
            "schedule_isps_empty_policy": "empty means the node ISP (local-network scheduling)",
            "transprov_policy": "empty/0 means local province; 100 means cross-province; other values are excluded",
            "repeat_day_weight": "1 / consecutive valid days in the same node-business run",
        },
        "target": {
            "miner_income": "cost_finalAmount",
            "platform_profit": "revenue_finalAmount - cost_finalAmount",
            "denominator": "node-day buildBandwidth (Mbps)",
            "objective": "0.5 * normalized miner unit income + 0.5 * normalized platform unit profit",
            "rank_weights": REWARD_ONLY_RANK_WEIGHTS,
        },
        "v2_test": metrics.get("test", {}),
        "metric_warning": (
            "hit@k compares recommendations with the historically observed action. "
            "It is not a causal best-business accuracy metric because one clean business is observed per node-day."
        ),
        "artifacts": {
            "raw": str(raw_path),
            "daily_facts": str(outcomes_path),
            "sampling_audit": str(audit_path),
            "excluded_raw": str(audit_raw_path),
            "business_support": str(support_path),
            **{key: str(value) for key, value in artifacts.items()},
        },
    }
    summary_path = args.output_dir / "v3_daily_training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    profit_report = run_profit_report(args.output_dir, artifacts, summary_path)
    summary["artifacts"]["profit_report"] = str(profit_report)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
