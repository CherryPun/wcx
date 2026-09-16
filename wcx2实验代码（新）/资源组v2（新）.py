# -*- coding: utf-8 -*-
"""[wcx2 新增] 资源组 v2（新）：对齐 RJ 组键后做组内逐业务单位收益/利用率排名。

组键不含硬件。组数对账用现网节点池；排名用指定日 pairs。
用法：python "资源组v2（新）.py" [--day 2026-09-09]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from 资源组键 import GROUP_DIMS, group_fields, group_key  # noqa: E402

V3 = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）"
PAIRS = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"
CURRENT = V3 / "current_online_inservice_non_idc_large_nodes_v3.csv"
PROFILE = ROOT / "recent_month_large_1d（新）" / "multibusiness_nodes_large_recent_1m.csv"
BIZMAP = ROOT / "v1_business_name_map_enriched.csv"
SNAP = ROOT / "snapshots"
RJ_GROUPS = 2757


def num(value):
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_present(row: dict, names: list[str]):
    lower = {k.lower(): k for k in row}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return row.get(key)
    return None


def load_index(path: Path, key: str, value_names: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_csv(path):
        node = str(first_present(row, [key, "nodeId"]) or "")
        if not node or node in out:
            continue
        out[node] = {name: first_present(row, [name]) for name in value_names}
        out[node]["node_id"] = node
    return out


def load_bench(day: str) -> dict[str, float | None]:
    path = SNAP / f"bench_{day.replace('-', '')}.csv"
    if not path.exists():
        return {}
    out: dict[str, float | None] = {}
    for row in read_csv(path):
        node = str(row.get("node_id") or "")
        if node and node not in out:
            out[node] = num(row.get("overall_packet_loss_benchmark_satisfaction_pct"))
    return out


def payload_from(node: dict, bench: dict[str, float | None]) -> dict:
    node_id = str(node.get("node_id") or "")
    return {
        "province": first_present(node, ["province"]),
        "city": first_present(node, ["city", "daily_city", "join_city"]),
        "isp": first_present(node, ["isp", "daily_isp"]),
        "scheduleisps": first_present(node, ["scheduleisps", "daily_scheduleisps", "scheduleisps_text"]),
        "transprovrate": first_present(node, ["analysis_transprovrate", "daily_transprovrate", "join_transprovrate"]),
        "nattype": first_present(node, ["nattype", "daily_nattype", "join_nattype"]),
        "ipv6": first_present(node, ["analysis_issupportipv6", "join_isipv6schedule", "dial_ipv6_enable"]),
        "quality_pct": bench.get(node_id),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", default="2026-09-09")
    args = parser.parse_args()
    day = args.day
    bench = load_bench(day)
    current = load_index(CURRENT, "node_id", [
        "province", "city", "isp", "nattype", "scheduleisps", "scheduleisps_text",
        "analysis_transprovrate", "analysis_issupportipv6", "join_isipv6schedule",
        "dial_ipv6_enable", "join_city", "join_isp", "join_nattype", "join_transprovrate",
    ])
    profile = load_index(PROFILE, "node_id", [
        "province", "city", "isp", "nattype", "scheduleisps", "scheduleisps_text",
        "analysis_transprovrate", "analysis_issupportipv6", "join_isipv6schedule",
        "dial_ipv6_enable", "join_city", "join_isp", "join_nattype", "join_transprovrate",
    ])
    def count_groups(nodes: dict[str, dict]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        keys: dict[str, int] = defaultdict(int)
        dims: dict[str, dict[str, int]] = {name: defaultdict(int) for name in GROUP_DIMS}
        for node_id, node in nodes.items():
            merged = {**profile.get(node_id, {}), **{k: v for k, v in node.items() if v not in (None, "")}}
            merged["node_id"] = node_id
            fields = group_fields(payload_from(merged, bench))
            keys[group_key(payload_from(merged, bench))] += 1
            for name in GROUP_DIMS:
                dims[name][fields[name]] += 1
        return keys, dims

    pair_nodes: dict[str, dict] = {}
    pair_day_nodes: dict[str, dict] = {}
    pair_rows = read_csv(PAIRS)
    for row in pair_rows:
        node_id = str(row.get("node_id") or "")
        if not node_id:
            continue
        payload = {
            "node_id": node_id,
            "province": first_present(row, ["province", "daily_province"]),
            "city": first_present(row, ["city", "daily_city"]),
            "isp": first_present(row, ["isp", "daily_isp"]),
            "nattype": first_present(row, ["nattype", "daily_nattype"]),
            "scheduleisps": first_present(row, ["scheduleisps", "daily_scheduleisps"]),
            "analysis_transprovrate": first_present(row, ["analysis_transprovrate", "daily_transprovrate"]),
            "analysis_issupportipv6": first_present(row, ["analysis_issupportipv6"]),
        }
        pair_nodes.setdefault(node_id, payload)
        if str(row.get("sample_day") or "") == day:
            pair_day_nodes.setdefault(node_id, payload)

    current_keys, dim_values = count_groups(current)
    day_keys, _ = count_groups(pair_day_nodes)
    all_keys, _ = count_groups(pair_nodes)
    pool_keys = current_keys
    n_groups = len(pool_keys)
    print(f"现网节点 {len(current)} 组 {len(current_keys)}")
    print(f"当日 pairs 节点 {len(pair_day_nodes)} 组 {len(day_keys)}")
    print(f"全窗 pairs 节点 {len(pair_nodes)} 组 {len(all_keys)} | RJ={RJ_GROUPS}")
    print("现网各维取值数：")
    for name in GROUP_DIMS:
        counts = dim_values[name]
        unknown = counts.get("未知", 0) / max(sum(counts.values()), 1)
        top = sorted(counts.items(), key=lambda item: -item[1])[:3]
        print(f"  {name:<12} {len(counts):>4} | 未知 {unknown:.1%} | Top {[k for k, _ in top]}")

    names = {str(row["business"]): row.get("business_name") or str(row["business"]) for row in read_csv(BIZMAP)}
    buckets: dict[tuple[str, str], dict] = {}
    for row in pair_rows:
        if str(row.get("sample_day") or "") != day:
            continue
        node_id = str(row.get("node_id") or "")
        bw = num(row.get("construction_bandwidth_mbps") or row.get("buildBandwidth"))
        if not node_id or not bw or bw <= 0:
            continue
        cost = num(row.get("cum_cost_7d"))
        peak = num(row.get("capacity_peak95_mbps"))
        if peak is None:
            bps = num(row.get("capacity_peak95_bps"))
            peak = None if bps is None else bps / 1e6
        merged = {**profile.get(node_id, {}), **row, "node_id": node_id}
        key = group_key(payload_from(merged, bench))
        business = str(row.get("business") or "")
        item = buckets.setdefault((key, business), {"nodes": set(), "bw": 0.0, "unit": [], "util": []})
        item["nodes"].add(node_id)
        item["bw"] += bw
        if cost is not None:
            item["unit"].append(cost / bw)
        if peak is not None:
            item["util"].append(peak / bw)

    ranks: dict[str, list[dict]] = defaultdict(list)
    for (key, business), item in buckets.items():
        ranks[key].append({
            "grp": key,
            "business": business,
            "business_name": names.get(business, business),
            "nodes": len(item["nodes"]),
            "bw": item["bw"],
            "unit": median(item["unit"]) if item["unit"] else "",
            "util": median(item["util"]) if item["util"] else "",
            "day": day,
        })
    rows = []
    for key, items in ranks.items():
        def ranked(field: str) -> dict[int, int]:
            ordered = sorted(items, key=lambda row: (-(row[field] if row[field] != "" else float("-inf")), row["business"]))
            out: dict[int, int] = {}
            last = None
            last_rank = 0
            for i, row in enumerate(ordered, start=1):
                value = row[field]
                if value != last:
                    last_rank = i
                    last = value
                out[id(row)] = last_rank
            return out
        unit_rank = ranked("unit")
        util_rank = ranked("util")
        for row in items:
            row["unit_rank"] = unit_rank[id(row)]
            row["util_rank"] = util_rank[id(row)]
            row["evidence"] = "可用" if row["nodes"] >= 3 else "证据不足"
            rows.append(row)
    rows.sort(key=lambda row: (row["grp"], row["unit_rank"]))
    recon = [{"grp": key, "nodes": n, **dict(zip(GROUP_DIMS, key.split("|")))} for key, n in sorted(day_keys.items())]
    rank_path = ROOT / "资源组v2排名（新）.csv"
    recon_path = ROOT / "资源组v2对账（新）.csv"
    write_csv(rank_path, rows, ["grp", "business", "business_name", "nodes", "bw", "unit", "util", "unit_rank", "util_rank", "evidence", "day"])
    write_csv(recon_path, recon, ["grp", "nodes", *GROUP_DIMS])
    usable = sum(1 for row in rows if row["evidence"] == "可用")
    print(f"排名 组 {len(ranks)} | 组×业务 {len(rows)} | 可用 {usable}")
    print(f"[out] {rank_path.name} , {recon_path.name}")
    closest = min(
        (("现网", len(current_keys)), ("当日pairs", len(day_keys)), ("全窗pairs", len(all_keys))),
        key=lambda item: abs(item[1] - RJ_GROUPS),
    )
    print(f"最接近 RJ 的池：{closest[0]} {closest[1]}（差 {closest[1] - RJ_GROUPS}）")
    if closest[1] != RJ_GROUPS:
        print(f"组数未对齐 RJ {RJ_GROUPS}。排名已出，但对账未过。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
