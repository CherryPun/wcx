# -*- coding: utf-8 -*-
"""[wcx2 新增] 每日快照（新）

把所有"不可逆"的原始状态按日存档，供后续 as-of 重建与对账使用：
  1) bench_<day>.csv      —— jarvis node_join 的 netbench 满意度（复用 fetch_latest_node_pressure.py）
  2) customer_<day>.csv   —— jarvis 5 分钟客户视图按 (node_id, customer_id) 聚合
  3) manifest_<day>.json  —— 行数 / 节点覆盖 / 文件 sha256 / 取值时间
另：成本与结算原始件需财务导出，本脚本只登记"未自动化"。

用法：
  python "每日快照（新）.py" [--day 2026-09-14] [--nodes <节点清单csv>] [--force]
默认节点清单 = V3 当前大节点清单；输出目录 = 仓库根 snapshots/（已 gitignore）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "skills" / "superset-sql-query" / "common" / "run_superset_sql.py"
DEFAULT_NODES = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）" / "current_online_inservice_non_idc_large_nodes_v3.csv"
OUT_DIR = ROOT / "snapshots"
ID_FIELDS = ("node_id", "nodeid", "id")

CUSTOMER_SQL = """SELECT node_id, customer_id, vendor, COUNT(*) AS n, ROUND(SUM(`in`),1) AS in_bytes, ROUND(SUM(`out`),1) AS out_bytes
FROM jarvis.dws_jarvis_customer_5min_view
WHERE day = DATE '{day}' AND node_id IN ({ids})
GROUP BY node_id, customer_id, vendor"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_nodes(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        field = next((f for f in ID_FIELDS if f in (reader.fieldnames or [])), None)
        if not field:
            raise SystemExit(f"节点清单缺少 id 列：{path}")
        return [str(r[field]).strip() for r in reader if str(r[field]).strip()]


def run_runner(sql: str, out_csv: Path, database_id: int, schema: str) -> bool:
    sql_file = OUT_DIR / "_tmp.sql"
    sql_file.write_text(sql, encoding="utf-8")
    r = subprocess.run([sys.executable, str(RUNNER), "--database-id", str(database_id), "--schema", schema,
                        "--sql-file", str(sql_file), "--output-csv", str(out_csv), "--preview", "1"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    sql_file.unlink(missing_ok=True)
    if not out_csv.exists():
        print(f"   [warn] 查询失败：{(r.stdout or '')[-300:]}{(r.stderr or '')[-200:]}")
        return False
    return True


def snapshot_customer(day: str, nodes: list[str], batch: int) -> int:
    parts = []
    for i in range(0, len(nodes), batch):
        chunk = nodes[i:i + batch]
        ids = ",".join("'" + n + "'" for n in chunk)
        out = OUT_DIR / f"_part_{i // batch}.csv"
        if run_runner(CUSTOMER_SQL.format(day=day, ids=ids), out, 19, "test"):
            parts.append(out)
    target = OUT_DIR / f"customer_{day}.csv"
    rows = 0
    with target.open("w", encoding="utf-8-sig", newline="") as out:
        writer = None
        for p in parts:
            with p.open(encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if writer is None:
                        writer = csv.writer(out)
                    writer.writerow(row)
                    rows += 1
            p.unlink(missing_ok=True)
    return max(rows - 1, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default=date.today().isoformat())
    ap.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    ap.add_argument("--batch-size", type=int, default=800)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    day = args.day
    if not args.nodes.exists():
        raise SystemExit(f"节点清单不存在：{args.nodes}")
    nodes = load_nodes(args.nodes)
    print(f"[snapshot] day={day} 节点清单 {len(nodes)} 个 -> {OUT_DIR}")

    manifest: dict[str, object] = {"day": day, "captured_at": datetime.now().isoformat(timespec="seconds"),
                                   "node_universe": len(nodes), "sources": {}}

    bench = OUT_DIR / f"bench_{day}.csv"
    if bench.exists() and not args.force:
        print(f"  bench 已存在，跳过（--force 覆盖）")
    else:
        r = subprocess.run([sys.executable, str(ROOT / "fetch_latest_node_pressure.py"),
                            "--nodes", str(args.nodes), "--snapshot-day", day, "--output", str(bench)],
                           cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not bench.exists():
            print(f"  [warn] bench 抓取失败：{(r.stdout or '')[-300:]}{(r.stderr or '')[-200:]}")

    cust = OUT_DIR / f"customer_{day}.csv"
    if cust.exists() and not args.force:
        print("  customer 已存在，跳过（--force 覆盖）")
    else:
        n = snapshot_customer(day, nodes, args.batch_size)
        print(f"  customer 写入 {n} 行")

    for p in (bench, cust):
        if p.exists():
            with p.open(encoding="utf-8-sig", newline="") as fh:
                rows = max(sum(1 for _ in fh) - 1, 0)
            manifest["sources"][p.name] = {"rows": rows, "sha256_16": sha256(p), "bytes": p.stat().st_size}
    manifest["sources"]["cost_settlement"] = {"rows": None, "note": "需财务导出，未自动化"}

    mf = OUT_DIR / f"manifest_{day}.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[snapshot] manifest -> {mf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
