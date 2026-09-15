# -*- coding: utf-8 -*-
"""[wcx2 新增] 压测回填（新）：按天回填历史 netbench 快照（断点续跑）

复用 fetch_latest_node_pressure.py（已验证路径），逐日导出到 snapshots/bench_YYYYMMDD.csv。
已存在的日期默认跳过（--force 覆盖）。只读 Superset，不改模型。

用法：
  python "压测回填（新）.py" [--start 2026-03-16] [--end 2026-09-09] [--force]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "snapshots"
NODES = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）" / "current_online_inservice_non_idc_large_nodes_v3.csv"
FETCH = ROOT / "fetch_latest_node_pressure.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-03-16")
    ap.add_argument("--end", default="2026-09-09")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not NODES.exists():
        raise SystemExit(f"节点清单不存在：{NODES}")
    SNAP.mkdir(exist_ok=True)

    d, end, ok, skip, fail = date.fromisoformat(args.start), date.fromisoformat(args.end), 0, 0, 0
    while d <= end:
        out = SNAP / f"bench_{d.strftime('%Y%m%d')}.csv"
        if out.exists() and not args.force:
            skip += 1
            d += timedelta(days=1)
            continue
        r = subprocess.run([sys.executable, str(FETCH), "--nodes", str(NODES),
                            "--snapshot-day", d.isoformat(), "--output", str(out)],
                           cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if out.exists() and out.stat().st_size > 0:
            ok += 1
        else:
            fail += 1
            print(f"[warn] {d} 失败：{(r.stdout or '')[-200:]}{(r.stderr or '')[-200:]}")
        d += timedelta(days=1)

    total = ok + skip
    print(f"\n[回填] 完成 {ok} 天（跳过 {skip}，失败 {fail}）| snapshots/ 现有 {total} 天")
    print("下一步：以 --snapshot-day 对齐后重跑 V5（as-of 压测特征），对照现行静态口径。")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
