# -*- coding: utf-8 -*-
"""[wcx2 新增] 业务小时形态与分时共享增益（新）

按日查询 jarvis.dws_jarvis_business_5min，服务端聚合到 (business, hour)，得到 24 维峰谷曲线，
并估算"分时共享"相对"日聚合计量"的有效容量增益：
  日聚合口径  = Σ_b max_h flow(b,h)      （每业务按各自峰值定容）
  分时共享口径 = max_h Σ_b flow(b,h)      （同时刻叠加后的峰值）
  增益 = 1 − 分时共享 / 日聚合
用法：python "业务小时形态（新）.py" [--days 2026-09-02,2026-09-03,...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "skills" / "superset-sql-query" / "common" / "run_superset_sql.py"
TMP = ROOT / "snapshots" / "_hourly"
DEFAULT_DAYS = ["2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06", "2026-09-07", "2026-09-08"]


def sql(day: str) -> str:
    return f"""SELECT customer_id AS business, hour(from_unixtime(ts)) AS hr, SUM(`out`) AS out_bytes
FROM jarvis.dws_jarvis_business_5min
WHERE day = DATE '{day}'
GROUP BY customer_id, hour(from_unixtime(ts))"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", default=",".join(DEFAULT_DAYS))
    args = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    frames = []
    for day in args.days.split(","):
        day = day.strip()
        out = TMP / f"hourly_{day}.csv"
        if not out.exists():
            sqlf = TMP / f"_{day}.sql"
            sqlf.write_text(sql(day), encoding="utf-8")
            r = subprocess.run([sys.executable, str(RUNNER), "--database-id", "19", "--schema", "test",
                                "--sql-file", str(sqlf), "--output-csv", str(out), "--preview", "1"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            sqlf.unlink(missing_ok=True)
            if not out.exists():
                print(f"[warn] {day} 失败：{(r.stdout or '')[-300:]}{(r.stderr or '')[-200:]}")
                continue
        d = pd.read_csv(out)
        d["day"] = day
        frames.append(d)
        print(f"[ok] {day} rows={len(d)}")
    if not frames:
        print("无数据")
        return 0
    h = pd.concat(frames, ignore_index=True)
    h["out_gbps"] = pd.to_numeric(h["out_bytes"], errors="coerce") / 1e9 / 24  # 24 个 5min 段 -> 平均 Gbps
    h["business"] = pd.to_numeric(h["business"], errors="coerce")
    piv = h.pivot_table(index="business", columns="hr", values="out_gbps", aggfunc="mean").fillna(0.0)

    per_biz_peak = piv.max(axis=1)            # 每业务峰值
    per_hour_total = piv.sum(axis=0)          # 各小时全业务叠加
    agg_sized = float(per_biz_peak.sum())
    shared_sized = float(per_hour_total.max())
    gain = 1 - shared_sized / agg_sized if agg_sized else float("nan")

    print(f"\n业务数 {len(piv)} | 小时 {piv.shape[1]}")
    print(f"日聚合定容 Σ_b max_h = {agg_sized:,.1f} Gbps")
    print(f"分时共享定容 max_h Σ_b = {shared_sized:,.1f} Gbps")
    print(f"→ 分时共享可省容量 {agg_sized - shared_sized:,.1f} Gbps（{gain:.1%}）")
    print("\n全天叠加最高的 5 个小时（hr: Gbps）：")
    print(per_hour_total.sort_values(ascending=False).head(5).round(1).to_dict())
    print("\n峰谷比最高的 8 个业务（峰值/谷值）：")
    ratio = (per_biz_peak / piv.replace(0, pd.NA).min(axis=1)).dropna().sort_values(ascending=False)
    print(pd.to_numeric(ratio, errors="coerce").head(8).round(1).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
