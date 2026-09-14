# -*- coding: utf-8 -*-
"""[wcx2 新增] 滚动时间回测（新）

按采用配置（HL7/cap2.0/单线程）在多个时间窗上重跑 V5 build，汇总时间外指标（均值/最差窗）。
用法：python "滚动时间回测（新）.py" [窗口数=3]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
V3 = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）"
PAIRS = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"
PROFILES = ROOT / "recent_month_large_1d（新）" / "multibusiness_nodes_large_recent_1m.csv"
CURRENT = V3 / "current_online_inservice_non_idc_large_nodes_v3.csv"
SUMMARY = V3 / "v3_daily_training_summary.json"
END_DAYS = ["2026-09-09", "2026-09-02", "2026-08-26", "2026-08-19"]
START_DAY = "2026-08-10"


def metric(summary: Path) -> dict:
    j = json.loads(summary.read_text(encoding="utf-8"))
    sel, base = j["model_comparison"]["selected"], j["model_comparison"]["baseline"]
    w = j["temporal_windows"]
    return {
        "test": f'{w["test_start"]}~{w["test_end"]}',
        "rows": sel["raw_rows"],
        "miner_r2": sel["miner"]["r2"], "miner_rmse": sel["miner"]["rmse"],
        "plat_r2": sel["platform"]["r2"], "plat_rmse": sel["platform"]["rmse"],
        "base_miner_r2": base["miner"]["r2"], "base_plat_r2": base["platform"]["r2"],
        "miner_cov90": sel.get("miner_interval_coverage90"),
        "plat_cov90": sel.get("platform_interval_coverage90"),
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    env = dict(os.environ)
    env.update({"V5_RECENCY_HALF_LIFE": "7", "V5_RECENCY_WEIGHT_CAP": "2.0", "V5_NTHREAD": "1"})
    env["PYTHONIOENCODING"] = "utf-8"
    env["MULTIBUSINESS_SUPERSET_COMMON"] = str(ROOT / "skills" / "superset-sql-query" / "common")
    out = []
    for i, end_day in enumerate(END_DAYS[:n], 1):
        out_dir = ROOT / f"recent_month_large_mainstream_v5_roll_w{i}"
        cmd = [PY, "-u", str(ROOT / "v5_hybrid_recommendation.py"), "build",
               "--pairs", str(PAIRS), "--historical-profiles", str(PROFILES),
               "--current-nodes", str(CURRENT), "--source-summary", str(SUMMARY),
               "--output-dir", str(out_dir), "--start-day", START_DAY, "--end-day", end_day]
        print(f"[w{i}] {START_DAY}..{end_day} -> {out_dir.name}", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[w{i}] FAILED: {(r.stdout or '')[-400:]}{(r.stderr or '')[-400:]}")
            continue
        m = metric(out_dir / "v5_training_summary.json")
        m["window"] = f"{START_DAY}..{end_day}"
        out.append(m)
        print(f"[w{i}] test={m['test']} rows={m['rows']} miner R2={m['miner_r2']:.4f} plat R2={m['plat_r2']:.4f}")

    if not out:
        sys.exit("无有效窗口")
    print("\n=== 汇总 ===")
    print(f"{'window':<24}{'test':<24}{'rows':>7}{'minerR2':>10}{'minerRMSE':>11}{'platR2':>9}{'platRMSE':>10}{'mCov90':>8}{'pCov90':>8}")
    for m in out:
        print(f"{m['window']:<24}{m['test']:<24}{m['rows']:>7}{m['miner_r2']:>10.4f}{m['miner_rmse']:>11.4f}"
              f"{m['plat_r2']:>9.4f}{m['plat_rmse']:>10.4f}{(m['miner_cov90'] or 0):>8.3f}{(m['plat_cov90'] or 0):>8.3f}")
    for k, lab in [("miner_r2", "矿主R2"), ("plat_r2", "平台R2")]:
        v = [m[k] for m in out]
        print(f"{lab}: mean={sum(v)/len(v):.4f} min={min(v):.4f} max={max(v):.4f} std={(sum((x-sum(v)/len(v))**2 for x in v)/len(v))**0.5:.4f}")


if __name__ == "__main__":
    main()
