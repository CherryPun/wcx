# -*- coding: utf-8 -*-
"""[wcx2 新增] 流水线（新）：一个包、四条命令

  ingest   # 抽数 + 归因 + 账本落盘（调用 build_v3_daily_business_training.py）
  quote    # 池报价（调用 v5_hybrid_recommendation.py build，默认口径 bw>=500 + eval winsorize）
  allocate # 容量约束分配（调用 v6_capacity_aware_allocation.py build）
  report   # 复核清单 / 看板 / 可信度（生成RJ模板报告、门槛看板、业务级可信度）

路径来自 `pipeline_config.json`，凭证只从环境变量读取（SUPERSET_* / NIULINK_AUTH）。
`--dry-run` 只打印将执行的命令。
用法：python "wcx2流水线（新）.py" quote [--dry-run] [--config pipeline_config.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pipeline_config.json"


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("ingest", "quote", "allocate", "report"):
        cfg.setdefault(key, {})
    return cfg


def run(cmd: list[str], env: dict, dry: bool) -> int:
    printable = " ".join(f'"{c}"' if " " in c or "（" in c else c for c in cmd)
    print(f"[run] {printable}")
    if dry:
        return 0
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, encoding="utf-8", errors="replace")
    return r.returncode


def build_env(cfg: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for k, v in cfg.get("env", {}).items():
        env.setdefault(k, str(ROOT / v) if "/" in v else v)
    return env


def cmd_ingest(cfg: dict, env: dict, dry: bool) -> int:
    c, w = cfg["ingest"], cfg["window"]
    return run([sys.executable, "-u", str(ROOT / "build_v3_daily_business_training.py"),
                "--source-dir", str(ROOT / c["source_dir"]),
                "--output-dir", str(ROOT / c["output_dir"]),
                "--current-nodes", str(ROOT / c["current_nodes"]),
                "--allowlist", str(ROOT / "mainstream_business_allowlist.csv"),
                "--business-map", str(ROOT / "v1_business_name_map_enriched.csv"),
                "--start-day", w["start_day"], "--end-day", w["end_day"],
                "--min-business-support", "30", "--refresh"], env, dry)


def cmd_quote(cfg: dict, env: dict, dry: bool) -> int:
    c, w = cfg["quote"], cfg["window"]
    return run([sys.executable, "-u", str(ROOT / "v5_hybrid_recommendation.py"), "build",
                "--pairs", str(ROOT / c["pairs"]),
                "--historical-profiles", str(ROOT / c["historical_profiles"]),
                "--latest-pressure-profiles", str(ROOT / c["latest_pressure_profiles"]),
                "--current-nodes", str(ROOT / c["current_nodes"]),
                "--source-summary", str(ROOT / c["source_summary"]),
                "--business-map", str(ROOT / c["business_map"]),
                "--output-dir", str(ROOT / c["output_dir"]),
                "--min-build-bandwidth", str(c.get("min_build_bandwidth", 500)),
                "--eval-winsorize", str(c.get("eval_winsorize", 1)),
                "--start-day", w["start_day"], "--end-day", w["end_day"]], env, dry)


def cmd_allocate(cfg: dict, env: dict, dry: bool) -> int:
    c, w = cfg["allocate"], cfg["window"]
    return run([sys.executable, "-u", str(ROOT / "v6_capacity_aware_allocation.py"), "build",
                "--source-dir", str(ROOT / c["source_dir"]),
                "--recommendations", str(ROOT / c["recommendations"]),
                "--output-dir", str(ROOT / c["output_dir"]),
                "--start-day", w["start_day"], "--end-day", w["end_day"],
                "--target-utilization", str(c.get("target_utilization", 0.7))], env, dry)


def cmd_report(cfg: dict, env: dict, dry: bool) -> int:
    c = cfg["report"]
    rc = run([sys.executable, str(ROOT / "wcx2实验代码（新）" / "门槛看板（新）.py"), str(ROOT / c["v5_dir"])], env, dry)
    if rc:
        return rc
    rc = run([sys.executable, str(ROOT / "wcx2实验代码（新）" / "业务级可信度（新）.py"), str(ROOT / c["v5_dir"])], env, dry)
    if rc:
        return rc
    return run([sys.executable, str(ROOT / "wcx2实验代码（新）" / "生成RJ模板报告（新）.py"), str(ROOT / c["v5_dir"])], env, dry)


COMMANDS = {"ingest": cmd_ingest, "quote": cmd_quote, "allocate": cmd_allocate, "report": cmd_report}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    env = build_env(cfg)
    return COMMANDS[args.command](cfg, env, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
