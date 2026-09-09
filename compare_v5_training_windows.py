#!/usr/bin/env python3
"""Compare two V5 training windows on the same temporal test period."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def metric(summary: dict[str, Any], *path: str) -> Any:
    value: Any = summary
    for key in path:
        value = value[key]
    return value


def same_test_window(primary: dict[str, Any], challenger: dict[str, Any]) -> bool:
    keys = ["test_start", "test_end"]
    return all(
        primary["temporal_windows"].get(key) == challenger["temporal_windows"].get(key)
        for key in keys
    )


def compare(primary: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    if not same_test_window(primary, challenger):
        raise RuntimeError("V5 windows do not share the same test period")
    p_val = primary["temporal_validation"]
    c_val = challenger["temporal_validation"]
    p_cred = primary["credibility"]
    c_cred = challenger["credibility"]
    p_conc = primary["recommendation_concentration"]
    c_conc = challenger["recommendation_concentration"]
    p_businesses = primary["training_data"]["candidate_businesses"]
    c_businesses = challenger["training_data"]["candidate_businesses"]

    gates = {
        "same_test_window": True,
        "miner_rmse_within_10pct": c_val["miner"]["rmse"] <= p_val["miner"]["rmse"] * 1.10,
        "platform_rmse_within_10pct": c_val["platform"]["rmse"] <= p_val["platform"]["rmse"] * 1.10,
        "candidate_coverage_at_least_95pct": c_businesses >= p_businesses * 0.95,
        "concentration_acceptable": c_conc["concentration_risk"] != "high",
        "low_confidence_increase_at_most_5pp": (
            c_cred["low_confidence_node_share"]
            <= p_cred["low_confidence_node_share"] + 0.05
        ),
    }
    promote = all(gates.values())

    def window_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
        validation = summary["temporal_validation"]
        credibility = summary["credibility"]
        concentration = summary["recommendation_concentration"]
        return {
            "name": name,
            "start": summary["date_window"]["start"],
            "end": summary["date_window"]["end"],
            "raw_node_days": summary["training_data"]["raw_node_days"],
            "effective_support": summary["training_data"]["effective_run_support"],
            "nodes": summary["training_data"]["nodes"],
            "candidate_businesses": summary["training_data"]["candidate_businesses"],
            "miner_rmse": validation["miner"]["rmse"],
            "miner_r2": validation["miner"]["r2"],
            "platform_rmse": validation["platform"]["rmse"],
            "platform_r2": validation["platform"]["r2"],
            "observed_hit_at_3": summary["ranking_and_dr_diagnostics"]["observed_business_hit_at_3"],
            "maximum_top1_share": concentration["maximum_top1_share"],
            "effective_business_count": concentration["effective_business_count"],
            "concentration_risk": concentration["concentration_risk"],
            "low_confidence_share": credibility["low_confidence_node_share"],
            "platform_interval_crosses_zero_share": credibility[
                "top1_platform_interval_crosses_zero_share"
            ],
            "credibility": credibility["level"],
        }

    return {
        "version": "v5_dual_window_comparison_v1",
        "test_window": {
            "start": primary["temporal_windows"]["test_start"],
            "end": primary["temporal_windows"]["test_end"],
        },
        "windows": [window_row("31-day primary", primary), window_row("14-day challenger", challenger)],
        "challenger_gates": gates,
        "decision": {
            "promote_challenger": promote,
            "selected": "14-day challenger" if promote else "31-day primary",
            "use_challenger_for": (
                "primary recommendations and recency signal"
                if promote
                else "recency monitoring and 14-day capacity forecast only"
            ),
            "reason": (
                "The challenger passed accuracy, coverage, concentration, and confidence gates."
                if promote
                else "The challenger did not pass all coverage, concentration, and confidence gates."
            ),
        },
    }


def markdown(payload: dict[str, Any]) -> str:
    primary, challenger = payload["windows"]
    pct = lambda value: f"{value * 100:.1f}%"
    lines = [
        "# V5 双窗口训练对比",
        "",
        f"验证窗口：{payload['test_window']['start']} 至 {payload['test_window']['end']}（两套模型完全一致）。",
        "",
        "| 指标 | 31 天主模型 | 14 天挑战模型 |",
        "|---|---:|---:|",
        f"| 干净节点日 | {primary['raw_node_days']:,} | {challenger['raw_node_days']:,} |",
        f"| 有效连续段支持 | {primary['effective_support']:,.0f} | {challenger['effective_support']:,.0f} |",
        f"| 候选业务 | {primary['candidate_businesses']} | {challenger['candidate_businesses']} |",
        f"| 矿主 RMSE | {primary['miner_rmse']:.6f} | {challenger['miner_rmse']:.6f} |",
        f"| 矿主 R² | {primary['miner_r2']:.3f} | {challenger['miner_r2']:.3f} |",
        f"| 平台 RMSE | {primary['platform_rmse']:.6f} | {challenger['platform_rmse']:.6f} |",
        f"| 平台 R² | {primary['platform_r2']:.3f} | {challenger['platform_r2']:.3f} |",
        f"| Top1 最大集中度 | {pct(primary['maximum_top1_share'])} | {pct(challenger['maximum_top1_share'])} |",
        f"| 有效业务数 | {primary['effective_business_count']:.2f} | {challenger['effective_business_count']:.2f} |",
        f"| 低可信节点占比 | {pct(primary['low_confidence_share'])} | {pct(challenger['low_confidence_share'])} |",
        f"| Top1 平台区间跨零 | {pct(primary['platform_interval_crosses_zero_share'])} | {pct(challenger['platform_interval_crosses_zero_share'])} |",
        "",
        f"## 结论：保留 {payload['decision']['selected']}",
        "",
        "14 天窗口用于近期趋势监控和容量预测，不直接替换生产推荐模型。",
        "",
        "## 挑战模型门槛",
        "",
    ]
    labels = {
        "same_test_window": "同一测试窗口",
        "miner_rmse_within_10pct": "矿主 RMSE 不恶化超过 10%",
        "platform_rmse_within_10pct": "平台 RMSE 不恶化超过 10%",
        "candidate_coverage_at_least_95pct": "候选业务覆盖至少为主模型的 95%",
        "concentration_acceptable": "Top1 集中度风险不是高",
        "low_confidence_increase_at_most_5pp": "低可信占比增幅不超过 5 个百分点",
    }
    lines.extend(
        f"- {'通过' if passed else '未通过'}：{labels[key]}"
        for key, passed in payload["challenger_gates"].items()
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    challenger = json.loads(args.challenger.read_text(encoding="utf-8"))
    payload = compare(primary, challenger)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v5_dual_window_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "v5_dual_window_comparison.md").write_text(
        markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
