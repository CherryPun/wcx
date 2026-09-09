#!/usr/bin/env python3
"""Build an audited business/ISP historical price-signature report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SIGNATURE_FIELDS = [
    "business",
    "business_name",
    "daily_isp",
    "miner_price_type",
    "miner_price_item_id",
    "miner_price_item_name",
    "miner_unit_price",
    "miner_price_after_bonus",
    "customer_price_item_id",
    "customer_price_item_name",
    "customer_unit_price",
]


def build_price_signatures(facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = facts.copy()
    for column in SIGNATURE_FIELDS:
        if column not in frame:
            frame[column] = pd.NA
    for column in ["sample_weight", "cum_cost_7d", "cum_revenue_7d"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)
    frame["miner_unit_price"] = pd.to_numeric(frame["miner_unit_price"], errors="coerce")
    frame["customer_unit_price"] = pd.to_numeric(
        frame["customer_unit_price"], errors="coerce"
    )
    frame = frame[
        frame["miner_unit_price"].gt(0) | frame["customer_unit_price"].gt(0)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=SIGNATURE_FIELDS), pd.DataFrame(columns=SIGNATURE_FIELDS)

    frame["nominal_unit_spread"] = (
        frame["customer_unit_price"] - frame["miner_unit_price"]
    )
    frame["realized_platform_profit"] = frame["cum_revenue_7d"] - frame["cum_cost_7d"]
    grouped = (
        frame.groupby(SIGNATURE_FIELDS, dropna=False, sort=False)
        .agg(
            node_days=("node_id", "size"),
            effective_support=("sample_weight", "sum"),
            nodes=("node_id", "nunique"),
            first_observed_day=("sample_day", "min"),
            last_observed_day=("sample_day", "max"),
            realized_miner_income=("cum_cost_7d", "sum"),
            realized_customer_revenue=("cum_revenue_7d", "sum"),
            realized_platform_profit=("realized_platform_profit", "sum"),
            nominal_unit_spread=("nominal_unit_spread", "first"),
        )
        .reset_index()
    )
    base_keys = ["business", "business_name", "daily_isp"]
    grouped["signature_count_in_business_isp"] = grouped.groupby(base_keys)[
        "business"
    ].transform("size")
    grouped["business_isp_effective_support"] = grouped.groupby(base_keys)[
        "effective_support"
    ].transform("sum")
    grouped["signature_support_share"] = (
        grouped["effective_support"]
        / grouped["business_isp_effective_support"].where(
            grouped["business_isp_effective_support"].gt(0)
        )
    ).fillna(0.0)
    grouped = grouped.sort_values(
        [*base_keys, "effective_support", "node_days", "last_observed_day"],
        ascending=[True, True, True, False, False, False],
    ).reset_index(drop=True)
    grouped["is_dominant_signature"] = ~grouped.duplicated(base_keys)
    dominant = grouped[grouped["is_dominant_signature"]].copy().reset_index(drop=True)
    return grouped, dominant


def summary_payload(
    facts: pd.DataFrame,
    signatures: pd.DataFrame,
    dominant: pd.DataFrame,
) -> dict[str, Any]:
    miner_covered = pd.to_numeric(
        facts.get("miner_unit_price"), errors="coerce"
    ).gt(0)
    customer_covered = pd.to_numeric(
        facts.get("customer_unit_price"), errors="coerce"
    ).gt(0)
    miner_conflict = facts.get(
        "miner_price_conflict", pd.Series(False, index=facts.index)
    ).fillna(False).astype(bool)
    customer_conflict = facts.get(
        "customer_price_conflict", pd.Series(False, index=facts.index)
    ).fillna(False).astype(bool)
    return {
        "version": "business_price_signatures_v1",
        "date_window": {
            "start": str(facts["sample_day"].min()),
            "end": str(facts["sample_day"].max()),
        },
        "daily_facts": int(len(facts)),
        "businesses": int(facts["business"].nunique()),
        "business_isp_groups": int(len(dominant)),
        "exact_price_signatures": int(len(signatures)),
        "miner_price_coverage": float(miner_covered.mean()),
        "customer_price_coverage": float(customer_covered.mean()),
        "miner_price_conflict_node_days": int(miner_conflict.sum()),
        "customer_price_conflict_node_days": int(customer_conflict.sum()),
        "customer_price_conflict_share": float(customer_conflict.mean()),
        "multiple_signature_business_isp_groups": int(
            dominant["signature_count_in_business_isp"].gt(1).sum()
        ),
        "definition": {
            "grain": (
                "business + node ISP + each clean node-day's amount-dominant "
                "miner/customer price signature"
            ),
            "dominant_signature": (
                "within a node-day, select the signature carrying the largest absolute final amount; "
                "within a business/ISP group, select the largest effective support"
            ),
            "effective_support": "sum of 1 / consecutive valid days within each node-business run",
            "warning": (
                "Prices are historically observed settlement fields, not contractual quotations. "
                "Different price types and item IDs are never averaged together. A displayed "
                "nominal spread does not replace realized platform profit."
            ),
        },
    }


def write_html(path: Path, data_js_name: str) -> None:
    template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>业务运营商历史计价矩阵</title>
<style>
*{box-sizing:border-box}html,body{max-width:100%;overflow-x:hidden}body{margin:0;background:#f4f6f8;color:#17212b;font:14px/1.45 system-ui,-apple-system,"PingFang SC",sans-serif;letter-spacing:0}header{background:#fff;border-bottom:1px solid #d9e0e7;padding:20px 28px}h1{font-size:22px;margin:0 0 6px}p{margin:0;color:#607080;overflow-wrap:anywhere}.wrap{width:100%;max-width:100%;padding:18px 28px;overflow:hidden}.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:14px}.metric{min-width:0;background:#fff;border:1px solid #d9e0e7;border-radius:6px;padding:12px}.metric b{display:block;font-size:20px}.toolbar{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}input,select{min-width:0;height:36px;border:1px solid #bcc8d4;border-radius:5px;background:#fff;padding:0 10px}input{flex:1 1 240px;max-width:420px}select{flex:0 1 160px}.table{width:100%;max-width:100%;overflow:auto;background:#fff;border:1px solid #d9e0e7;border-radius:6px;max-height:70vh}table{border-collapse:collapse;width:100%;min-width:1500px}th,td{padding:9px 10px;border-bottom:1px solid #e5eaf0;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#eef2f6;z-index:1;font-size:12px}tr:hover{background:#f7fafc}.warn{color:#9b4d00}.muted{color:#718096}@media(max-width:800px){header,.wrap{padding:14px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body>
<header><h1>业务 + 运营商历史计价矩阵</h1><p id="subtitle"></p></header>
<main class="wrap"><section class="metrics" id="metrics"></section><div class="toolbar"><input id="search" placeholder="搜索业务、运营商、计价项"><select id="scope"><option value="dominant">主计价签名</option><option value="all">全部计价签名</option></select></div><div class="table"><table><thead><tr><th>业务</th><th>节点运营商</th><th>矿主计价类型</th><th>矿主计价项</th><th>矿主单价</th><th>客户计价项</th><th>客户单价</th><th>名义价差</th><th>有效支持</th><th>节点日</th><th>节点数</th><th>签名占比</th><th>观测日期</th></tr></thead><tbody id="rows"></tbody></table></div></main>
<script src="__DATA_JS__"></script><script>
const D=window.BUSINESS_PRICE_REPORT_DATA,S=D.summary;const pct=v=>(100*Number(v||0)).toFixed(1)+'%';const num=v=>Number(v||0).toLocaleString('zh-CN',{maximumFractionDigits:2});
document.querySelector('#subtitle').textContent=`${S.date_window.start} 至 ${S.date_window.end} · 历史观测价格，不代表合同报价`;
document.querySelector('#metrics').innerHTML=[['业务',S.businesses],['业务运营商组合',S.business_isp_groups],['主价格签名',S.exact_price_signatures],['矿主价格覆盖',pct(S.miner_price_coverage)],['客户价格覆盖',pct(S.customer_price_coverage)],['客户日内多签名',pct(S.customer_price_conflict_share)]].map(x=>`<div class="metric"><span class="muted">${x[0]}</span><b>${x[1]}</b></div>`).join('');
const search=document.querySelector('#search'),scope=document.querySelector('#scope'),body=document.querySelector('#rows');
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function render(){const q=search.value.trim().toLowerCase();const source=scope.value==='all'?D.signatures:D.dominant;const rows=source.filter(r=>!q||[r.business_name,r.business,r.daily_isp,r.miner_price_item_name,r.customer_price_item_name,r.miner_price_type].join(' ').toLowerCase().includes(q));body.innerHTML=rows.map(r=>`<tr><td><b>${esc(r.business_name)}</b><br><span class="muted">${esc(r.business)}</span></td><td>${esc(r.daily_isp)}</td><td>${esc(r.miner_price_type)}</td><td>${esc(r.miner_price_item_name)}<br><span class="muted">${esc(r.miner_price_item_id)}</span></td><td>${num(r.miner_unit_price)}</td><td>${esc(r.customer_price_item_name)}<br><span class="muted">${esc(r.customer_price_item_id)}</span></td><td>${num(r.customer_unit_price)}</td><td class="${Number(r.nominal_unit_spread)<0?'warn':''}">${num(r.nominal_unit_spread)}</td><td>${num(r.effective_support)}</td><td>${num(r.node_days)}</td><td>${num(r.nodes)}</td><td>${pct(r.signature_support_share)}</td><td>${esc(r.first_observed_day)} ~ ${esc(r.last_observed_day)}</td></tr>`).join('')}search.addEventListener('input',render);scope.addEventListener('change',render);render();
</script></body></html>"""
    path.write_text(template.replace("__DATA_JS__", data_js_name), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    facts = pd.read_csv(args.facts, dtype={"business": "string", "node_id": "string"}, low_memory=False)
    signatures, dominant = build_price_signatures(facts)
    summary = summary_payload(facts, signatures, dominant)
    signatures.to_csv(args.output_dir / "business_price_signatures.csv", index=False)
    dominant.to_csv(args.output_dir / "business_isp_dominant_prices.csv", index=False)
    (args.output_dir / "business_price_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload = {"summary": summary, "dominant": dominant.to_dict("records"), "signatures": signatures.to_dict("records")}
    (args.output_dir / "business_price_report_data.js").write_text(
        "window.BUSINESS_PRICE_REPORT_DATA = "
        + json.dumps(payload, ensure_ascii=False, default=str)
        + ";\n",
        encoding="utf-8",
    )
    write_html(args.output_dir / "business_price_report.html", "business_price_report_data.js")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
