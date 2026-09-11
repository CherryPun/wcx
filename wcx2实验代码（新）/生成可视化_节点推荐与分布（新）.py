# -*- coding: utf-8 -*-
"""生成 节点推荐与分布（新）.html —— 采用 RJ 报告风格（深色头 + 面板 + KPI + 表格 + 搜索）。
数据：采用配置的 v5 推荐 + v6 容量节点报告。输出单文件 HTML（自包含）。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V5 = ROOT / "recent_month_large_mainstream_v5_default（新）"
V6 = ROOT / "recent_month_large_mainstream_v6_capacity_default（新）"
OUT = ROOT / "节点推荐与分布（新）.html"


def esc(x):
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


rec = pd.read_csv(V5 / "v5_node_recommendations.csv", dtype={"node_id": "str"}, low_memory=False)
nr = pd.read_csv(V6 / "v6_capacity_node_report.csv", dtype={"node_id": "str"}, low_memory=False)
summary = json.loads((V6 / "v6_capacity_summary.json").read_text(encoding="utf-8"))

keep = ["node_id", "province", "isp", "current_business", "current_business_name",
        "business_top1", "business_name_top1", "business_top2", "business_top3", "combined_score_top1"]
d = rec[[c for c in keep if c in rec.columns]].copy()
d = d.merge(nr[["node_id", "planned_business_name", "capacity_report_category"]], on="node_id", how="left")
dist = d["business_top1"].astype(str).value_counts().head(10)
total = len(d)

kpi = [
    ("节点数", f"{total:,}"),
    ("规划切换", f"{summary.get('planned_change_nodes', 0):,}"),
    ("容量拦截", f"{summary.get('capacity_blocked_nodes', 0):,}"),
    ("有效容量池", f"{summary.get('capacity_pools_with_sufficient_evidence', 0):,}"),
    ("全切溢出(Gbps)", f"{summary.get('unconstrained_top1_overflow_mbps', 0)/1000:,.0f}"),
]
kpi_html = "".join(f"<div class='kpi'><span class='muted'>{k}</span><b>{v}</b></div>" for k, v in kpi)

pal = ["#1769aa", "#16835f", "#a86600", "#7c3aed", "#c9362b", "#0d9488", "#ca8a04", "#db2777", "#4f46e5", "#16a34a"]
mx = int(dist.max()) if len(dist) else 1
bars = []
for i, (b, c) in enumerate(dist.items()):
    w = max(4, int(430 * c / mx))
    bars.append(f"<div class='bar'><span class='lbl'>{esc(b)}</span>"
                f"<span class='track'><i style='width:{w/430*100:.1f}%;background:{pal[i%len(pal)]}'></i></span>"
                f"<span class='val'>{int(c)}（{c/total*100:.1f}%）</span></div>")
dist_html = "".join(bars)

rows = []
for _, r in d.iterrows():
    rows.append("<tr><td>%s</td><td>%s/%s</td><td>%s<br><span class='muted'>%s</span></td>"
                "<td><b>%s</b><br><span class='muted'>%s</span></td><td>%s</td><td>%s</td>"
                "<td>%s</td><td>%s</td></tr>" % (
                    esc(r["node_id"]), esc(r.get("province", "")), esc(r.get("isp", "")),
                    esc(r.get("current_business_name", "")), esc(r.get("current_business", "")),
                    esc(r.get("business_name_top1", "")), esc(r.get("business_top1", "")),
                    esc(r.get("business_top2", "")), esc(r.get("business_top3", "")),
                    esc(r.get("planned_business_name", "") or "-"), esc(r.get("capacity_report_category", "") or "-")))
tbody = "".join(rows)

html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>节点推荐与分布（wcx2 采用配置）</title><style>
:root{{--bg:#f4f6f8;--panel:#fff;--line:#d9e0e7;--text:#18212b;--muted:#607080}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
header{{background:#132a3a;color:#fff;padding:18px 26px;border-bottom:4px solid #25a57a}}header h1{{font-size:22px;margin:0 0 4px}}
header p{{margin:0;color:#c9d6df;font-size:13px}}main{{padding:16px 22px;max-width:1800px;margin:auto}}
.notice{{background:#fff8e5;border:1px solid #e9c56b;padding:10px 14px;margin-bottom:12px;font-size:13px}}
.kpis{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:10px;margin-bottom:12px}}
.kpi,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:6px}}.kpi{{padding:12px}}
.kpi b{{display:block;font-size:20px;margin-top:2px}}.muted{{color:var(--muted)}}.panel{{margin-bottom:14px}}
.panel h2{{font-size:16px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}}
.bar{{display:flex;align-items:center;gap:10px;padding:5px 14px;font-size:13px}}.bar .lbl{{width:220px;text-align:right}}
.track{{flex:1;background:#eef2f5;height:14px;border-radius:3px;overflow:hidden}}.track i{{display:block;height:14px}}
.bar .val{{width:150px;color:#334155}}
.tools{{padding:10px 12px;border-bottom:1px solid var(--line)}}input{{border:1px solid #aeb9c4;border-radius:4px;padding:7px 9px;min-width:280px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid #e6ebef;vertical-align:top}}
th{{position:sticky;top:0;background:#edf2f5;z-index:1}}tr:hover td{{background:#f7fafc}}.wrap{{overflow:auto;max-height:70vh}}
</style></head><body>
<header><h1>节点推荐与分布（wcx2 采用配置）</h1>
<p>窗口 2026-08-10~09-09 ｜ 大节点（专线/汇聚，非 IDC）｜ 主流 allowlist ｜ 配置：V5 时效权重 HL7+cap1.5（默认）</p></header>
<main>
<div class="notice">本页为人工规划参考，<b>不自动切业务</b>；推荐=候选/参考，金额为预测，不承诺兑现。</div>
<section class="kpis">{kpi_html}</section>
<section class="panel"><h2>推荐业务分布（Top1，前十）</h2>{dist_html}</section>
<section class="panel"><h2>节点推荐列表</h2>
<div class="tools"><input id="q" placeholder="搜索 node_id / 业务 / 省份 / 状态"></div>
<div class="wrap"><table><thead><tr><th>node_id</th><th>省/运营商</th><th>当前业务</th><th>Top1 推荐</th><th>Top2</th><th>Top3</th><th>规划业务</th><th>状态</th></tr></thead>
<tbody id="rows">{tbody}</tbody></table></div></section>
</main>
<script>
var q=document.getElementById('q');q.addEventListener('input',function(){{var v=q.value.toLowerCase();document.querySelectorAll('#rows tr').forEach(function(r){{r.style.display=r.innerText.toLowerCase().indexOf(v)>=0?'':'none';}});}});
</script></body></html>"""

OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, round(OUT.stat().st_size / 1024, 1), "KB; nodes", total)
