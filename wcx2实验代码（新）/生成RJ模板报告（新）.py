# -*- coding: utf-8 -*-
"""生成 RJ 模板报告（新）：复用 v5 前端报告模板（四页：模型概览/节点推荐/训练数据/集中度），
数据取自采用配置的 v5 报告数据；清理空值、精简未用字段后输出：
  节点推荐与分布（新）.html + 节点推荐与分布_data（新）.js
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "recent_month_large_mainstream_v5_cap20_curfull（新）"
HTML_IN = SRC / "v5_business_recommendation_report.html"
JS_IN = SRC / "v5_frontend_report_data.js"
HTML_OUT = ROOT / "节点推荐与分布（新）.html"
JS_OUT = ROOT / "节点推荐与分布_data（新）.js"

tmpl = HTML_IN.read_text(encoding="utf-8")
raw = JS_IN.read_text(encoding="utf-8")
data = json.loads(raw[raw.find("=") + 1:].rstrip().rstrip(";"))

# 需要的节点字段：数据键名出现在模板文本中的（+ 核心保留）
CORE = ["node_id", "province", "city", "isp", "network_schedule_type",
        "build_bandwidth_mbps", "recommendation_action", "recommendation_confidence",
        "current_business", "current_business_name", "current_business_status",
        "review_switch_candidate"]
for k in (1, 2, 3):
    CORE += [f"business_top{k}", f"business_name_top{k}", f"combined_score_top{k}",
             f"miner_unit_income_top{k}", f"platform_unit_profit_top{k}",
             f"estimated_miner_income_1d_top{k}", f"estimated_platform_profit_1d_top{k}",
             f"confidence_top{k}", f"reason_top{k}"]
recs = data.get("recommendations", [])
used = set(CORE)
if recs:
    for key in recs[0].keys():
        if key in used or re.search(re.escape(key), tmpl):
            used.add(key)


def clean(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return round(v, 5)
    return v


slim = {
    "summary": data.get("summary", {}),
    "recommendations": [{k: clean(r.get(k)) for k in used if k in r} for r in recs],
    "business_metrics": [{k: clean(v) for k, v in b.items()} for b in data.get("business_metrics", [])],
    "concentration": [{k: clean(v) for k, v in c.items()} for c in data.get("concentration", [])],
}
JS_OUT.write_text("window.V5_REPORT_DATA=" + json.dumps(slim, ensure_ascii=False, separators=(",", ":")) + ";", encoding="utf-8")


def pie_html(conc, topn=10):
    import math
    items = [(str(c.get("business_name_top1") or c.get("business_top1")),
              int(c.get("node_count") or 0), float(c.get("share") or 0)) for c in conc]
    items = [x for x in items if x[1] > 0][:topn]
    total = sum(n for _, n, _ in items)
    rest = max(int(sum(int(c.get("node_count") or 0) for c in conc)) - total, 0)
    pal = ["#1769aa", "#16835f", "#a86600", "#7c3aed", "#c9362b", "#0d9488", "#ca8a04", "#db2777", "#4f46e5", "#16a34a"]
    cx, cy, r = 190, 190, 160
    segs, ang = [], 0.0
    def seg(span, color):
        a0, a1 = math.radians(ang - 90), math.radians(ang + span - 90)
        x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        lg = 1 if span > 180 else 0
        return f'<path d="M{cx},{cy} L{x0:.2f},{y0:.2f} A{r},{r} 0 {lg} 1 {x1:.2f},{y1:.2f} Z" fill="{color}" stroke="#fff" stroke-width="1"/>'
    total_all = max(total + rest, 1)
    for i, (nm, n, sh) in enumerate(items):
        span = n / total_all * 360.0
        segs.append(seg(span, pal[i % len(pal)]))
        ang += span
    if rest > 0:
        segs.append(seg(rest / total_all * 360.0, "#e2e8f0"))
    leg = "".join(f"<div style='display:flex;gap:8px;align-items:center;font-size:13px;margin:3px 0'>"
                  f"<span style='width:12px;height:12px;background:{pal[i%len(pal)]};display:inline-block'></span>"
                  f"{nm}（{n}，{sh*100:.1f}%）</div>" for i, (nm, n, sh) in enumerate(items))
    if rest:
        leg += (f"<div style='display:flex;gap:8px;align-items:center;font-size:13px;margin:3px 0'>"
                f"<span style='width:12px;height:12px;background:#e2e8f0;display:inline-block'></span>其余业务（{rest}）</div>")
    svg = (f"<svg viewBox='0 0 380 400' width='330'>" + "".join(segs) +
           f"<text x='{cx}' y='385' text-anchor='middle' font-size='12' fill='#607080'>Top10 合计占比 {total/total_all*100:.1f}%</text></svg>")
    return ("<section class='panel'><h2>Top10 推荐业务占比（饼图）</h2>"
            "<div style='display:flex;gap:24px;flex-wrap:wrap;padding:12px 14px'>"
            f"<div>{svg}</div><div>{leg}</div></div></section>")


html = tmpl.replace("v5_frontend_report_data.js", JS_OUT.name).replace("<title>", "<title>wcx2_")
html = html.replace("</main>", pie_html(slim.get("concentration", [])) + "</main>", 1)
HTML_OUT.write_text(html, encoding="utf-8")
print("html", round(HTML_OUT.stat().st_size / 1024, 1), "KB; js", round(JS_OUT.stat().st_size / 1024, 1),
      "KB; kept fields", len(used), "of", len(recs[0]) if recs else 0)
print("has names:", "business_name_top1" in used, "| has amounts:",
      "estimated_miner_income_1d_top1" in used)
