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
HTML_OUT.write_text(
    tmpl.replace("v5_frontend_report_data.js", JS_OUT.name).replace("<title>", "<title>wcx2_"),
    encoding="utf-8",
)
print("html", round(HTML_OUT.stat().st_size / 1024, 1), "KB; js", round(JS_OUT.stat().st_size / 1024, 1),
      "KB; kept fields", len(used), "of", len(recs[0]) if recs else 0)
print("has names:", "business_name_top1" in used, "| has amounts:",
      "estimated_miner_income_1d_top1" in used)
