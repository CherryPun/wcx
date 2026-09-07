# -*- coding: utf-8 -*-
"""最终交付生成：从 v1 主版 final_top3_e2e.csv 生成 HTML（含业务名）与说明用的摘要 JSON。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
DATA = TASK / "05_shared_data"
CSV = HERE / "final_top3_e2e.csv"

# 业务名映射
outcomes = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
outcomes["business"] = outcomes["business"].astype(str).str.replace(r"\.0$", "", regex=True)
name_map = (
    outcomes.dropna(subset=["business_name"])
    .drop_duplicates("business")
    .set_index("business")["business_name"]
    .to_dict()
)

d = pd.read_csv(CSV, dtype={"node_id": str}, low_memory=False)
d["top1_business"] = d["top1_business"].astype(str).str.replace(r"\.0$", "", regex=True)
for c in ["top2_business", "top3_business"]:
    d[c] = d[c].astype(str).str.replace(r"\.0$", "", regex=True)
for i in (1, 2, 3):
    d[f"top{i}_name"] = d[f"top{i}_business"].map(lambda b: name_map.get(b, ""))

cur = pd.read_csv(DATA / "cur7d_map.csv", dtype={"node_id": str}).set_index("node_id")


def cur_cell(nid: str) -> str:
    if nid not in cur.index or not int(cur.loc[nid, "has_cur"]):
        return "<td><span class='m'>—</span></td>"
    row = cur.loc[nid]
    return (f"<td>{row['cur_name'] or ''}<br>"
            f"<span class='m'>近7天结算 {row['cur_cost7']} · 平台利润 {row['cur_profit7']}</span></td>")


tbody = []
for _, r in d.iterrows():
    cells = [f"<td>{r.node_id}</td>", cur_cell(r.node_id)]
    for i in (1, 2, 3):
        cells.append(
            f"<td>{r[f'top{i}_name'] or ''} {r[f'top{i}_business']}<br>"
            f"<span class='m'>结算 {r[f'top{i}_pred_cost']} · 平台利润 {r[f'top{i}_pred_profit']}</span></td>"
        )
    tbody.append("<tr>" + "".join(cells) + "</tr>")

html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>全节点矿主优先推荐总览（最终版）</title>
<style>
body{{font-family:-apple-system,PingFang SC,Microsoft YaHei,sans-serif;margin:14px;background:#f6f7f9;color:#17202a}}
input{{padding:6px 10px;width:380px;margin-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:#fff}}
th{{position:sticky;top:0;background:#e8edf3}}
th,td{{border:1px solid #dfe3e8;padding:5px 7px;text-align:left;vertical-align:top}}
.m{{color:#647181}}
.card{{background:#fff;padding:12px 16px;border-radius:10px;margin-bottom:10px}}
</style></head><body>
<h2>全节点矿主优先推荐总览（最终版）</h2>
<div class="card">共 {len(d)} 个节点。「当前在跑」为该节点近30日内最近一段连续满7天实际账（取窗口内结算最高的业务），
与实际结算/平台利润同为实际口径；不满足（宽表无记录/不足7天/仅成本保底）则显示 —。
Top1~3 来自<b>相似节点真实跑过的业务</b>候选，按<b>矿主预测 7 天结算</b>主序 + 平台利润/容量软降权，数值为预测（7天金额），定位：候选/参考，不承诺最赚。</div>
<p><input id="q" placeholder="过滤 node_id 或业务" oninput="f()"></p>
<table><thead><tr><th>node_id</th><th>当前在跑<br><span class='m'>近7天实际</span></th><th>Top1</th><th>Top2</th><th>Top3</th></tr></thead>
<tbody>{''.join(tbody)}</tbody></table>
<script>var __f_rows=[],__f_txt=[];(function(){{var trs=document.querySelectorAll('tbody tr');for(var i=0;i<trs.length;i++){{__f_rows.push(trs[i]);__f_txt.push(trs[i].textContent.toLowerCase())}}}})();
function f(){{var q=document.getElementById('q').value.trim().toLowerCase(),n=__f_rows.length;for(var i=0;i<n;i++){{__f_rows[i].style.display=(!q||__f_txt[i].indexOf(q)>=0)?'':'none'}}}}</script>
</body></html>"""
html_out = TASK / "06_html" / "全节点推荐_最终版.html"
html_out.write_text(html, encoding="utf-8")
print("wrote HTML", html_out)
print("nodes", len(d))
