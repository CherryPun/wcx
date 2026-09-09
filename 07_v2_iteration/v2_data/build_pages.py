# -*- coding: utf-8 -*-
"""D：生成 全节点_large.html / 全节点_ant.html（v2 交付页，不动主版 06_html）。"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "05_shared_data"
OUT = HERE / "_rerun"
o = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
o["business"] = o["business"].astype(str).str.replace(r"\.0$", "", regex=True)
mp = o.dropna(subset=["business_name"]).drop_duplicates("business").set_index("business")["business_name"].to_dict()


def page(fn_csv, out_html, title, note):
    d = pd.read_csv(OUT / fn_csv, dtype={"node_id": str})
    rows = []
    for _, r in d.iterrows():
        cells = [f"<td>{r.node_id}</td>"]
        for i in (1, 2, 3):
            b = str(r[f"top{i}_business"]).replace(".0", "")
            cells.append(f"<td>{mp.get(b, '')} {b}<br><span class='m'>结算 {r[f'top{i}_pred_cost']} · 利润 {r[f'top{i}_pred_profit']}</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    html = (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title><style>"
            "body{font-family:Microsoft YaHei,sans-serif;margin:14px}table{border-collapse:collapse;width:100%;font-size:12px}"
            "th,td{border:1px solid #dfe3e8;padding:5px 7px;text-align:left}.m{color:#647181}</style></head><body>"
            f"<h2>{title}</h2><p class='m'>{note}</p>"
            "<table><thead><tr><th>node_id</th><th>Top1</th><th>Top2</th><th>Top3</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></body></html>")
    (OUT / out_html).write_text(html, encoding="utf-8")
    print("wrote", out_html, "rows", len(d))


page("final_gate.csv", "全节点_large.html",
     "全节点推荐_large（大节点/专线+汇聚，nonant 409）",
     "时间外 n=82：hit1 18.3% / hit3 48.8% / WAPE 0.56-0.74。定位：候选/参考，不承诺最赚。")
page("final_ant.csv", "全节点_ant.html",
     "全节点推荐_ant（盒子/小盒子，弱产品单独页）",
     "ant 为弱产品：独立模型命中约 8%，混训 36% 不作交付。勿与 large 混排。")
