# -*- coding: utf-8 -*-
"""V2 分布可视化（对齐主版）：大节点/小节点各自 前十推荐业务横向条形 + 饼图(Top10 占全部+其余)。
读 large=exp_final(家族 token→代表 ID)、small=final_ant。输出两 HTML 到 _rerun/ 与 outputs/。"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "_rerun"
PUB = HERE.parent / "outputs"
o = pd.read_csv(HERE.parent.parent / "05_shared_data" / "outcomes_7d_named.csv",
                dtype={"node_id": str, "business": str}, low_memory=False)
o["business"] = o["business"].astype(str).str.replace(r"\.0$", "", regex=True)
mp = o.dropna(subset=["business_name"]).drop_duplicates("business").set_index("business")["business_name"].to_dict()
REP = {"FQCDNV": "10000292", "FQTZV": "10000281"}
def disp(b):
    return REP.get(str(b), str(b))

def build_html(fn, out_name, title, note):
    d = pd.read_csv(OUT / fn, dtype={"node_id": str})
    d["t1"] = d.top1_business.astype(str).map(disp)
    vc = d["t1"].value_counts()
    total = len(d)
    top = vc.head(10)
    rest = int(vc.iloc[10:].sum()) if len(vc) > 10 else 0
    share = top.sum() / total
    pal = ["#2563eb", "#15803d", "#b45309", "#7c3aed", "#dc2626", "#0d9488", "#ca8a04", "#db2777", "#4f46e5", "#16a34a"]
    # 横向条形
    mx = int(top.max()); row_h = 30
    bars = [f'<text x="0" y="22" font-size="15" fill="#17202a">被推荐为 Top1 的节点数（前十）</text>']
    for i, (b, cnt) in enumerate(top.items()):
        y = 44 + i * row_h
        w = max(4, int(430 * cnt / mx))
        lbl = f"{mp.get(b, '')} {b}"
        if len(lbl) > 26:
            lbl = lbl[:26] + "…"
        bars.append(f'<rect x="330" y="{y}" width="{w}" height="18" rx="3" fill="{pal[i]}"/>'
                    f'<text x="320" y="{y + 14}" text-anchor="end" font-size="11" fill="#17202a">{lbl}</text>'
                    f'<text x="{338 + w}" y="{y + 14}" font-size="11" fill="#334155">{int(cnt)}</text>')
    bar_svg = f'<svg viewBox="0 0 900 {44 + 10 * row_h + 10}" width="100%" style="max-width:900px">' + "".join(bars) + "</svg>"
    # 饼图（path 扇形，顶部顺时针）
    cx, cy, rr = 200, 200, 170
    total_cnt = sum(top) + rest
    segs = []
    ang = 0.0
    for i, (b, cnt) in enumerate(top.items()):
        span = cnt / total_cnt * 360.0
        a0, a1 = math.radians(ang - 90), math.radians(ang + span - 90)
        x0, y0 = cx + rr * math.cos(a0), cy + rr * math.sin(a0)
        x1, y1 = cx + rr * math.cos(a1), cy + rr * math.sin(a1)
        large = 1 if span > 180 else 0
        segs.append(f'<path d="M{cx},{cy} L{x0:.2f},{y0:.2f} A{rr},{rr} 0 {large} 1 {x1:.2f},{y1:.2f} Z" fill="{pal[i]}" stroke="#fff" stroke-width="1"/>')
        ang += span
    if rest > 0:
        span = rest / total_cnt * 360.0
        a0, a1 = math.radians(ang - 90), math.radians(ang + span - 90)
        x0, y0 = cx + rr * math.cos(a0), cy + rr * math.sin(a0)
        x1, y1 = cx + rr * math.cos(a1), cy + rr * math.sin(a1)
        large = 1 if span > 180 else 0
        segs.append(f'<path d="M{cx},{cy} L{x0:.2f},{y0:.2f} A{rr},{rr} 0 {large} 1 {x1:.2f},{y1:.2f} Z" fill="#e2e8f0" stroke="#fff" stroke-width="1"/>')
    pie_svg = (f'<svg viewBox="0 0 400 430" width="360">' + "".join(segs) +
               f'<text x="{cx}" y="420" text-anchor="middle" font-size="13" fill="#17202a">Top10 合计占全部 Top1 推荐 {share * 100:.1f}%</text></svg>')
    leg = []
    for i, (b, cnt) in enumerate(top.items()):
        leg.append(f'<div class="lg"><span style="background:{pal[i]}"></span>{mp.get(b, "")} {b}（{int(cnt)}，{cnt/total_cnt*100:.1f}%）</div>')
    if rest:
        leg.append(f'<div class="lg"><span style="background:#e2e8f0"></span>其余业务（{rest}，{rest/total_cnt*100:.1f}%）</div>')
    html = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>" + title + "</title><style>"
            "body{font-family:Microsoft YaHei,sans-serif;margin:16px;background:#f6f7f9}.card{background:#fff;border-radius:10px;padding:16px 20px;margin:12px 0}"
            ".lg{display:flex;align-items:center;gap:8px;font-size:12px;margin:3px 0}.lg span{width:13px;height:13px;border-radius:3px;flex:none}.two{display:flex;gap:26px;flex-wrap:wrap}.two>div{min-width:300px;flex:1}.m{color:#647181}"
            "</style></head><body><h2>" + title + "</h2><p class='m'>" + note + "</p>"
            "<div class='card'>" + bar_svg + "</div><div class='card two'><div><h3>Top10 占全部推荐</h3>" + pie_svg + "</div><div><h3>图例</h3>" + "".join(leg) + "</div></div></body></html>")
    (OUT / out_name).write_text(html, encoding="utf-8")
    (PUB / out_name).write_text(html, encoding="utf-8")
    print("wrote", out_name)

build_html("final_ant.csv", "推荐分布_小节点_ant.html",
           "V2 推荐分布_小节点（ant/盒子）",
           "口径：ant 池（5%）Top1 推荐分布；ant 为弱产品单独页。")
build_html("exp_final.csv", "推荐分布_大节点_large.html",
           "V2 推荐分布_大节点（全库 nonant，家族粒 21.0/51.7, n=2,994）",
           "口径：全库 large Top1 推荐分布（七牛家族 token 已映射为代表具体 ID）；n=3,000 时间外。")
