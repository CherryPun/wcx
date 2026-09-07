# -*- coding: utf-8 -*-
"""从主版 final_top3_e2e.csv 生成推荐分布可视化：
1) 横向条形图：各业务作为节点 Top1 被推荐的节点数，取前十（数量排序）
2) 环形图：前十合计占全部 Top1 推荐的百分比 + 其余
输出到 06_html/推荐业务分布_可视化.html（纯 SVG，离线可看）。
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "02_main_recommendation" / "final_top3_e2e.csv"
DATA = ROOT / "05_shared_data"
OUT = ROOT / "06_html" / "推荐业务分布_可视化.html"

outcomes = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
outcomes["business"] = outcomes["business"].astype(str).str.replace(r"\.0$", "", regex=True)
name_map = outcomes.dropna(subset=["business_name"]).drop_duplicates("business").set_index("business")["business_name"].to_dict()

d = pd.read_csv(CSV, dtype={"node_id": str})
d["top1_business"] = d["top1_business"].astype(str).str.replace(r"\.0$", "", regex=True)
vc = d["top1_business"].value_counts()
total = int(len(d))
top = vc.head(10)
rest = int(vc.iloc[10:].sum()) if len(vc) > 10 else 0
top_share = float(top.sum()) / total

palette = ["#2563eb", "#15803d", "#b45309", "#7c3aed", "#dc2626", "#0d9488",
           "#ca8a04", "#db2777", "#4f46e5", "#16a34a"]

# ---------- 横向条形图 ----------
maxv = int(top.max())
row_h = 30
top_h = len(top) * row_h + 52
bar_w = 720
parts = []
parts.append('<text x="0" y="22" font-size="15" fill="#17202a">被推荐为 Top1 的节点数（前十，数量排序）</text>')
for i, (biz, cnt) in enumerate(top.items()):
    y = 44 + i * row_h
    w = max(4, int(bar_w * cnt / maxv))
    parts.append(
        f'<rect x="160" y="{y}" width="{w}" height="18" rx="3" fill="{palette[i]}" opacity="0.9"/>'
        f'<text x="154" y="{y + 14}" text-anchor="end" font-size="12" fill="#17202a">{name_map.get(biz, "")} {biz}</text>'
        f'<text x="{168 + w}" y="{y + 14}" font-size="12" fill="#334155">{int(cnt)}</text>'
    )
bar_svg = ('<svg viewBox="0 0 900 ' + str(top_h + 8) + '" width="100%" style="max-width:900px">'
           + "".join(parts) + "</svg>")

# ---------- 环形图 ----------
cx = cy = 220
r = 150
width = 58
circ = 2 * math.pi * r
total_cnt = sum(int(c) for c in top) + rest
pieces = [(int(c), palette[i]) for i, (_, c) in enumerate(top.items())]
if rest > 0:
    pieces.append((rest, "#e2e8f0"))
start = -90
segs = []
for cnt, color in pieces:
    span = cnt / max(total_cnt, 1) * circ
    offset = -start / 360.0 * circ
    segs.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-dasharray="{span:.2f} {circ:.2f}" stroke-dashoffset="{offset:.2f}"/>'
    )
    start += span / circ * 360.0
legend_parts = []
for i, (biz, cnt) in enumerate(top.items()):
    legend_parts.append(
        f'<div class="lg"><span style="background:{palette[i]}"></span>{name_map.get(biz, "")} {biz}（{int(cnt)}）</div>')
if rest > 0:
    legend_parts.append(f'<div class="lg"><span style="background:#e2e8f0"></span>其余业务（{rest}）</div>')
donut_svg = (
    f'<svg viewBox="0 0 440 440" width="360">' + "".join(segs) +
    f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" font-size="15" fill="#17202a">Top10 合计</text>'
    f'<text x="{cx}" y="{cy + 18}" text-anchor="middle" font-size="26" font-weight="bold" fill="#17202a">{top_share * 100:.1f}%</text>'
    f'<text x="{cx}" y="{cy + 36}" text-anchor="middle" font-size="11" fill="#647181">占全部 Top1 推荐</text></svg>'
)

html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>推荐业务分布可视化</title>
<style>
body{{font-family:-apple-system,PingFang SC,Microsoft YaHei,sans-serif;margin:20px;background:#f6f7f9;color:#17202a}}
.card{{background:#fff;border-radius:12px;padding:18px 22px;margin:14px 0;box-shadow:0 8px 24px rgba(15,23,42,.06)}}
h2{{margin-top:0}}
.m{{color:#647181;font-size:13px}}
.lg{{display:flex;align-items:center;gap:8px;font-size:12px;margin:4px 0}}
.lg span{{width:14px;height:14px;border-radius:3px;display:inline-block;flex:none}}
.two{{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start}}
</style></head><body>
<h1>矿主推荐分布可视化</h1>
<p class="m">口径：主版 v1，共 {total} 个节点的 Top1 推荐。数据：final_top3_e2e.csv。数量排序取前十。</p>
<div class="card">{bar_svg}</div>
<div class="card two">
  <div>
    <h2>Top10 占全部推荐</h2>
    {donut_svg}
  </div>
  <div>
    <h3>图例</h3>
    {"".join(legend_parts)}
    <p class="m">其余 {rest} 个节点落在 Top10 之外、共 {max(len(vc) - 10, 0)} 个业务里。</p>
  </div>
</div>
</body></html>
"""
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT)
print("top10 share %", round(top_share * 100, 2))
