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
row_h = 34
top_h = len(top) * row_h + 52
label_col = 392      # 名称区域右边界（文字右对齐到此）
bar_max = 900 - label_col - 78   # 柱形可用宽度（右侧预留数量标签）
count_x0 = label_col + 8


def wrap_label(s: str, maxpx: float = 300.0, fs: int = 12) -> list:
    """按近似像素宽度把标签折成最多两行，避免越界被裁。"""
    rows, cur, curpx = [], "", 0.0
    for ch in s:
        w = fs * 0.5 if ord(ch) < 0x2E80 else fs   # 英文约半宽，中文全宽
        if cur and curpx + w > maxpx:
            rows.append(cur); cur, curpx = ch, w
        else:
            cur, curpx = cur + ch, curpx + w
    rows.append(cur)
    if len(rows) > 2:
        rows = [rows[0], rows[1][: max(1, int((maxpx / (fs * 1.0)) * 1.4))] + "…"]
    return rows or [""]


parts = []
parts.append('<text x="0" y="22" font-size="15" fill="#17202a">被推荐为 Top1 的节点数（前十，数量排序）</text>')
for i, (biz, cnt) in enumerate(top.items()):
    y = 44 + i * row_h
    w = max(4, int(bar_max * cnt / maxv))
    rows = wrap_label(f"{name_map.get(biz, '')} {biz}")
    tsp = "".join(
        f'<tspan x="{label_col - 8}"' + (f' dy="{13}"' if r > 0 else "") + f'>{txt}</tspan>'
        for r, txt in enumerate(rows))
    parts.append(
        f'<rect x="{label_col}" y="{y + 3}" width="{w}" height="18" rx="3" fill="{palette[i]}" opacity="0.9"/>'
        f'<text x="{label_col - 8}" y="{y + 16}" text-anchor="end" font-size="12" fill="#17202a">{tsp}</text>'
        f'<text x="{count_x0 + w}" y="{y + 17}" font-size="12" fill="#334155">{int(cnt)}</text>'
    )
bar_svg = ('<svg viewBox="0 0 900 ' + str(top_h + 8) + '" width="100%" style="max-width:900px">'
           + "".join(parts) + "</svg>")

# ---------- 饼图（每段一个实心扇形 path，不依赖 dashoffset） ----------
cx = cy = 200
r = 175
total_cnt = sum(int(c) for c in top) + rest
pieces = [(int(c), palette[i]) for i, (_, c) in enumerate(top.items())]
if rest > 0:
    pieces.append((rest, "#e2e8f0"))


def pie_path(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    """角度从顶部(12点)顺时针；返回闭合扇形 path。"""
    t0, t1 = math.radians(a0 - 90), math.radians(a1 - 90)
    x0, y0 = cx + r * math.cos(t0), cy + r * math.sin(t0)
    x1, y1 = cx + r * math.cos(t1), cy + r * math.sin(t1)
    large = 1 if (a1 - a0) > 180 else 0
    return (f"M{cx},{cy} L{x0:.2f},{y0:.2f} A{r},{r} 0 {large} 1 "
            f"{x1:.2f},{y1:.2f} Z")


slices = []
ang = 0.0
for cnt, color in pieces:
    span = cnt / max(total_cnt, 1) * 360.0
    slices.append(f'<path d="{pie_path(cx, cy, r, ang, ang + span)}" fill="{color}" '
                  f'stroke="#fff" stroke-width="1.5"/>')
    ang += span
pie_svg = (
    f'<svg viewBox="0 0 400 430" width="360">' + "".join(slices) +
    f'<text x="{cx}" y="420" text-anchor="middle" font-size="13" fill="#17202a">Top10 合计占全部 Top1 推荐 {top_share * 100:.1f}%</text>'
    "</svg>"
)
legend_parts = []
for i, (biz, cnt) in enumerate(top.items()):
    pct = int(cnt) / total_cnt * 100.0
    legend_parts.append(
        f'<div class="lg"><span style="background:{palette[i]}"></span>{name_map.get(biz, "")} {biz}（{int(cnt)}，{pct:.1f}%）</div>')
if rest > 0:
    pct = rest / total_cnt * 100.0
    legend_parts.append(f'<div class="lg"><span style="background:#e2e8f0"></span>其余业务（{rest}，{pct:.1f}%）</div>')

html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>推荐业务分布可视化</title>
<style>
body{{font-family:-apple-system,PingFang SC,Microsoft YaHei,sans-serif;margin:20px;background:#f6f7f9;color:#17202a}}
.card{{background:#fff;border-radius:12px;padding:18px 22px;margin:14px 0;box-shadow:0 8px 24px rgba(15,23,42,.06)}}
h2{{margin-top:0}}
.m{{color:#647181;font-size:13px}}
.lg{{display:flex;align-items:center;gap:8px;font-size:12px;margin:4px 0;white-space:normal;overflow:visible;overflow-wrap:anywhere}}
.lg span{{width:14px;height:14px;border-radius:3px;display:inline-block;flex:none}}
.two{{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start}}
.two>div{{min-width:300px;flex:1}}
</style></head><body>
<h1>矿主推荐分布可视化</h1>
<p class="m">口径：主版 v1，共 {total} 个节点的 Top1 推荐。数据：final_top3_e2e.csv。数量排序取前十。</p>
<div class="card">{bar_svg}</div>
<div class="card two">
  <div>
    <h2>Top10 占全部推荐</h2>
    {pie_svg}
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

