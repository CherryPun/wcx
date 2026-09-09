# -*- coding: utf-8 -*-
"""V2 分布可视化（对齐主版）：大节点/小节点 各两饼 —— ①Top1 推荐分布 ②当前在跑(近7天实际)分布。
- 推荐 Top1：读 large=exp_final(家族 token→代表 ID)、small=final_ant；
- 当前在跑：解析已交付页面(全节点_large_fullpool.html / 全节点_ant.html)的“当前在跑(近7天实际)”列（含业务 ID），
  缺当前(近7天无有效结算/空载)节点不计入该饼，导语标注 n。
- 输出两 HTML 到 _rerun/ 与 outputs/。"""
from __future__ import annotations
import math
import re
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

IDRE = re.compile(r"(\d{6,12})")


def parse_current_ids(hpath: Path) -> dict:
    """从已交付全节点页解析 当前在跑 业务 ID（无有效当前 → None）。"""
    s = hpath.read_text(encoding="utf-8")
    tb = s[s.find("<tbody>"):s.find("</tbody>")]
    out = {}
    for row in re.findall(r"<tr>(.*?)</tr>", tb, flags=re.S):
        tds = re.findall(r"<td>(.*?)</td>", row, flags=re.S)
        if len(tds) < 2:
            continue
        nid = re.sub(r"<[^>]+>", "", tds[0]).strip()
        first = re.sub(r"<[^>]+>", "", tds[1].split("<br")[0]).strip()
        m = IDRE.findall(first)
        out[nid] = m[-1] if m else None
    return out


def label(b):
    nm = str(mp.get(b, "") or "")
    return f"{nm} {b}".strip()


def pie_of(series, n_title, pal, total_note):
    """series: 业务级计数(全量 Top1 推荐或当前)。返回 (bar_svg, pie_svg, legend_html)。"""
    total = len(series)
    vc = series.value_counts()
    top = vc.head(10)
    rest = int(vc.iloc[10:].sum()) if len(vc) > 10 else 0
    mx = int(top.max())
    row_h = 30
    bars = [f'<text x="0" y="22" font-size="15" fill="#17202a">{n_title}</text>']
    for i, (b, cnt) in enumerate(top.items()):
        y = 44 + i * row_h
        w = max(4, int(430 * cnt / mx))
        lbl = label(b)
        if len(lbl) > 26:
            lbl = lbl[:26] + "…"
        bars.append(f'<rect x="330" y="{y}" width="{w}" height="18" rx="3" fill="{pal[i]}"/>'
                    f'<text x="320" y="{y + 14}" text-anchor="end" font-size="11" fill="#17202a">{lbl}</text>'
                    f'<text x="{338 + w}" y="{y + 14}" font-size="11" fill="#334155">{int(cnt)}</text>')
    bar_svg = f'<svg viewBox="0 0 900 {44 + 10 * row_h + 10}" width="100%" style="max-width:900px">' + "".join(bars) + "</svg>"

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
    share = top.sum() / total if total else 0
    pie_svg = (f'<svg viewBox="0 0 400 430" width="360">' + "".join(segs) +
               f'<text x="{cx}" y="420" text-anchor="middle" font-size="13" fill="#17202a">Top10 合计占全部 {share * 100:.1f}%</text></svg>')
    leg = []
    for i, (b, cnt) in enumerate(top.items()):
        leg.append(f'<div class="lg"><span style="background:{pal[i]}"></span>{label(b)}（{int(cnt)}，{cnt/total_cnt*100:.1f}%）</div>')
    if rest:
        leg.append(f'<div class="lg"><span style="background:#e2e8f0"></span>其余业务（{rest}，{rest/total_cnt*100:.1f}%）</div>')
    leg_html = "".join(leg)
    return bar_svg, pie_svg, leg_html


def build_html(fn, out_name, title, note, cur_series):
    d = pd.read_csv(OUT / fn, dtype={"node_id": str})
    t1 = d["top1_business"].astype(str).map(disp)
    pal = ["#2563eb", "#15803d", "#b45309", "#7c3aed", "#dc2626", "#0d9488", "#ca8a04", "#db2777", "#4f46e5", "#16a34a"]
    bar_svg, pie_svg, leg = pie_of(t1, "被推荐为 Top1 的节点数（前十）", pal, "")
    block = f"""<div class='card'>{bar_svg}</div>
<div class='card two'><div><h3>Top10 占全部推荐</h3>{pie_svg}</div><div><h3>图例</h3>{leg}</div></div>"""
    if cur_series is not None:
        cbar, cpie, cleg = pie_of(cur_series, "当前在跑业务（近7天有效结算）—— 前十", pal, "")
        block += (f"""<div class='card'>{cbar}</div>
<div class='card two'><div><h3>Top10 占全部当前在跑</h3>{cpie}</div><div><h3>图例</h3>{cleg}</div></div>""")
    html = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>" + title + "</title><style>"
            "body{font-family:Microsoft YaHei,sans-serif;margin:16px;background:#f6f7f9}.card{background:#fff;border-radius:10px;padding:16px 20px;margin:12px 0}"
            ".lg{display:flex;align-items:center;gap:8px;font-size:12px;margin:3px 0}.lg span{width:13px;height:13px;border-radius:3px;flex:none}.two{display:flex;gap:26px;flex-wrap:wrap}.two>div{min-width:300px;flex:1}.m{color:#647181}"
            "</style></head><body><h2>" + title + "</h2><p class='m'>" + note + "</p>"
            + block + "</body></html>")
    (OUT / out_name).write_text(html, encoding="utf-8")
    (PUB / out_name).write_text(html, encoding="utf-8")
    print("wrote", out_name)


def cur_series_for(src_html: Path, fn: str):
    if not src_html.exists():
        print("WARN 缺当前列源页", src_html, "→ 该页不生成对照饼")
        return None
    d = pd.read_csv(OUT / fn, dtype={"node_id": str})
    cmap = parse_current_ids(src_html)
    have = pd.Series([cmap.get(n) for n in d["node_id"]], dtype="object")
    have = have[have.notna()]
    n_cur = int(have.size)
    print(f"  当前在跑解析: n={n_cur} / {len(d)}")
    return have.astype(str)


cur_a = cur_series_for(PUB / "全节点_ant.html", "final_ant.csv")
cur_l = cur_series_for(PUB / "全节点_large_fullpool.html", "exp_final.csv")


def cur_note(prefix, s, total):
    if s is None:
        return f"{prefix}对照饼未生成（缺源页 全节点页）。"
    return f"{prefix}当前在跑样本 n={len(s)} / {total}（解析自已交付 全节点页 当前列；近7天无有效结算/空载节点不计入）。"


build_html("final_ant.csv", "推荐分布_小节点_ant.html",
           "V2 推荐分布_小节点（ant/盒子）",
           "口径：上图为 Top1 推荐分布，样本 = ant 池被推荐节点（final_ant，n=4,762，5% 数据）；ant 为弱产品单独页，不参与 large。"
           + cur_note("下图为当前在跑(近7天有效结算)分布，", cur_a, 4762)
           + "图表占比为各自样本口径，与 large 命中指标（时间外 n=2,994）不同，勿混读。",
           cur_a)
build_html("exp_final.csv", "推荐分布_大节点_large.html",
           "V2 推荐分布_大节点（全库 nonant）",
           "口径：上图为 Top1 推荐分布，样本 = 全库 nonant 全部被推荐节点（exp_final，n=14,966，画像交集，含训练/测试）；七牛家族 token 已映射为代表 ID。"
           + cur_note("下图为当前在跑(近7天有效结算)分布，", cur_l, 14966)
           + "家族命中指标另按时间外测试 n=2,994 计：hit1 21.0% / hit3 51.7%——分布占比(14,966)与命中指标(2,994)样本不同，勿混读。",
           cur_l)
