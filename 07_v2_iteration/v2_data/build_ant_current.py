# -*- coding: utf-8 -*-
"""给 ant 池重建 全节点_ant.html：加入“当前在跑（近7天实际）”列——近7天连续有效结算日内的业务
（取窗口内收入最高 customer），给该业务结算/利润与天数；不满 7 天标“近 N 天”。"""
from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "_rerun"
PUB = HERE.parent / "outputs"
TMP = Path(r"C:\Users\大禧子\AppData\Local\Temp\opencode")
o = pd.read_csv(ROOT / "05_shared_data" / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
o["business"] = o["business"].astype(str).str.replace(r"\.0$", "", regex=True)
mp = o.dropna(subset=["business_name"]).drop_duplicates("business").set_index("business")["business_name"].to_dict()

raw = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(TMP / "cw_*.csv")))], ignore_index=True)
raw.columns = [c.lower() for c in raw.columns]
raw["cost"] = pd.to_numeric(raw["cost"], errors="coerce")
raw["rev"] = pd.to_numeric(raw["rev"], errors="coerce")
raw["profit"] = pd.to_numeric(raw["profit"], errors="coerce")
raw["day"] = pd.to_datetime(raw["day"], errors="coerce")
print("rows", len(raw), "nodes", raw.nodeid.nunique())

def cur_of(g):
    g = g.sort_values("day")
    g = g[pd.notna(g["day"])]
    if len(g) == 0:
        return None
    # 连续有效自然日（含任一行）自最近日起
    dayset = set(g["day"])
    d0 = g["day"].max()
    n = 0
    cur = d0
    while cur in dayset:
        n += 1
        cur -= pd.Timedelta(days=1)
        if n >= 7:
            break
    win = g[g["day"] >= d0 - pd.Timedelta(days=n - 1)]
    # 当前业务=窗口内收入最高 customer
    if win["rev"].fillna(0).sum() <= 0:
        return None
    best = win.loc[win["rev"].fillna(0).idxmax()]
    biz = str(int(best["customerid"]))
    bg = win[win["customerid"].astype(str) == str(best["customerid"])]
    cost = float(bg["cost"].sum())
    pcol = float(bg["profit"].sum())
    prof = pcol if abs(pcol) > 1e-9 else float(bg["rev"].sum() - cost)
    return {"biz": biz, "name": (str(best.get("customerName", "")) or mp.get(biz, "")), "days": n,
            "cost": round(cost, 2), "profit": round(prof, 2)}

_rec = []
for _nid, _g in raw.groupby("nodeid"):
    _x = cur_of(_g)
    _rec.append({"node_id": str(_nid), "cur": _x or {"days": 0, "biz": "", "name": "", "cost": 0, "profit": 0}})
cur = pd.DataFrame(_rec)

d = pd.read_csv(OUT / "final_ant.csv", dtype={"node_id": str})
d = d.merge(cur, on="node_id", how="left")
rows = []
for _, r in d.iterrows():
    c = r.get("cur")
    if isinstance(c, dict) and c.get("days", 0) > 0:
        dlab = f"近{c['days']}天" if c["days"] < 7 else "近7天"
        curcell = f"<td>{c['name']} {c['biz']}<br><span class='m'>{dlab}结算 {c['cost']} · 平台利润 {c['profit']}</span></td>"
    else:
        curcell = "<td><span class='m'>—</span></td>"
    cells = [f"<td>{r.node_id}</td>", curcell]
    for i in (1, 2, 3):
        b = str(r[f"top{i}_business"]).replace(".0", "")
        cells.append(f"<td>{mp.get(b, '')} {b}<br><span class='m'>结算 {r[f'top{i}_pred_cost']} · 利润 {r[f'top{i}_pred_profit']}</span></td>")
    rows.append("<tr>" + "".join(cells) + "</tr>")
html = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>全节点推荐_ant（盒子，弱产品）</title><style>"
        "body{font-family:Microsoft YaHei,sans-serif;margin:14px}table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border:1px solid #dfe3e8;padding:5px 7px;text-align:left}.m{color:#647181}</style></head><body>"
        "<h2>全节点推荐_ant（盒子/小盒子，弱产品单独页）</h2>"
        "<p class='m'>ant 独立命中约 8%，勿与 large 混排。当前列口径：近7天连续有效结算内收入最高业务；不满7天标注近 N 天。</p>"
        "<table><thead><tr><th>node_id</th><th>当前在跑(近7天实际)</th><th>Top1</th><th>Top2</th><th>Top3</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></body></html>")
(OUT / "全节点_ant.html").write_text(html, encoding="utf-8")
(PUB / "全节点_ant.html").write_text(html, encoding="utf-8")
filled = int(sum(1 for r in d.itertuples() if (r.cur or {}).get("days", 0) > 0))
print("rows", len(d), "有当前在跑", filled, "wrote 全节点_ant.html")
