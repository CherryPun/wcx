# -*- coding: utf-8 -*-
"""重建 全节点_large_fullpool.html：加入“当前在跑(近7天实际)”列（近7天连续有效结算内收入最高 customer；
不满 7 天标近 N 天）。输入 lg_*.csv（已全拉 42 批）。"""
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
REP = {"FQCDNV": "10000292", "FQTZV": "10000281"}

raw = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(TMP / "lg_*.csv")))], ignore_index=True)
raw.columns = [c.lower() for c in raw.columns]
for c in ["cost", "rev", "profit"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["day"] = pd.to_datetime(raw["day"], errors="coerce")


def cur_of(g):
    g = g.dropna(subset=["day"])
    if len(g) == 0:
        return None
    dayset = set(g["day"])
    d0 = g["day"].max()
    n = 0
    cur = d0
    while cur in dayset and n < 7:
        n += 1
        cur -= pd.Timedelta(days=1)
    win = g[g["day"] >= d0 - pd.Timedelta(days=n - 1)]
    if win["rev"].fillna(0).sum() <= 0:
        return None
    best = win.loc[win["rev"].fillna(0).idxmax()]
    biz = str(int(best["customerid"]))
    bg = win[win["customerid"].astype(str) == biz]
    cost = float(bg["cost"].sum())
    pcol = float(bg["profit"].sum())
    prof = pcol if abs(pcol) > 1e-9 else float(bg["rev"].sum() - cost)
    return {"biz": biz, "name": (str(best.get("customerName", "")) or mp.get(biz, "")), "days": n,
            "cost": round(cost, 2), "profit": round(prof, 2)}


_rec = []
for _nid, _g in raw.groupby("nodeid"):
    _x = cur_of(_g)
    _rec.append({"node_id": str(_nid), "cur": _x or {"days": 0}})
cur = pd.DataFrame(_rec)

d = pd.read_csv(OUT / "exp_final.csv", dtype={"node_id": str})
for c in ["top1_business", "top2_business", "top3_business"]:
    d[c] = d[c].astype(str)
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
        b = REP.get(str(r[f"top{i}_business"]), str(r[f"top{i}_business"]))
        cells.append(f"<td>{mp.get(b, '')} {b}<br><span class='m'>结算 {r[f'top{i}_pred_cost']} · 利润 {r[f'top{i}_pred_profit']}</span></td>")
    rows.append("<tr>" + "".join(cells) + "</tr>")
html = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>全节点推荐_large（全库 nonant，家族粒 21.0/51.7, n=2994）</title><style>"
        "body{font-family:Microsoft YaHei,sans-serif;margin:14px}table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border:1px solid #dfe3e8;padding:5px 7px;text-align:left}.m{color:#647181}</style></head><body>"
        "<h2>全节点推荐_large（全库 nonant，clean 账）</h2>"
        "<p class='m'>家族粒 hit1 21.0 / hit3 51.7 / WAPE 0.41（n=2,994）。当前列=近7天连续有效结算内收入最高业务，不满7天标近N天；Top 吐代表具体 ID。候选/参考，不自动切。</p>"
        "<table><thead><tr><th>node_id</th><th>当前在跑(近7天实际)</th><th>Top1</th><th>Top2</th><th>Top3</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></body></html>")
(OUT / "全节点_large_fullpool.html").write_text(html, encoding="utf-8")
(PUB / "全节点_large_fullpool.html").write_text(html, encoding="utf-8")
filled = int(sum(1 for r in d.itertuples() if isinstance(getattr(r, "cur", None), dict) and r.cur.get("days", 0) > 0))
print("rows", len(d), "有当前在跑", filled, "wrote large fullpool html(带当前列)")
