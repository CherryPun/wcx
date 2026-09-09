# -*- coding: utf-8 -*-
"""终版页：用全库扩样 final 生成 large 页（家族 token→代表具体 ID），并出 large 复核表（真实最优/命中/家族一致性）。
ant 页保持 5% final_ant。只写 _rerun/。"""
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

d = pd.read_csv(OUT / "exp_final.csv", dtype={"node_id": str})
for c in ["top1_business", "top2_business", "top3_business"]:
    d[c] = d[c].astype(str).str.replace(r"\.0$", "", regex=True)
# 家族代表 ID：用 5% name map 估算（FQCDNV→10000292、FQTZV→10000281 作诊断代表）
REP = {"FQCDNV": "10000292", "FQTZV": "10000281"}
def disp(b):
    return REP.get(b, b)
for i in (1, 2, 3):
    d[f"top{i}_rep"] = d[f"top{i}_business"].map(disp)
    d[f"top{i}_label"] = d[f"top{i}_rep"].map(lambda b: f"{mp.get(b,'')} {b}")

def page(df, out_html, title, note):
    rows = []
    for _, r in df.iterrows():
        cells = [f"<td>{r.node_id}</td>"]
        for i in (1, 2, 3):
            cells.append(f"<td>{r[f'top{i}_label']}<br><span class='m'>结算 {r[f'top{i}_pred_cost']} · 利润 {r[f'top{i}_pred_profit']}</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    html = (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title><style>"
            "body{font-family:Microsoft YaHei,sans-serif;margin:14px}table{border-collapse:collapse;width:100%;font-size:12px}"
            "th,td{border:1px solid #dfe3e8;padding:5px 7px;text-align:left}.m{color:#647181}</style></head><body>"
            f"<h2>{title}</h2><p class='m'>{note}</p>"
            "<table><thead><tr><th>node_id</th><th>Top1</th><th>Top2</th><th>Top3</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></body></html>")
    (OUT / out_html).write_text(html, encoding="utf-8")
    print("wrote", out_html, "rows", len(df))

page(d, "全节点_large_fullpool.html",
     "全节点推荐_large（全库 nonant，家族粒 21.0/51.7, n=2994）",
     "全库非 ant 满7天账扩样；推荐吐代表具体 customerId（家族=FQCDNV/FQTZV 之代表 ID）；候选/参考，不承诺最赚。WAPE 0.41。")

# 复核表（真实最优/是否命中/家族一致）
eo = pd.read_csv(OUT / "exp_outcomes_clean.csv", dtype={"node_id": str, "business": str})
eo["business"] = eo["business"].astype(str).str.replace(r"\.0$", "", regex=True)
eo["online"] = pd.to_datetime(eo["online_day"], errors="coerce")
eo["cc"] = pd.to_numeric(eo["cum_cost_7d"], errors="coerce")
nf = eo[eo.node_id.isin(set(d.node_id))].groupby("node_id")["online"].min().reset_index().sort_values("online").reset_index(drop=True)
cut = int(len(nf) * 0.8)
test = set(nf.iloc[cut:]["node_id"])
tb = (eo[eo.node_id.isin(test)].sort_values("cc").groupby("node_id", as_index=False).tail(1)
      [["node_id", "business"]].rename(columns={"business": "true_best"}))
def fm(b): return {"10000292": "FQCDNV", "10000282": "FQCDNV", "10000281": "FQTZV", "10000279": "FQTZV"}.get(str(b), str(b))
rv = d.merge(tb, on="node_id", how="inner")
rv["tb_fam"] = rv.true_best.map(fm)
rv["top1_fam"] = rv.top1_business.map(fm)
rv["hit1"] = (rv.top1_fam == rv.tb_fam).astype(int)
rv["hit3"] = ((rv.top1_fam == rv.tb_fam) | (rv.top2_business.map(fm) == rv.tb_fam) | (rv.top3_business.map(fm) == rv.tb_fam)).astype(int)
out = rv[["node_id", "true_best", "tb_fam", "top1_rep", "top1_label", "hit1", "hit3"]].rename(
    columns={"top1_rep": "top1_business_id"})
out.to_csv(OUT / "large_review_fullpool.csv", index=False, encoding="utf-8-sig")
print("review rows", len(out), " 命中@1", round(out.hit1.mean()*100, 1), "@3", round(out.hit3.mean()*100, 1))
