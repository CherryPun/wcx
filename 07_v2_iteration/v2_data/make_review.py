# -*- coding: utf-8 -*-
"""V2 阶段E：人工复核底表。读 v2_data/_rerun 主版最终清单 + 当前列映射 + 业务名，输出可下钻复核 CSV。
不改主版文件，输出写本代 _rerun/。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "05_shared_data"
OUT = HERE / "_rerun"

rec = pd.read_csv(OUT / "final_top3_e2e.csv", dtype={"node_id": str})
cur = pd.read_csv(DATA / "cur7d_map.csv", dtype={"node_id": str})
o = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str},
                low_memory=False, encoding="utf-8")
o["business"] = o["business"].astype(str).str.replace(r"\.0$", "", regex=True)
name_map = o.dropna(subset=["business_name"]).drop_duplicates("business").set_index("business")["business_name"].to_dict()

cur = cur[["node_id", "has_cur", "cur_name", "cur_start", "cur_end", "cur_cost7", "cur_profit7"]].copy()
m = rec.merge(cur, on="node_id", how="left")
for i in (1, 2, 3):
    m[f"top{i}_name"] = m[f"top{i}_business"].astype(str).map(lambda b: name_map.get(str(b).replace(".0", ""), ""))
m["cur_top1_diff_cost"] = m["top1_pred_cost"] - m["cur_cost7"]
m["cur_top1_diff_profit"] = m["top1_pred_profit"] - m["cur_profit7"]

cols = ["node_id", "cur_name", "cur_start", "cur_end", "cur_cost7", "cur_profit7",
        "top1_name", "top1_business", "top1_pred_cost", "top1_pred_profit",
        "top2_name", "top2_business", "top2_pred_cost", "top2_pred_profit",
        "top3_name", "top3_business", "top3_pred_cost", "top3_pred_profit",
        "cur_top1_diff_cost", "cur_top1_diff_profit"]
m[cols].to_csv(OUT / "review_list.csv", index=False, encoding="utf-8-sig")
has = (m["has_cur"] == 1).sum()
print("节点", len(m), "当前有实际账", int(has),
      "其中当前利润为负", int(((m["has_cur"] == 1) & (m["cur_profit7"] < 0)).sum()),
      "当前与Top1预测利润方向相反", int(((m["has_cur"] == 1) & (m["cur_profit7"] < 0) & (m["top1_pred_profit"] > 0)).sum()))
print("wrote", OUT / "review_list.csv")
