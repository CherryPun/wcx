# -*- coding: utf-8 -*-
"""P1 字节系容量处置清单（新）：对全切溢出的字节系业务出拦截/扩容建议。"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
V6 = HERE / "recent_month_large_mainstream_v6_capacity_full（新）"
TARGETS = ["1382680008", "10000224", "10000096", "10000041"]

a = pd.read_csv(V6 / "v6_capacity_allocation_summary.csv", low_memory=False)
a["business"] = a["business"].astype(str)
biz = a[a["pool_level"].eq("business")].copy()
rows = []
o = io.StringIO()
o.write("== P1 字节系容量处置清单 ==\n")
o.write("（全切 Top1 压力；扩容需求 = 全切负载 - 容量上限）\n\n")
for b in TARGETS:
    t = biz[biz["business"].eq(b)]
    if t.empty:
        continue
    r = t.iloc[0]
    need = max(float(r["unconstrained_top1_load_mbps"]) - float(r["raw_capacity_ceiling_mbps"]), 0.0)
    rows.append({
        "business": b, "business_name": r["business_name"], "capacity_state": r["capacity_state"],
        "current_bw_mbps": round(float(r["current_active_build_bandwidth_mbps"]), 1),
        "ceiling_mbps": round(float(r["raw_capacity_ceiling_mbps"]), 1),
        "unconstrained_top1_load_mbps": round(float(r["unconstrained_top1_load_mbps"]), 1),
        "unconstrained_overflow_mbps": round(float(r["unconstrained_top1_overflow_mbps"]), 1),
        "expand_need_gbps": round(need / 1000, 1),
        "planned_switch_in": r.get("planned_switch_in_nodes"),
    })
    o.write("%s %s: state=%s 当前=%.0f 上限=%.0f 全切负载=%.0f 溢出=%.0f；扩容需求≈%.0f Gbps；规划切入=%s\n" % (
        b, r["business_name"], r["capacity_state"], r["current_active_build_bandwidth_mbps"],
        r["raw_capacity_ceiling_mbps"], r["unconstrained_top1_load_mbps"],
        r["unconstrained_top1_overflow_mbps"], need / 1000, r.get("planned_switch_in_nodes")))
    # 最紧池（同业务，非 business 层，按 planned_headroom 升序取 5）
    pools = a[(a["business"].eq(b)) & (~a["pool_level"].eq("business"))].copy()
    pools["planned_headroom_mbps"] = pd.to_numeric(pools["planned_headroom_mbps"], errors="coerce")
    tight = pools.sort_values("planned_headroom_mbps").head(5)
    o.write("   最紧池（planned 余量最小 Top5）:\n")
    for _, q in tight.iterrows():
        o.write("     %s %s: 状态=%s 上限=%.0f 全切负载=%.0f 余量=%.0f\n" % (
            q["pool_level"], q.get("pool_key"), q["capacity_state"],
            q["raw_capacity_ceiling_mbps"], q["unconstrained_top1_load_mbps"],
            q["planned_headroom_mbps"]))
    o.write("\n")

pd.DataFrame(rows).to_csv(HERE / "P1字节系容量清单（新）.csv", index=False, encoding="utf-8-sig")

bl = pd.read_csv(V6 / "v6_capacity_blocked_nodes.csv", dtype=str)
rv = pd.read_csv(V6 / "v6_capacity_review_nodes.csv", dtype=str)
o.write("== 拦截/复核节点（现状） ==\n")
o.write("拦截节点 n=%d；按模型Top1: %s\n" % (len(bl), bl.get("model_top1_business", pd.Series(dtype=str)).value_counts().head(5).to_dict()))
o.write("复核切换节点 n=%d；按规划业务: %s\n" % (len(rv), rv.get("planned_business", pd.Series(dtype=str)).value_counts().head(5).to_dict()))
txt = o.getvalue()
(HERE / "P1字节系容量清单（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
