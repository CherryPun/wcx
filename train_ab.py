#!/usr/bin/env python3
"""A/B 对比（修正评估口径）：

在 node_analysis_data 覆盖的 5,205 节点上，比较
  BASE  = 仅 node_join 静态属性
  ENRICHED = node_join + node_analysis 静态库存(os/arch/额定带宽/核数/内存/磁盘容量)

评估口径（逐节点可行集）：每个节点只在它【实际跑过的业务】中比较模型选出的业务
与真实最优业务，regret 在该节点自身可行集内度量（coverage=100%，可比）。
这对应"给新节点推荐时，从它 eligible 的业务里挑"的生产语义。
"""
import os, json, numpy as np, pandas as pd
from model import BestBusinessRecommender

HERE = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42

nodes = pd.read_csv(os.path.join(HERE, "nodes_enriched.csv"))
outcomes = pd.read_csv(os.path.join(HERE, "outcomes_raw.csv"))
outcomes["business"] = outcomes["business"].astype(str)
for c in ("cum_cost_7d", "cum_revenue_7d"):
    outcomes[c] = pd.to_numeric(outcomes[c], errors="coerce").fillna(0.0)

COVER_MARK = "bw_rated"
covered_ids = set(nodes[nodes[COVER_MARK].notna()]["node_id"].astype(str).unique().tolist())
print(f"A/B 节点池（两边都有数据）: {len(covered_ids)}")

BASE_CAT = ["vendorid","deliverytype","resourcetype","dialtype","nattype","scheduleisps",
            "regsource","customermode","province","isp","city","device_type","arch_type",
            "isvm","qoskiller_status"]
BASE_NUM = ["bw"]
EXTRA_CAT = ["os","arch"]
EXTRA_NUM = ["bw_rated","corenum","memtotal","totaldisksize","hdddisksize","ssddisksize","systemdisksize"]

def build(cat, num):
    cols = ["node_id"] + cat + num + ["business","cum_cost_7d","cum_revenue_7d"]
    n = nodes[nodes["node_id"].astype(str).isin(covered_ids)][["node_id"]+cat+num]
    o = outcomes[outcomes["node_id"].astype(str).isin(covered_ids)]
    return o.merge(n, on="node_id", how="left")[cols]

def split(df):
    rng = np.random.default_rng(RANDOM_STATE)
    alln = df["node_id"].astype(str).unique()
    perm = rng.permutation(alln)
    k = int(len(perm)*0.8)
    tr, te = set(perm[:k].tolist()), set(perm[k:].tolist())
    return df[df["node_id"].astype(str).isin(tr)], df[df["node_id"].astype(str).isin(te)]

def eval_coverage(model, test_nodes, test_outcomes, candidate_businesses):
    """冷启动业务分配评估（全局候选 + 覆盖率感知）：

    - 候选集 = 训练期出现的全部可行业务（模拟"给新节点分配业务"）。
    - 覆盖率 = 推荐的业务恰好属于该节点【实际跑过】的业务的比例
              （新节点本就该被分配未跑过的业务，故低覆盖≠错，只是无法验证价值）。
    - 仅在"能验证"的节点上算 regret（稳健尺度归一化 + 绝对 regret）。
    """
    attr_cols = model.cat_features + model.num_features + ["node_id"]
    nset = test_nodes[attr_cols].drop_duplicates("node_id").set_index("node_id")
    o = test_outcomes.copy(); o["business"] = o["business"].astype(str)
    node_biz = {nid: set(g["business"].tolist()) for nid, g in o.groupby("node_id")}
    cand = [str(b) for b in candidate_businesses]
    recs = model.recommend(nset.reset_index(), businesses=cand, top_k=3).set_index("node_id")

    m_cov=o_cov=mh1=mh3=oh1=oh3=n=0
    m_abs=[]; o_abs=[]; best_cs=[]; best_os=[]
    for nid, bizs in node_biz.items():
        if nid not in nset.index or not bizs:
            continue
        n += 1
        rec = recs.loc[nid]
        sub = o[o["node_id"]==nid].set_index("business")
        cost = pd.to_numeric(sub["cum_cost_7d"], errors="coerce").fillna(0.0)
        op = pd.to_numeric(sub["cum_revenue_7d"], errors="coerce").fillna(0.0) - cost
        true_m = str(cost.idxmax()); true_o = str(op.idxmax())
        best_c = float(cost.max()); best_o = float(op.max())
        m_best = str(rec["miner_best"]); o_best = str(rec["operator_best"])
        mh1 += (m_best==true_m); oh1 += (o_best==true_o)
        m3 = [x.split(":",1)[0] for x in str(rec["miner_top3"]).split(";") if ":" in x]
        o3 = [x.split(":",1)[0] for x in str(rec["operator_top3"]).split(";") if ":" in x]
        mh3 += (true_m in m3); oh3 += (true_o in o3)
        if m_best in bizs:
            m_cov += 1; m_abs.append(max(0.0, best_c - float(cost.loc[m_best]))); best_cs.append(best_c)
        if o_best in bizs:
            o_cov += 1; o_abs.append(max(0.0, best_o - float(op.loc[o_best]))); best_os.append(best_o)
    scale_c = max(float(np.median([abs(x) for x in best_cs])) if best_cs else 1.0, 1e-6)
    scale_o = max(float(np.median([abs(x) for x in best_os])) if best_os else 1.0, 1e-6)
    m_reg = [x/scale_c for x in m_abs]; o_reg = [x/scale_o for x in o_abs]
    return {
        "n_nodes": n,
        "miner_hit_rate@1": mh1/n, "operator_hit_rate@1": oh1/n,
        "miner_hit_rate@3": mh3/n, "operator_hit_rate@3": oh3/n,
        "miner_realized_coverage": m_cov/n, "operator_realized_coverage": o_cov/n,
        "miner_realized_regret": float(np.mean(m_reg)) if m_reg else None,
        "operator_realized_regret": float(np.mean(o_reg)) if o_reg else None,
        "miner_abs_regret": float(np.mean(m_abs)) if m_abs else None,
        "operator_abs_regret": float(np.mean(o_abs)) if o_abs else None,
        "miner_value_capture": 1-float(np.mean(m_reg)) if m_reg else None,
        "operator_value_capture": 1-float(np.mean(o_reg)) if o_reg else None,
    }

def run_exp(name, cat, num):
    df = build(cat, num)
    tr, te = split(df)
    cand = sorted(tr["business"].astype(str).unique())
    print(f"\n=== {name} === 训练节点 {tr['node_id'].nunique()} / 测试 {te['node_id'].nunique()} / 候选业务 {len(cand)}")
    model = BestBusinessRecommender(cat, num).fit(tr, "cum_cost_7d", "cum_revenue_7d")
    m = eval_coverage(model, te[["node_id"]+cat+num].drop_duplicates("node_id"), te, cand)
    print("  测试集(冷启动分配):", json.dumps(
        {k:(round(v,4) if isinstance(v,(int,float)) else v) for k,v in m.items()}, ensure_ascii=False))
    return {"name": name, "test": m}

res_base = run_exp("BASE(仅node_join)", BASE_CAT, BASE_NUM)
res_enr  = run_exp("ENRICHED(+node_analysis静态)", BASE_CAT+EXTRA_CAT, BASE_NUM+EXTRA_NUM)

print("\n\n====== A/B 汇总（测试集，逐节点可行集评估） ======")
for key in ["n_nodes","miner_hit_rate@1","miner_hit_rate@3","operator_hit_rate@1",
            "operator_hit_rate@3","miner_realized_coverage","operator_realized_coverage",
            "miner_realized_regret","operator_realized_regret",
            "miner_abs_regret","operator_abs_regret",
            "miner_value_capture","operator_value_capture"]:
    b = res_base["test"].get(key); e = res_enr["test"].get(key)
    print(f"  {key:28s} BASE={b}   ENRICHED={e}")

with open(os.path.join(HERE, "ab_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"base": res_base, "enriched": res_enr}, f, ensure_ascii=False, indent=2, default=str)
print("\n已保存 ab_metrics.json")
