# -*- coding: utf-8 -*-
"""冷启动诊断：时间外测试集（新节点）上比较基线。
基线：
1. global   —— 全局热度（业务在训练集出现节点数）
2. biz_prior—— 业务训练平均结算
3. segment  —— (省|ISP|资源类型) 内业务结算均值，样本少回退业务先验
4. knn      —— 节点特征余弦近邻 Top10 中该业务结算的相似加权均值
5. xgb      —— 重新在时间外训练上的 XGB 成本精排
评估：每测试节点对候选池排序，看“真实最优业务（该节点跑过且满7天成本最大）”是否进 Top1/Top3。
只输出指标，不改模型产物。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler, normalize
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
DATA = TASK / "05_shared_data"
TDIR = HERE / "_rerun"
TDIR.mkdir(parents=True, exist_ok=True)

MIN_SUPPORT = 30
TRAIN_RATIO = 0.8
KNN_K = 10
RANDOM_STATE = 42

CAT_COLS = ["province", "isp", "resourcetype", "deliverytype", "nattype", "dialtype", "device_type", "os"]
NUM_COLS = ["bw", "corenum", "memtotal", "totaldisksize", "hdddisksize", "ssddisksize", "systemdisksize"]


def load_temporal():
    attrs = pd.read_csv(DATA / "nodes_attr_filled.csv", dtype={"node_id": str}, low_memory=False).drop_duplicates("node_id")
    outcomes = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
    outcomes["business"] = outcomes["business"].str.replace(r"\.0$", "", regex=True)
    for c in ["cum_cost_7d", "cum_revenue_7d"]:
        outcomes[c] = pd.to_numeric(outcomes[c], errors="coerce")
    outcomes = outcomes[outcomes["outcome_days"] >= 7].copy()
    outcomes["online_day_dt"] = pd.to_datetime(outcomes["online_day"], errors="coerce")
    frame = outcomes.merge(attrs, on="node_id", how="inner", validate="many_to_one")
    node_first = frame.groupby("node_id")["online_day_dt"].min().reset_index().rename(columns={"online_day_dt": "intro_dt"})
    node_first = node_first.sort_values("intro_dt").reset_index(drop=True)
    cut = int(len(node_first) * TRAIN_RATIO)
    train_ids = set(node_first.iloc[:cut]["node_id"].astype(str))
    test_ids = set(node_first.iloc[cut:]["node_id"].astype(str))
    return attrs, frame, train_ids, test_ids


def node_matrix(nodes: pd.DataFrame, fit_cats=None, fit_num=None):
    cats = [c for c in CAT_COLS if c in nodes.columns]
    nums = [c for c in NUM_COLS if c in nodes.columns]
    x_cat = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True, dtype=np.float32)
    Xc = normalize(x_cat.fit_transform(nodes[cats].fillna("__UNK__").astype(str)), norm="l2")
    num = nodes[nums].apply(pd.to_numeric, errors="coerce").mask(lambda x: x <= 0)
    num_imp = SimpleImputer(strategy="median").fit_transform(num)
    q = QuantileTransformer(n_quantiles=min(500, len(nodes)), output_distribution="uniform", random_state=RANDOM_STATE)
    Xn = normalize(sparse.csr_matrix(q.fit_transform(num_imp).astype(np.float32)), norm="l2")
    X = sparse.hstack([np.sqrt(.65) * Xc, np.sqrt(.35) * Xn], format="csr")
    return normalize(X, norm="l2"), x_cat, q


def main():
    attrs, frame, train_ids, test_ids = load_temporal()
    pos_tr = frame[frame.node_id.isin(train_ids)].copy()
    pos_te = frame[frame.node_id.isin(test_ids)].copy()

    # 候选池
    support = pos_tr.groupby("business")["node_id"].nunique()
    pool = support[support >= MIN_SUPPORT].index.astype(str).tolist()
    true_best = pos_te.loc[pos_te.groupby("node_id")["cum_cost_7d"].idxmax()]
    true_best["business"] = true_best["business"].astype(str)
    true_best.index = true_best["node_id"].astype(str)
    eval_nodes = [str(n) for n, r in true_best.iterrows() if str(r["business"]) in pool]
    print("测试节点", len(true_best), "真实最优在候选池的评估节点", len(eval_nodes))

    # 训练侧聚合
    cost_tr = pos_tr[pos_tr["business"].isin(pool)].copy()
    biz_prior = cost_tr.groupby("business", observed=True)["cum_cost_7d"].mean().to_dict()
    gmean = float(cost_tr["cum_cost_7d"].mean())
    biz_heat = support.to_dict()

    # segment key：省|ISP|资源类型
    def seg_key(df):
        return (df["province"].fillna("__UNK__").astype(str) + "|"
                + df["isp"].fillna("__UNK__").astype(str) + "|"
                + df["resourcetype"].fillna("__UNK__").astype(str))
    cost_tr = cost_tr.copy()
    cost_tr["seg"] = seg_key(cost_tr)
    seg_cnt = cost_tr.groupby(["seg", "business"], observed=True).size()
    seg_sum = cost_tr.groupby(["seg", "business"], observed=True)["cum_cost_7d"].sum()
    seg_cnt_d = seg_cnt.to_dict()
    seg_mean = seg_sum.to_dict()

    # 业务统计特征（XGB）
    stats = cost_tr.groupby("business", observed=True)["cum_cost_7d"].agg(["mean", lambda s: s.quantile(0.9)]).rename(
        columns={"mean": "cost_mean", "<lambda_0>": "cost_p90"})
    stats["cost_mean_log"] = np.log1p(stats["cost_mean"])
    stats["cost_p90_log"] = np.log1p(stats["cost_p90"])

    # 节点相似度（训练节点作参照）
    tr_nodes = attrs[attrs.node_id.isin(train_ids)].copy()
    all_nodes = pd.concat([tr_nodes, attrs[attrs.node_id.isin(test_ids)].copy()], ignore_index=True)
    X, _, _ = node_matrix(all_nodes)
    id_pos = {n: i for i, n in enumerate(all_nodes["node_id"].astype(str))}
    tr_pos_idx = np.array([id_pos[n] for n in tr_nodes["node_id"].astype(str)])
    tr_ids_arr = tr_nodes["node_id"].astype(str).to_numpy()

    def run_baseline(name, scorer):
        hit1 = hit3 = 0
        for nid in eval_nodes:
            true_b = str(true_best.loc[nid, "business"])
            scores = scorer(nid, pool)
            order = sorted(pool, key=lambda b: scores.get(b, -1e18), reverse=True)[:3]
            hit1 += int(order[0] == true_b)
            hit3 += int(true_b in order)
        return {"hit@1": round(hit1 / len(eval_nodes), 4), "hit@3": round(hit3 / len(eval_nodes), 4)}

    # 1) global（热度）
    g = run_baseline("global", lambda nid, pool: {b: biz_heat.get(b, 0) for b in pool})
    # 2) biz prior
    b = run_baseline("biz_prior", lambda nid, pool: {b: biz_prior.get(b, gmean) for b in pool})
    # 3) segment
    def seg_scorer(nid, pool):
        seg = seg_key(attrs[attrs.node_id == nid])
        skey = seg.iloc[0] if len(seg) else "__UNK__"
        out = {}
        for biz in pool:
            v = seg_mean.get((skey, biz))
            if v is not None and seg_cnt_d.get((skey, biz), 0) >= 3:
                out[biz] = float(v)
            else:
                out[biz] = biz_prior.get(biz, gmean)
        return out
    sgm = run_baseline("segment", seg_scorer)
    # 4) knn
    def knn_scorer(nid, pool):
        qi = id_pos[nid]
        sims = (X[qi] @ X[tr_pos_idx].T).toarray().ravel()
        top = np.argsort(-sims)[:KNN_K]
        neigh_ids = tr_ids_arr[top]
        neigh_sims = sims[top]
        train_rows = cost_tr[cost_tr.node_id.isin(neigh_ids.tolist())]
        out = {}
        for biz in pool:
            sub = train_rows[train_rows["business"] == biz]
            if len(sub) == 0:
                out[biz] = biz_prior.get(biz, gmean)
                continue
            w = np.array([neigh_sims[np.where(neigh_ids == x)[0][0]] for x in sub.node_id], dtype=float)
            w = np.maximum(w, 0)
            out[biz] = float(np.average(sub["cum_cost_7d"].astype(float), weights=w)) if w.sum() > 0 else biz_prior.get(biz, gmean)
        return out
    kn = run_baseline("knn", knn_scorer)

    # 5) XGB（重训成本）
    cats_feat = [c for c in CAT_COLS if c in pos_tr.columns]
    tr_x = pos_tr[pos_tr["business"].isin(pool)].copy()
    te_x = pos_te[pos_te["business"].isin(pool)].copy()
    def feat(df):
        d = df[["node_id", "business"]].copy()
        d = d.merge(attrs, on="node_id", how="left")
        d = d.merge(stats.reset_index(), on="business", how="left")
        for c in cats_feat:
            d[c] = d[c].fillna("__UNK__").astype(str)
        for c in NUM_COLS + ["cost_mean_log", "cost_p90_log"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        return d
    tr_feat = feat(tr_x)
    te_feat = feat(te_x)
    feats = cats_feat + NUM_COLS + ["cost_mean_log", "cost_p90_log"]
    for c in cats_feat:
        cats = pd.Index(pd.concat([tr_feat[c], te_feat[c]]).unique())
        tr_feat[c] = pd.Categorical(tr_feat[c], categories=cats)
        te_feat[c] = pd.Categorical(te_feat[c], categories=cats)
    mdl = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=5,
                       subsample=0.85, colsample_bytree=0.85, reg_lambda=8, tree_method="hist",
                       enable_categorical=True, objective="reg:squarederror", random_state=RANDOM_STATE,
                       n_jobs=-1, early_stopping_rounds=30)
    mdl.fit(tr_feat[feats], tr_x["cum_cost_7d"].astype(float), eval_set=[(te_feat[feats], te_x["cum_cost_7d"].astype(float))], verbose=False)

    def xgb_scorer(nid, pool):
        pairs = pd.DataFrame([(nid, b) for b in pool], columns=["node_id", "business"])
        f = feat(pairs)
        for c in cats_feat:
            f[c] = pd.Categorical(f[c], categories=tr_feat[c].cat.categories)
        for c in NUM_COLS + ["cost_mean_log", "cost_p90_log"]:
            if c in f.columns:
                f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
        p = np.clip(mdl.predict(f[feats]), 0, None)
        return {b: float(x) for b, x in zip(pool, p)}
    xg = run_baseline("xgb", xgb_scorer)

    res = {"eval_nodes": len(eval_nodes), "global": g, "biz_prior": b, "segment": sgm, "knn": kn, "xgb": xg,
           "note": "时间外冷启动：真实最优业务是否进入候选池排序的 Top1/Top3"}
    (TDIR / "coldstart_baseline_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
