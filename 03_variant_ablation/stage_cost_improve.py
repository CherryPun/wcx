# -*- coding: utf-8 -*-
"""B：降低金额预测误差。
改进点：
1. 目标 log1p(cost) 回归 -> expm1 反变换（压长尾大额）
2. 特征加 neighbor 同业务金额统计（相似节点加权前K成本均值/支持度）
对比：原 XGB(直接回归) vs 改进(neighbor + log1p)
评估：时间外测试正样本的 cost WAPE/R²/MAE
输出：本目录 _rerun/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
DATA = TASK / "05_shared_data"
OUT = HERE / "_rerun"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(TASK / "02_main_recommendation"))
sys.path.insert(0, str(HERE))

import stage_e2e as e2e  # noqa: E402

KNN_FEAT_K = 10


def build_neighbor_feats(rows: pd.DataFrame, train_ledger: pd.DataFrame, X, id_pos, sim_pow=4.0):
    """为 rows(有 node_id,business) 计算同业务相似节点加权成本统计。"""
    # 候选节点集合：训练账中有该业务的行
    tr_biz_rows = {}
    for b, g in train_ledger.groupby("business"):
        tr_biz_rows[str(b)] = g[["node_id", "cum_cost_7d"]].copy()
    outs = []
    for _, r in rows.iterrows():
        nid = str(r["node_id"])
        biz = str(r["business"])
        feats = {"neighbor_cost_k10": np.nan, "neighbor_support_log1p": 0.0}
        cand = tr_biz_rows.get(biz)
        if cand is None or len(cand) == 0 or nid not in id_pos:
            outs.append(feats)
            continue
        cand = cand[cand["node_id"].astype(str) != nid].copy()
        if len(cand) == 0:
            outs.append(feats)
            continue
        pos = np.array([id_pos[c] for c in cand["node_id"].astype(str) if c in id_pos], dtype=int)
        if len(pos) == 0:
            outs.append(feats)
            continue
        sims = (X[id_pos[nid]] @ X[pos].T).toarray().ravel()
        values = cand.loc[[i for i in pos], "cum_cost_7d"].astype(float).to_numpy() if False else None
        # cand 与 pos 一一对应需确保顺序一致
        cand_idx = cand["node_id"].astype(str).isin(set(id_pos.keys()))
        cand = cand[cand_idx].copy()
        pos = np.array([id_pos[c] for c in cand["node_id"].astype(str)], dtype=int)
        sims = (X[id_pos[nid]] @ X[pos].T).toarray().ravel()
        values = cand["cum_cost_7d"].astype(float).to_numpy()
        order = np.argsort(-sims)[:KNN_FEAT_K]
        s = sims[order]
        v = values[order]
        w = np.maximum(s, 0.0) ** sim_pow
        if w.sum() > 1e-12:
            feats["neighbor_cost_k10"] = float(np.average(v, weights=w))
        else:
            feats["neighbor_cost_k10"] = float(np.mean(v))
        feats["neighbor_support_log1p"] = float(np.log1p(len(cand)))
        outs.append(feats)
    return pd.DataFrame(outs, index=rows.index)


def main():
    attrs, frame = e2e.load()
    node_first = frame.groupby("node_id")["online_day_dt"].min().reset_index().rename(columns={"online_day_dt": "intro_dt"})
    node_first = node_first.sort_values("intro_dt").reset_index(drop=True)
    cut = int(len(node_first) * e2e.TRAIN_RATIO)
    train_ids = set(node_first.iloc[:cut]["node_id"].astype(str))
    test_ids = set(node_first.iloc[cut:]["node_id"].astype(str))
    pos_tr = frame[frame.node_id.isin(train_ids)].copy()
    pos_te = frame[frame.node_id.isin(test_ids)].copy()

    stats = e2e.build_stats(pos_tr)
    pool = stats[stats["support"] >= e2e.MIN_SUPPORT]["business"].astype(str).tolist()
    pool_set = set(pool)
    tr_x = pos_tr[pos_tr["business"].isin(pool_set)].copy()
    te_x = pos_te[pos_te["business"].isin(pool_set)].copy()

    X, _, _ = e2e.node_matrix(attrs)
    id_pos = {n: i for i, n in enumerate(attrs["node_id"].astype(str))}

    # 只在参与建模的行上加 neighbor 特征
    tr_x = tr_x.copy()
    te_x = te_x.copy()
    print("计算 neighbor 特征 ...")
    tr_nb = build_neighbor_feats(tr_x[["node_id", "business"]], tr_x, X, id_pos)
    te_nb = build_neighbor_feats(te_x[["node_id", "business"]], tr_x, X, id_pos)
    tr_x = pd.concat([tr_x, tr_nb], axis=1)
    te_x = pd.concat([te_x, te_nb], axis=1)

    cats_feat = e2e.CAT_COLS
    def prep(x_df, fit_cats=None):
        d = e2e.feature_df(x_df[["node_id", "business"]], attrs, stats, cats_feat)
        d = d.merge(x_df[["node_id", "business", "cum_cost_7d", "cum_revenue_7d", "cum_profit_7d"]],
                    on=["node_id", "business"], how="left")
        for c in ["neighbor_cost_k10", "neighbor_support_log1p"]:
            d[c] = x_df[c].values
        if fit_cats is not None:
            for c in cats_feat:
                d[c] = pd.Categorical(d[c], categories=fit_cats[c])
        return d
    base_feats = cats_feat + e2e.NUM_COLS + ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log", "biz_prior"]
    new_feats = base_feats + ["neighbor_cost_k10", "neighbor_support_log1p"]

    tr_feat = prep(tr_x)
    te_feat = prep(te_x, fit_cats=None)
    cats_map = {}
    for c in cats_feat:
        cats = pd.Index(pd.concat([tr_feat[c], te_feat[c]]).unique())
        cats_map[c] = cats
        tr_feat[c] = pd.Categorical(tr_feat[c], categories=cats)
        te_feat[c] = pd.Categorical(te_feat[c], categories=cats)
    # 填充 neighbor 缺失
    for df in (tr_feat, te_feat):
        df["neighbor_cost_k10"] = pd.to_numeric(df["neighbor_cost_k10"], errors="coerce").fillna(df["biz_prior"])
        df["neighbor_support_log1p"] = pd.to_numeric(df["neighbor_support_log1p"], errors="coerce").fillna(0.0)

    def fit_eval(feats, use_log):
        y_tr = tr_x["cum_cost_7d"].astype(float).to_numpy()
        y_te = te_x["cum_cost_7d"].astype(float).to_numpy()
        tgt_tr = np.log1p(y_tr) if use_log else y_tr
        tgt_te = np.log1p(y_te) if use_log else y_te
        mdl = XGBRegressor(n_estimators=700, learning_rate=0.04, max_depth=6, min_child_weight=5,
                           subsample=0.85, colsample_bytree=0.85, reg_lambda=10, tree_method="hist",
                           objective="reg:squarederror", random_state=e2e.RANDOM_STATE,
                           n_jobs=-1, early_stopping_rounds=40)
        mdl.fit(tr_feat[feats], tgt_tr, eval_set=[(te_feat[feats], tgt_te)], verbose=False)
        pred = mdl.predict(te_feat[feats])
        if use_log:
            pred = np.clip(np.expm1(pred), 0, None)
        else:
            pred = np.clip(pred, 0, None)
        mae = float(mean_absolute_error(y_te, pred))
        wape = float(np.abs(y_te - pred).sum() / max(np.abs(y_te).sum(), 1e-12))
        r2 = float(r2_score(y_te, pred))
        return {"mae": mae, "wape": wape, "r2": r2}, pred

    base_metrics, _ = fit_eval(base_feats, use_log=False)
    new_metrics, _ = fit_eval(new_feats, use_log=True)
    mid_metrics, _ = fit_eval(new_feats, use_log=False)  # 中间版：neighbor、不 log1p

    res = {
        "n_train": int(len(tr_x)), "n_test": int(len(te_x)),
        "baseline_direct": base_metrics,
        "neighbor_only_no_log1p": mid_metrics,
        "log1p_plus_neighbor": new_metrics,
        "delta": {
            "direct_vs_neighbor_only_wape": round(base_metrics["wape"] - mid_metrics["wape"], 4),
            "neighbor_only_vs_log1p_wape": round(mid_metrics["wape"] - new_metrics["wape"], 4),
        },
        "note": "三变体同时间外测试：direct / neighbor(不log1p) / neighbor+log1p。拆解误差下降来源。",
    }
    (OUT / "cost_improve_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
