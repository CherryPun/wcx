# -*- coding: utf-8 -*-
"""E2E v2（批量、只测测试节点）：改进版金额模型接入排序对比 v1。
成本：log1p+neighbor；利润：direct。候选/软约束沿用 v1。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
DATA = TASK / "05_shared_data"
OUT = HERE / "_rerun"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(TASK / "02_main_recommendation"))
sys.path.insert(0, str(HERE))

import stage_e2e as e2e  # noqa: E402
from stage_cost_improve import build_neighbor_feats  # noqa: E402

KNN_K = 15
KNN_FEAT_K = 10


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

    tr_nb = build_neighbor_feats(tr_x[["node_id", "business"]], tr_x, X, id_pos)
    te_nb = build_neighbor_feats(te_x[["node_id", "business"]], tr_x, X, id_pos)
    tr_x = pd.concat([tr_x, tr_nb], axis=1)
    te_x = pd.concat([te_x, te_nb], axis=1)

    cats_feat = e2e.CAT_COLS

    def prep(x_df):
        d = e2e.feature_df(x_df[["node_id", "business"]], attrs, stats, cats_feat)
        d = d.merge(x_df[["node_id", "business", "cum_cost_7d", "cum_revenue_7d", "cum_profit_7d"]],
                    on=["node_id", "business"], how="left")
        for c in ["neighbor_cost_k10", "neighbor_support_log1p"]:
            d[c] = x_df[c].values
        return d

    tr_feat = prep(tr_x)
    te_feat = prep(te_x)
    base_feats = cats_feat + e2e.NUM_COLS + ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log", "biz_prior"]
    feat_cost = base_feats + ["neighbor_cost_k10", "neighbor_support_log1p"]
    for c in cats_feat:
        cats = pd.Index(pd.concat([tr_feat[c], te_feat[c]]).unique())
        tr_feat[c] = pd.Categorical(tr_feat[c], categories=cats)
        te_feat[c] = pd.Categorical(te_feat[c], categories=cats)
    for df in (tr_feat, te_feat):
        df["neighbor_cost_k10"] = pd.to_numeric(df["neighbor_cost_k10"], errors="coerce").fillna(df["biz_prior"])
        df["neighbor_support_log1p"] = pd.to_numeric(df["neighbor_support_log1p"], errors="coerce").fillna(0.0)

    def make_model(feats, use_log, ycol):
        y_tr = tr_x[ycol].astype(float).to_numpy()
        y_te = te_x[ycol].astype(float).to_numpy()
        tgt_tr = np.log1p(y_tr) if use_log else y_tr
        tgt_te = np.log1p(y_te) if use_log else y_te
        mdl = XGBRegressor(n_estimators=700, learning_rate=0.04, max_depth=6, min_child_weight=5,
                           subsample=0.85, colsample_bytree=0.85, reg_lambda=10, tree_method="hist",
                           objective="reg:squarederror", random_state=e2e.RANDOM_STATE, n_jobs=-1,
                           early_stopping_rounds=40)
        mdl.fit(tr_feat[feats], tgt_tr, eval_set=[(te_feat[feats], tgt_te)], verbose=False)
        return mdl, use_log

    cost_mdl, cost_log = make_model(feat_cost, True, "cum_cost_7d")
    profit_mdl, _ = make_model(base_feats, False, "cum_profit_7d")
    print("训练完成")

    # biz 账本：相对 tr_pos 顺序（供 sims 切片）
    tr_pos_idx = np.array([id_pos[n] for n in sorted(train_ids) if n in id_pos], dtype=int)
    tr_pos_ids = np.array([str(attrs.iloc[i]["node_id"]) for i in tr_pos_idx])
    rel_pos = {gidx: r for r, gidx in enumerate(tr_pos_idx)}
    train_ledger = tr_x[["node_id", "business", "cum_cost_7d"]].copy()
    biz_ledger = {}
    for b, g in train_ledger.groupby("business"):
        gg = g[g.node_id.astype(str).isin(tr_pos_ids.tolist())].copy()
        gg["rel"] = [rel_pos[id_pos[n]] for n in gg.node_id.astype(str)]
        biz_ledger[str(b)] = gg[["rel", "cum_cost_7d"]].reset_index(drop=True)

    cap_df = pd.read_csv(DATA / "business_capacity_summary.csv", dtype={"business": str})
    cap_map = cap_df.set_index("business").to_dict("index")
    bw_daily = pd.read_csv(DATA / "bw_daily_7d.csv", dtype={"node_id": str, "business": str}, low_memory=False)
    bw_daily["peak95_mbps"] = pd.to_numeric(bw_daily["peak95"], errors="coerce") / 1e6
    bw_daily["business"] = bw_daily["business"].astype(str).str.replace(r"\.0$", "", regex=True)
    bw_map = {}
    for (nid, biz), g in bw_daily.groupby(["node_id", "business"]):
        bw_map[(str(nid), str(biz))] = float(g["peak95_mbps"].max())
    bw_nom = pd.to_numeric(attrs.set_index("node_id")["bw"], errors="coerce").to_dict()
    hist_map = frame.groupby("node_id")["business"].apply(lambda s: set(s.astype(str))).to_dict()

    def prep_predict(pairs, neighbor_cols):
        d = e2e.feature_df(pairs, attrs, stats, cats_feat)
        for c in cats_feat:
            d[c] = pd.Categorical(d[c], categories=tr_feat[c].cat.categories)
        for c in e2e.NUM_COLS + ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        for c in ["neighbor_cost_k10", "neighbor_support_log1p"]:
            d[c] = neighbor_cols[c].to_numpy()
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        return d

    def cap_penalty(b):
        info = cap_map.get(str(b))
        if not info:
            return 0.0
        over = float(info.get("over_capacity_mbps") or 0)
        cv = float(info.get("capacity_upper_mbps") or 0)
        return min(1.0, over / cv) if cv > 0 else 0.0

    eval_nodes = sorted(n for n in test_ids if n in hist_map)
    rows = []
    for nid in eval_nodes:
        if nid not in id_pos:
            continue
        sims_node = (X[id_pos[nid]] @ X[tr_pos_idx].T).toarray().ravel()
        order = np.argsort(-sims_node)[:KNN_K]
        neigh_rows = train_ledger[train_ledger.node_id.isin(tr_pos_ids[order].tolist())]
        cand = sorted(set(neigh_rows["business"].astype(str)) | (hist_map[nid] & pool_set))
        if not cand:
            continue
        # neighbor features
        nb = np.empty(len(cand))
        supp = np.empty(len(cand))
        for j, b in enumerate(cand):
            ld = biz_ledger.get(str(b))
            if ld is not None and len(ld):
                rr = ld["rel"].to_numpy(dtype=int)
                vv = ld["cum_cost_7d"].astype(float).to_numpy()
                ss = sims_node[rr]
                o = np.argsort(-ss)[:KNN_FEAT_K]
                w = np.maximum(ss[o], 0) ** 4
                nb[j] = float(np.average(vv[o], weights=w)) if w.sum() > 1e-12 else float(np.mean(vv[o]))
                supp[j] = float(np.log1p(len(ld)))
            else:
                nb[j] = 0.0
                supp[j] = 0.0
        pairs = pd.DataFrame([(nid, b) for b in cand], columns=["node_id", "business"])
        d = prep_predict(pairs, {"neighbor_cost_k10": pd.Series(nb), "neighbor_support_log1p": pd.Series(supp)})
        pc = np.expm1(cost_mdl.predict(d[feat_cost])) if cost_log else cost_mdl.predict(d[feat_cost])
        pp = profit_mdl.predict(d[base_feats])
        f = pd.DataFrame({"business": cand, "pred_cost": np.clip(pc, 0, None), "pred_profit": np.clip(pp, 0, None)})
        mn, mx = f["pred_cost"].min(), f["pred_cost"].max()
        f["cost_norm"] = (f["pred_cost"] - mn) / (mx - mn) if mx - mn > 1e-9 else 0.5
        f["profit_pen"] = np.where(f["pred_profit"] <= 0, -e2e.W_PROFIT * f["cost_norm"], 0.0)
        f["cap_pen"] = -e2e.W_CAP * f["business"].map(cap_penalty)
        peak = f["business"].map(lambda b: bw_map.get((str(nid), str(b)), np.nan))
        nominal = bw_nom.get(str(nid), np.nan)
        f["bw_aux"] = (peak / nominal).fillna(0.0).clip(0, 2) if nominal and nominal > 0 else 0.0
        f["score"] = f["pred_cost"] + f["profit_pen"] + f["cap_pen"] + e2e.W_BW * f["cost_norm"] * f["bw_aux"]
        top = f.sort_values("score", ascending=False).head(3)
        rec = {"node_id": str(nid)}
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            rec[f"top{i}_business"] = str(r["business"])
            rec[f"top{i}_pred_cost"] = round(float(r["pred_cost"]), 2)
            rec[f"top{i}_pred_profit"] = round(float(r["pred_profit"]), 2)
            rec[f"top{i}_profit_positive"] = int(float(r["pred_profit"]) > 0)
        rows.append(rec)
    final = pd.DataFrame(rows)
    final.to_csv(OUT / "final_top3_e2e_v2.csv", index=False, encoding="utf-8-sig")

    true_best = pos_te.sort_values("cum_cost_7d").groupby("node_id", as_index=False).tail(1).copy()
    true_best["business"] = true_best["business"].astype(str)
    true_best = true_best.set_index("node_id")

    def eval_hit(df):
        fi = df.set_index("node_id")
        h1 = h3 = n = 0
        for nid in fi.index:
            if nid not in true_best.index:
                continue
            tb = str(true_best.loc[nid, "business"]).replace(".0", "")
            t = [str(fi.loc[nid].get(f"top{i}_business", "")).replace(".0", "") for i in (1, 2, 3)]
            n += 1
            h1 += t[0] == tb
            h3 += tb in t
        pp = pd.to_numeric(df["top1_profit_positive"], errors="coerce").fillna(0)
        return {"n": int(n), "hit1": round(h1 / max(n, 1), 4), "hit3": round(h3 / max(n, 1), 4),
                "profit_positive": round(float(pp.mean()), 4)}

    cur = eval_hit(final)
    # 与 v1 对比（v1 全量文件取测试节点）
    v1_path = OUT / "final_top3_e2e.csv"
    prev = None
    if v1_path.exists():
        v1 = pd.read_csv(v1_path, dtype={"node_id": str})
        v1_test = v1[v1.node_id.isin(eval_nodes)].copy()
        prev = eval_hit(v1_test)
    res = {
        "v2_test_nodes": cur,
        "v1_test_nodes": prev,
        "avg_pred_cost_top1_v2": round(float(pd.to_numeric(final["top1_pred_cost"], errors="coerce").mean()), 2),
        "note": "v2=log1p+neighbor 成本模型；v1=direct。仅测试节点对比，同一候选与软约束。",
    }
    (OUT / "e2e_v2_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
