# -*- coding: utf-8 -*-
"""带宽辅助分 开/关 对比（A/B）。
带辅助分排序：score = pred_cost + profit_pen + cap_pen + W_BW * bw_aux
关辅助分排序：score = pred_cost + profit_pen + cap_pen
bw_aux：该节点跑该候选业务时的近7天利用率（peak95/名义带宽），仅自身账存在该业务时非空，
        否则 0。利用率先除一个典型参照值(1000 Mbps)防过大。
评估（测试节点）：Top1平均预测结算、真实最优命中Top1/Top3、利润为正率、两组变化节点数。
本机运行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
DATA = TASK / "05_shared_data"
sys.path.insert(0, str(TASK / "02_main_recommendation"))
sys.path.insert(0, str(HERE))

import stage_e2e as e2e  # noqa: E402

OUT = HERE / "_rerun"
OUT.mkdir(parents=True, exist_ok=True)
BW_REF = 1000.0


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

    cats_feat = e2e.CAT_COLS
    tr_x = pos_tr[pos_tr["business"].isin(pool_set)].copy()
    te_x = pos_te[pos_te["business"].isin(pool_set)].copy()

    def prep(x_df):
        d = e2e.feature_df(x_df[["node_id", "business"]], attrs, stats, cats_feat)
        d = d.merge(x_df[["node_id", "business", "cum_cost_7d", "cum_revenue_7d", "cum_profit_7d"]],
                    on=["node_id", "business"], how="left")
        return d
    tr_feat = prep(tr_x)
    te_feat = prep(te_x)
    feats = cats_feat + e2e.NUM_COLS + ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log", "biz_prior"]
    for c in cats_feat:
        cats = pd.Index(pd.concat([tr_feat[c], te_feat[c]]).unique())
        tr_feat[c] = pd.Categorical(tr_feat[c], categories=cats)
        te_feat[c] = pd.Categorical(te_feat[c], categories=cats)

    models = {}
    for short, ycol in {"cost": "cum_cost_7d", "profit": "cum_profit_7d"}.items():
        mdl = e2e.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=5,
                               subsample=0.85, colsample_bytree=0.85, reg_lambda=8, tree_method="hist",
                               enable_categorical=True, objective="reg:squarederror",
                               random_state=e2e.RANDOM_STATE, n_jobs=-1, early_stopping_rounds=30)
        mdl.fit(tr_feat[feats], tr_x[ycol].astype(float),
                eval_set=[(te_feat[feats], te_x[ycol].astype(float))], verbose=False)
        models[short] = mdl

    cap_df = pd.read_csv(DATA / "business_capacity_summary.csv", dtype={"business": str})
    cap_map = cap_df.set_index("business").to_dict("index")

    def cap_penalty(b):
        info = cap_map.get(str(b))
        if not info:
            return 0.0
        over = float(info.get("over_capacity_mbps") or 0)
        cv = float(info.get("capacity_upper_mbps") or 0)
        return min(1.0, over / cv) if cv > 0 else 0.0

    bw_daily = pd.read_csv(DATA / "bw_daily_7d.csv", dtype={"node_id": str, "business": str}, low_memory=False)
    bw_daily["peak95_mbps"] = pd.to_numeric(bw_daily["peak95"], errors="coerce") / 1e6
    bw_daily["business"] = bw_daily["business"].astype(str).str.replace(r"\.0$", "", regex=True)
    bw_map = {}
    for (nid, biz), g in bw_daily.groupby(["node_id", "business"]):
        bw_map[(str(nid), str(biz))] = float(g["peak95_mbps"].max())
    # 名义带宽用于利用率
    bw_nominal = pd.to_numeric(attrs.set_index("node_id")["bw"], errors="coerce").to_dict()

    X, _, _ = e2e.node_matrix(attrs)
    id_pos = {n: i for i, n in enumerate(attrs["node_id"].astype(str))}
    tr_pos = np.array([id_pos[n] for n in sorted(train_ids) if n in id_pos], dtype=int)
    tr_arr = np.array([str(attrs.iloc[i]["node_id"]) for i in tr_pos])
    train_pos_ledger = frame[frame.node_id.isin(train_ids)].copy()
    hist_map = frame.groupby("node_id")["business"].apply(lambda s: set(s.astype(str))).to_dict()

    true_best = pos_te.sort_values("cum_cost_7d").groupby("node_id", as_index=False).tail(1).copy()
    true_best["business"] = true_best["business"].astype(str)
    true_best = true_best.set_index("node_id")

    rows = []
    for nid in sorted(test_ids):
        if nid not in id_pos or nid not in hist_map:
            continue
        q = X[id_pos[nid]]
        sims = (q @ X[tr_pos].T).toarray().ravel()
        order = np.argsort(-sims)[:e2e.KNN_K]
        neigh_rows = train_pos_ledger[train_pos_ledger.node_id.isin(tr_arr[order].tolist())]
        cand = set(neigh_rows["business"].astype(str)) | (hist_map[nid] & pool_set)
        cand = sorted(c for c in cand if c in pool_set)
        if not cand:
            continue
        pairs = pd.DataFrame([(nid, b) for b in cand], columns=["node_id", "business"])
        f = e2e.feature_df(pairs, attrs, stats, cats_feat)
        for c in cats_feat:
            f[c] = pd.Categorical(f[c], categories=tr_feat[c].cat.categories)
        for c in e2e.NUM_COLS + ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log"]:
            if c in f.columns:
                f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
        f["pred_cost"] = np.clip(models["cost"].predict(f[feats]), 0, None)
        f["pred_profit"] = np.clip(models["profit"].predict(f[feats]), 0, None)
        mn, mx = f["pred_cost"].min(), f["pred_cost"].max()
        f["cost_norm"] = (f["pred_cost"] - mn) / (mx - mn) if mx - mn > 1e-9 else 0.5
        f["profit_pen"] = np.where(f["pred_profit"] <= 0, -e2e.W_PROFIT * f["cost_norm"], 0.0)
        f["cap_pen"] = -e2e.W_CAP * f["business"].map(cap_penalty)
        # 利用率辅助：自身账跑过该业务才非空；无自身账=0（不增不减）
        peak = f["business"].map(lambda b: bw_map.get((str(nid), str(b)), np.nan))
        nominal = pd.to_numeric(pd.Series([bw_nominal.get(str(nid), np.nan)] * len(f)), errors="coerce")
        util = peak / nominal.where(nominal > 0, np.nan)
        f["bw_aux"] = util.fillna(0.0).clip(0, 2)  # 上限2倍防异常
        f["score_off"] = f["pred_cost"] + f["profit_pen"] + f["cap_pen"]
        f["score_on"] = f["score_off"] + e2e.W_BW * f["bw_aux"] * f["cost_norm"]
        top_on = f.sort_values("score_on", ascending=False).head(3)
        top_off = f.sort_values("score_off", ascending=False).head(3)
        rows.append({
            "node_id": str(nid),
            "on1": str(top_on.iloc[0]["business"]),
            "on1_cost": float(top_on.iloc[0]["pred_cost"]),
            "on1_profit": float(top_on.iloc[0]["pred_profit"]),
            "on3": "|".join(str(b) for b in top_on["business"]),
            "off1": str(top_off.iloc[0]["business"]),
            "off1_cost": float(top_off.iloc[0]["pred_cost"]),
            "off1_profit": float(top_off.iloc[0]["pred_profit"]),
            "off3": "|".join(str(b) for b in top_off["business"]),
            "changed_top1": int(str(top_on.iloc[0]["business"]) != str(top_off.iloc[0]["business"])),
            "bw_nonzero_count": int((f["bw_aux"] > 0).sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bw_aux_ab_results.csv", index=False, encoding="utf-8-sig")

    n = len(df)
    hit_on1 = sum(str(df.loc[i, "on1"]) == str(true_best.loc[df.loc[i, "node_id"], "business"]) for i in range(n))
    hit_off1 = sum(str(df.loc[i, "off1"]) == str(true_best.loc[df.loc[i, "node_id"], "business"]) for i in range(n))

    def hit3_series(col):
        h = 0
        for i in range(n):
            tb = str(true_best.loc[df.loc[i, "node_id"], "business"])
            h += int(tb in str(df.loc[i, col]).split("|"))
        return h

    res = {
        "n_nodes": int(n),
        "avg_pred_cost_top1": {"on": round(float(df["on1_cost"].mean()), 2), "off": round(float(df["off1_cost"].mean()), 2)},
        "avg_pred_profit_top1": {"on": round(float(df["on1_profit"].mean()), 2), "off": round(float(df["off1_profit"].mean()), 2)},
        "hit_at_1": {"on": round(hit_on1 / max(n, 1), 4), "off": round(hit_off1 / max(n, 1), 4)},
        "hit_at_3": {"on": round(hit3_series("on3") / max(n, 1), 4), "off": round(hit3_series("off3") / max(n, 1), 4)},
        "changed_top1_count": int(df["changed_top1"].sum()),
        "bw_aux_positive_node_count": int((df["bw_nonzero_count"] > 0).sum()),
        "note": "带宽辅助分=自身跑该业务近7天利用率；开=排序加入 W_BW*bw_aux*cost_norm，关=不含。",
    }
    (OUT / "bw_aux_ab_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
