# -*- coding: utf-8 -*-
"""P0：时间外切分 + 平台利润资格门槛（自包含，可在交付目录直接运行）。
- 切分：按节点引入日（最早 business_online_day）前 80% 训练、后 20% 测试（较晚上线新节点）
- 训练 XGB：成本/收入/利润（内联实现，不依赖外部脚本）
- 资格门槛：业务训练集利润 p10 > 0 才允许当 Top1（平台不为负）
- 输出写本目录 _rerun/，数据读 ../05_shared_data
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
DATA = TASK / "05_shared_data"
OUT = HERE / "_rerun"
OUT.mkdir(parents=True, exist_ok=True)

MIN_SUPPORT = 30
TRAIN_RATIO = 0.8
W_PROFIT_NEG = 0.8
RANDOM_STATE = 42

CAT_COLS = ["province", "isp", "resourcetype", "deliverytype", "nattype", "dialtype", "device_type", "os"]
NUM_COLS = ["bw", "corenum", "memtotal", "totaldisksize", "hdddisksize", "ssddisksize", "systemdisksize"]
BIZ_STATS = ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log"]
TARGETS = {"cost": "cum_cost_7d", "revenue": "cum_revenue_7d", "profit": "cum_profit_7d"}


def build_biz_stats(pos: pd.DataFrame) -> pd.DataFrame:
    g = pos.groupby("business", observed=True)
    st = g.agg(
        cost_mean=("cum_cost_7d", "mean"),
        cost_p90=("cum_cost_7d", lambda s: s.quantile(0.9)),
        profit_mean=("cum_profit_7d", "mean"),
        support=("node_id", "nunique"),
    ).reset_index()
    st["cost_mean_log"] = np.log1p(st["cost_mean"])
    st["cost_p90_log"] = np.log1p(st["cost_p90"])
    st["profit_mean_log"] = np.log1p(np.clip(st["profit_mean"], 0, None))
    st["support_log"] = np.log1p(st["support"])
    return st


def wape(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0, None)
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-12))


def train_eval_metric(ycol, tr_df, te_df, feature_cols):
    y_tr = tr_df[ycol].astype(float).to_numpy()
    y_te = te_df[ycol].astype(float).to_numpy()
    mdl = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
                       subsample=0.85, colsample_bytree=0.85, reg_lambda=8, tree_method="hist",
                       enable_categorical=True, objective="reg:squarederror",
                       random_state=RANDOM_STATE, n_jobs=-1, early_stopping_rounds=30)
    try:
        XGBRegressor(n_estimators=1, tree_method="hist", device="cuda").fit([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0])
        mdl.set_params(device="cuda")
    except Exception:
        pass
    mdl.fit(tr_df[feature_cols], y_tr, eval_set=[(te_df[feature_cols], y_te)], verbose=False)
    pred = np.clip(mdl.predict(te_df[feature_cols]), 0, None)
    m = {
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "r2": float(r2_score(y_te, pred)),
        "wape": wape(y_te, pred),
    }
    return mdl, m, pred


def norm01(s):
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if mx - mn <= 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def main():
    attrs = pd.read_csv(DATA / "nodes_attr_filled.csv", dtype={"node_id": str}, low_memory=False).drop_duplicates("node_id")
    outcomes = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
    outcomes["business"] = outcomes["business"].str.replace(r"\.0$", "", regex=True)
    for c in ["cum_cost_7d", "cum_revenue_7d"]:
        outcomes[c] = pd.to_numeric(outcomes[c], errors="coerce")
    outcomes["cum_profit_7d"] = outcomes["cum_revenue_7d"] - outcomes["cum_cost_7d"]
    outcomes = outcomes[outcomes["outcome_days"] >= 7].copy()
    outcomes["online_day_dt"] = pd.to_datetime(outcomes["online_day"], errors="coerce")
    frame = outcomes.merge(attrs[["node_id"]], on="node_id", how="inner")

    node_first = frame.groupby("node_id")["online_day_dt"].min().reset_index().rename(columns={"online_day_dt": "intro_dt"})
    node_first = node_first.sort_values("intro_dt").reset_index(drop=True)
    cut = int(len(node_first) * TRAIN_RATIO)
    train_ids = set(node_first.iloc[:cut]["node_id"].astype(str))
    test_ids = set(node_first.iloc[cut:]["node_id"].astype(str))
    print("节点引入日范围", node_first.intro_dt.min(), "~", node_first.intro_dt.max())
    print("训练节点", len(train_ids), "测试节点", len(test_ids))

    pos_tr = frame[frame.node_id.isin(train_ids)].copy()
    pos_te = frame[frame.node_id.isin(test_ids)].copy()
    print("正样本 训练", len(pos_tr), "测试", len(pos_te))

    stats = build_biz_stats(pos_tr)
    pool = stats[stats["support"] >= MIN_SUPPORT].copy()
    pool_biz = sorted(pool["business"].astype(str))
    print("候选池", len(pool_biz))

    prof_stats = pos_tr.groupby("business", observed=True)["cum_profit_7d"].quantile(0.10).rename("profit_p10").reset_index()
    prof_stats["business"] = prof_stats["business"].astype(str)
    p10_map = dict(zip(prof_stats["business"], prof_stats["profit_p10"]))

    def feature_frame(pairs, attrs_df):
        df = pairs[["node_id", "business"]].merge(attrs_df, on="node_id", how="left")
        df = df.merge(stats, on="business", how="left")
        for c in BIZ_STATS:
            df[c] = df[c].fillna(0.0)
        gmean_cost = stats["cost_mean"].mean()
        prior = stats.set_index("business")["cost_mean"].to_dict()
        df["biz_prior"] = df.business.map(prior).fillna(gmean_cost)
        for c in CAT_COLS:
            df[c] = df[c].fillna("__UNK__").astype(str)
        for c in NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df

    tr_df = feature_frame(pos_tr, attrs).merge(
        pos_tr[["node_id", "business", "cum_cost_7d", "cum_revenue_7d", "cum_profit_7d"]],
        on=["node_id", "business"], how="left")
    te_df = feature_frame(pos_te, attrs).merge(
        pos_te[["node_id", "business", "cum_cost_7d", "cum_revenue_7d", "cum_profit_7d"]],
        on=["node_id", "business"], how="left")
    feature_cols = CAT_COLS + NUM_COLS + BIZ_STATS + ["biz_prior"]
    for c in CAT_COLS:
        cats = pd.Index(pd.concat([tr_df[c], te_df[c]]).unique())
        tr_df[c] = pd.Categorical(tr_df[c], categories=cats)
        te_df[c] = pd.Categorical(te_df[c], categories=cats)

    metrics = {}
    models = {}
    for short, ycol in TARGETS.items():
        model, m, _ = train_eval_metric(ycol, tr_df, te_df, feature_cols)
        metrics[short] = m
        models[short] = model
    print("精排指标", json.dumps(metrics, ensure_ascii=False))

    test_nodes = sorted(pos_te.node_id.unique())
    final_rows = []
    for nid in test_nodes:
        pairs = pd.DataFrame([(nid, b) for b in pool_biz], columns=["node_id", "business"])
        ff = feature_frame(pairs, attrs)
        for c in CAT_COLS:
            ff[c] = pd.Categorical(ff[c], categories=tr_df[c].cat.categories)
        pred = {short: np.clip(models[short].predict(ff[feature_cols]), 0, None) for short in models}
        ff["pred_cost"] = pred["cost"]
        ff["pred_revenue"] = pred["revenue"]
        ff["pred_profit"] = pred["profit"]
        ff["profit_p10"] = ff.business.map(lambda b: p10_map.get(b, np.nan))
        ff["cost_norm"] = norm01(ff["pred_cost"])
        ff["profit_pen"] = np.where(ff["pred_profit"] <= 0, -W_PROFIT_NEG * ff["cost_norm"], 0.0)
        ff["soft_score"] = ff["cost_norm"] + ff["profit_pen"]
        eligible = (ff["pred_profit"] > 0) & (ff["profit_p10"] > 0)
        cand1 = ff[eligible].copy() if eligible.any() else ff.copy()
        cand1 = cand1.sort_values(["soft_score", "pred_cost"], ascending=[False, False])
        ff = ff.sort_values(["soft_score", "pred_cost"], ascending=[False, False])
        top1 = cand1.iloc[[0]] if len(cand1) else ff.iloc[[0]]
        rest = ff[~ff.index.isin(top1.index)].head(2)
        chosen = pd.concat([top1, rest], ignore_index=True)
        rec = {"node_id": nid}
        for i, (_, r) in enumerate(chosen.iterrows(), start=1):
            b = str(r["business"])
            rec[f"top{i}_business"] = b
            rec[f"top{i}_name"] = ""
            rec[f"top{i}_pred_cost"] = round(float(r["pred_cost"]), 2)
            rec[f"top{i}_pred_revenue"] = round(float(r["pred_revenue"]), 2)
            rec[f"top{i}_pred_profit"] = round(float(r["pred_profit"]), 2)
            rec[f"top{i}_profit_positive"] = int(float(r["pred_profit"]) > 0)
        final_rows.append(rec)
    final = pd.DataFrame(final_rows)
    final.to_csv(OUT / "final_top3_temporal.csv", index=False, encoding="utf-8-sig")

    true_best = pos_te.sort_values("cum_cost_7d").groupby("node_id", as_index=False).tail(1).copy()
    true_best["business"] = true_best["business"].astype(str)
    true_best = true_best.set_index("node_id")
    final_i = final.set_index("node_id")
    hit1 = hit3 = n = profit_ok = 0
    for nid, row in final_i.iterrows():
        if nid not in true_best.index:
            continue
        tb = str(true_best.loc[nid, "business"]).replace(".0", "")
        t = [str(row.get(f"top{i}_business", "")).replace(".0", "") for i in (1, 2, 3)]
        if not any(t):
            continue
        n += 1
        hit1 += t[0] == tb
        hit3 += tb in t
        pv = row.get("top1_profit_positive")
        profit_ok += int(pd.notna(pv) and int(pv))
    out = {
        "split": "temporal: node intro_day 前80%训练/后20%测试",
        "train_nodes": int(len(train_ids)),
        "test_nodes": int(len(test_ids)),
        "n_test_eval": int(n),
        "regression_metrics": metrics,
        "real_best_hit_at_1": round(hit1 / max(n, 1), 4),
        "real_best_hit_at_3": round(hit3 / max(n, 1), 4),
        "top1_profit_positive_rate": round(profit_ok / max(n, 1), 4),
        "top1_business_distribution": final.top1_business.value_counts().head(12).to_dict(),
        "note": "P0：时间外验证 + 平台利润资格门槛（业务训练利润p10>0 且 预测利润>0 才可当Top1）。",
    }
    (OUT / "final_top3_temporal_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
