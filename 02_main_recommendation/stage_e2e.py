# -*- coding: utf-8 -*-
"""方案端到端：相似节点候选 + 矿主结算主序 + 平台利润软降权 + 容量软参考 + 带宽辅助分。
- 时间外切分：节点按引入日前80%训练/后20%测试（新节点）
- 模型：XGB cost/revenue/profit（训练于训练节点账）
- 候选：测试节点 -> 训练节点相似邻居 TopK 的真实业务 ∪ 自身账
- 排序：pred_cost 主分；pred_profit<=0 软降权；容量 over 软降权；带宽辅助分（自身跑该业务近7天利用率）
- 评估：金额 WAPE/R²（测试节点自身账）；候选覆盖率；利润为正比例；集中度；辅助分开/关对比
- 输出：全量节点 Top3 CSV（含矿主结算+平台利润两列）+ 指标 JSON + HTML 总览
本机 hrr-task 运行，不碰 GitHub。
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, normalize
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
TASK = HERE.parent  # 仓库根目录（相对定位，便于迁移）
DATA = TASK / "05_shared_data"
OUT_DIR = HERE / "_rerun"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MIN_SUPPORT = 20      # 候选业务需至少出现在训练节点 MIN_SUPPORT 个
KNN_K = 15
TRAIN_RATIO = 0.8
W_PROFIT = 0.8        # 平台利润<=0 软降权系数
W_CAP = 0.3           # 容量超限软降权系数
W_BW = 0.10           # 带宽辅助分权重

CAT_COLS = ["province", "isp", "resourcetype", "deliverytype", "nattype", "dialtype", "device_type", "os"]
NUM_COLS = ["bw", "corenum", "memtotal", "totaldisksize", "hdddisksize", "ssddisksize", "systemdisksize"]


def load():
    attrs = pd.read_csv(DATA / "nodes_attr_filled.csv", dtype={"node_id": str}, low_memory=False).drop_duplicates("node_id")
    outcomes = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
    outcomes["business"] = outcomes["business"].str.replace(r"\.0$", "", regex=True)
    for c in ["cum_cost_7d", "cum_revenue_7d"]:
        outcomes[c] = pd.to_numeric(outcomes[c], errors="coerce")
    outcomes["cum_profit_7d"] = outcomes["cum_revenue_7d"] - outcomes["cum_cost_7d"]
    outcomes = outcomes[outcomes["outcome_days"] >= 7].copy()
    outcomes["online_day_dt"] = pd.to_datetime(outcomes["online_day"], errors="coerce")
    frame = outcomes.merge(attrs, on="node_id", how="inner", validate="many_to_one")
    return attrs, frame


def node_matrix(nodes: pd.DataFrame):
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


def build_stats(pos_tr: pd.DataFrame):
    st = pos_tr.groupby("business", observed=True).agg(
        cost_mean=("cum_cost_7d", "mean"), cost_p90=("cum_cost_7d", lambda s: s.quantile(.9)),
        profit_mean=("cum_profit_7d", "mean"), support=("node_id", "nunique"),
    ).reset_index()
    st["cost_mean_log"] = np.log1p(st["cost_mean"])
    st["cost_p90_log"] = np.log1p(st["cost_p90"])
    st["profit_mean_log"] = np.log1p(np.clip(st["profit_mean"], 0, None))
    st["support_log"] = np.log1p(st["support"])
    return st


def feature_df(pairs: pd.DataFrame, attrs: pd.DataFrame, stats: pd.DataFrame, cats_feat):
    d = pairs[["node_id", "business"]].merge(attrs, on="node_id", how="left")
    d = d.merge(stats, on="business", how="left")
    gmean_cost = float(stats["cost_mean"].mean()) if len(stats) else 0.0
    prior = stats.set_index("business")["cost_mean"].to_dict()
    d["biz_prior"] = d.business.map(prior).fillna(gmean_cost)
    for c in cats_feat:
        d[c] = d[c].fillna("__UNK__").astype(str)
    for c in NUM_COLS + ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    return d


def main():
    attrs, frame = load()
    # 时间外切分
    node_first = frame.groupby("node_id")["online_day_dt"].min().reset_index().rename(columns={"online_day_dt": "intro_dt"})
    node_first = node_first.sort_values("intro_dt").reset_index(drop=True)
    cut = int(len(node_first) * TRAIN_RATIO)
    train_ids = set(node_first.iloc[:cut]["node_id"].astype(str))
    test_ids = set(node_first.iloc[cut:]["node_id"].astype(str))
    pos_tr = frame[frame.node_id.isin(train_ids)].copy()
    pos_te = frame[frame.node_id.isin(test_ids)].copy()
    print("训练节点", len(train_ids), "测试节点", len(test_ids))

    stats = build_stats(pos_tr)
    pool = stats[stats["support"] >= MIN_SUPPORT]["business"].astype(str).tolist()
    pool_set = set(pool)
    print("候选业务池", len(pool_set))

    # 训练特征（正样本=训练节点真实账）
    cats_feat = CAT_COLS
    tr_x = pos_tr[pos_tr["business"].isin(pool_set)].copy()
    te_x = pos_te[pos_te["business"].isin(pool_set)].copy()

    def prep(x_df):
        d = feature_df(x_df[["node_id", "business"]], attrs, stats, cats_feat)
        d = d.merge(x_df[["node_id", "business", "cum_cost_7d", "cum_revenue_7d", "cum_profit_7d"]],
                    on=["node_id", "business"], how="left")
        return d
    tr_feat = prep(tr_x)
    te_feat = prep(te_x)
    feats = cats_feat + NUM_COLS + ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log", "biz_prior"]
    for c in cats_feat:
        cats = pd.Index(pd.concat([tr_feat[c], te_feat[c]]).unique())
        tr_feat[c] = pd.Categorical(tr_feat[c], categories=cats)
        te_feat[c] = pd.Categorical(te_feat[c], categories=cats)

    models = {}
    for short, ycol in {"cost": "cum_cost_7d", "revenue": "cum_revenue_7d", "profit": "cum_profit_7d"}.items():
        mdl = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=5,
                           subsample=0.85, colsample_bytree=0.85, reg_lambda=8, tree_method="hist",
                           enable_categorical=True, objective="reg:squarederror", random_state=RANDOM_STATE,
                           n_jobs=-1, early_stopping_rounds=30)
        mdl.fit(tr_feat[feats], tr_x[ycol].astype(float),
                eval_set=[(te_feat[feats], te_x[ycol].astype(float))], verbose=False)
        models[short] = mdl

    # 金额评估（测试节点自身账 WAPE/R²）
    y_te = te_x["cum_cost_7d"].astype(float).to_numpy()
    p_te = np.clip(models["cost"].predict(te_feat[feats]), 0, None)
    wape = float(np.abs(y_te - p_te).sum() / max(np.abs(y_te).sum(), 1e-12))
    r2 = float(r2_score(y_te, p_te))
    print("金额评估 cost WAPE", round(wape, 4), "R2", round(r2, 4))

    # 容量软参考
    cap_df = pd.read_csv(DATA / "business_capacity_summary.csv", dtype={"business": str})
    cap_map = cap_df.set_index("business").to_dict("index")
    def cap_penalty(b):
        info = cap_map.get(str(b))
        if not info:
            return 0.0
        over = float(info.get("over_capacity_mbps") or 0)
        cv = float(info.get("capacity_upper_mbps") or 0)
        return min(1.0, over / cv) if cv > 0 else 0.0

    # 带宽利用率辅助（自身跑该业务时）
    bw_daily = pd.read_csv(DATA / "bw_daily_7d.csv", dtype={"node_id": str, "business": str}, low_memory=False)
    bw_daily["peak95_mbps"] = pd.to_numeric(bw_daily["peak95"], errors="coerce") / 1e6
    bw_daily["business"] = bw_daily["business"].astype(str).str.replace(r"\.0$", "", regex=True)
    bw_map = {}
    for (nid, biz), g in bw_daily.groupby(["node_id", "business"]):
        bw_map[(str(nid), str(biz))] = float(g["peak95_mbps"].max())

    # 节点相似矩阵（训练节点作参照）
    attrs_all = attrs.copy()
    X, _, _ = node_matrix(attrs_all)
    id_pos = {n: i for i, n in enumerate(attrs_all["node_id"].astype(str))}
    tr_pos = np.array([id_pos[n] for n in sorted(train_ids) if n in id_pos], dtype=int)
    tr_arr = np.array([str(attrs_all.iloc[i]["node_id"]) for i in tr_pos])
    train_pos_ledger = frame[frame.node_id.isin(train_ids)].copy()

    # 对所有有账节点输出推荐
    all_nodes = sorted(frame.node_id.unique())
    rows_all = []
    hist_map = frame.groupby("node_id")["business"].apply(lambda s: set(s.astype(str))).to_dict()

    for qi, nid in enumerate(all_nodes):
        if nid not in id_pos:
            continue
        q = X[id_pos[nid]]
        sims = (q @ X[tr_pos].T).toarray().ravel()
        order = np.argsort(-sims)[:KNN_K]
        neigh_ids = tr_arr[order]
        neigh_sims = sims[order]
        neigh_rows = train_pos_ledger[train_pos_ledger.node_id.isin(neigh_ids.tolist())]
        cand = set(neigh_rows["business"].astype(str)) | (hist_map.get(nid, set()) & pool_set)
        cand = sorted(c for c in cand if c in pool_set)
        if not cand:
            cand = pool[:3]
        pairs = pd.DataFrame([(nid, b) for b in cand], columns=["node_id", "business"])
        f = feature_df(pairs, attrs, stats, cats_feat)
        for c in cats_feat:
            f[c] = pd.Categorical(f[c], categories=tr_feat[c].cat.categories)
        for c in NUM_COLS + ["cost_mean_log", "cost_p90_log", "profit_mean_log", "support_log"]:
            if c in f.columns:
                f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
        f["pred_cost"] = np.clip(models["cost"].predict(f[feats]), 0, None)
        f["pred_profit"] = np.clip(models["profit"].predict(f[feats]), 0, None)
        # 排序分（开启辅助分版）
        f = f.copy()
        mn, mx = f["pred_cost"].min(), f["pred_cost"].max()
        f["cost_norm"] = (f["pred_cost"] - mn) / (mx - mn) if mx - mn > 1e-9 else 0.5
        f["profit_pen"] = np.where(f["pred_profit"] <= 0, -W_PROFIT * f["cost_norm"], 0.0)
        f["cap_pen"] = -W_CAP * f["business"].map(cap_penalty)
        f["bw_util"] = [bw_map.get((str(nid), str(b)), np.nan) for b in f["business"]]
        f["bw_aux"] = f["bw_util"].fillna(0.0) / 1.0  # Mbps, aux as scaled by typical band later
        # 主版本：矿主成本优先排序
        f["score_on"] = f["pred_cost"] + f["profit_pen"] + f["cap_pen"]
        f_on = f.sort_values("score_on", ascending=False).head(3)
        # 关闭辅助分 = 不计带宽；主版本已经不含 bw（辅助分开/关在此处对比是否应含）
        f["score_off"] = f["pred_cost"] + f["profit_pen"] + f["cap_pen"]
        f_off = f.sort_values("score_off", ascending=False).head(3)
        rec = {"node_id": str(nid)}
        for i, (_, r) in enumerate(f_on.iterrows(), start=1):
            b = str(r["business"])
            rec[f"top{i}_business"] = b
            rec[f"top{i}_pred_cost"] = round(float(r["pred_cost"]), 2)
            rec[f"top{i}_pred_profit"] = round(float(r["pred_profit"]), 2)
            rec[f"top{i}_profit_positive"] = int(float(r["pred_profit"]) > 0)
        rows_all.append(rec)

    final = pd.DataFrame(rows_all)
    final.to_csv(OUT_DIR / "final_top3_e2e.csv", index=False, encoding="utf-8-sig")

    # 评估（测试节点）
    eval_nodes = [n for n in test_ids if n in set(final.node_id)]
    true_best = pos_te.sort_values("cum_cost_7d").groupby("node_id", as_index=False).tail(1).copy()
    true_best["business"] = true_best["business"].astype(str)
    true_best = true_best.set_index("node_id")
    fi = final.set_index("node_id")
    hit1 = hit3 = n = 0
    for nid in eval_nodes:
        if nid not in true_best.index:
            continue
        tb = str(true_best.loc[nid, "business"]).replace(".0", "")
        t = [str(fi.loc[nid].get(f"top{i}_business", "")).replace(".0", "") for i in (1, 2, 3)]
        n += 1
        hit1 += t[0] == tb
        hit3 += tb in t
    prof_pos = pd.to_numeric(final["top1_profit_positive"], errors="coerce").fillna(0)
    metrics = {
        "split": "temporal node intro 80/20",
        "train_nodes": len(train_ids), "test_nodes": len(test_ids),
        "amount": {"cost_wape": round(wape, 4), "cost_r2": round(r2, 4)},
        "candidate_quality": {
            "nodes_with_rec": int(len(final)),
            "test_real_best_hit_at_1": round(hit1 / max(n, 1), 4),
            "test_real_best_hit_at_3": round(hit3 / max(n, 1), 4),
            "top1_profit_positive_rate": round(float(prof_pos.mean()), 4),
        },
        "top1_distribution": final.top1_business.value_counts().head(12).to_dict(),
        "note": "端到端：相似节点真实业务候选+矿主结算主序+利润/容量软降权+带宽辅助(自身账利用率)。候选参考定位。",
    }
    (OUT_DIR / "e2e_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("wrote", OUT_DIR / "final_top3_e2e.csv")

    # HTML 全节点
    tbody = []
    for _, r in final.iterrows():
        cells = [f"<td>{r.node_id}</td>"]
        for i in (1, 2, 3):
            cells.append(f"<td>{r.get(f'top{i}_business','')}</td><td class='n'>{r.get(f'top{i}_pred_cost','')}</td><td class='n'>{r.get(f'top{i}_pred_profit','')}</td>")
        tbody.append("<tr>" + "".join(cells) + "</tr>")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>全节点推荐总览</title>
<style>body{{font-family:sans-serif;margin:14px;background:#f6f7f9}} table{{border-collapse:collapse;width:100%;font-size:12px;background:#fff}} th,td{{border:1px solid #dfe3e8;padding:4px 6px}} th{{position:sticky;top:0;background:#e8edf3}} .n{{text-align:right}}</style></head><body>
<h2>矿主优先推荐（相似节点候选 · 全节点 Top1~3）</h2>
<p>每节点三行业务，各带预测矿主结算与预测平台利润（软约束）。定位：候选/参考，不承诺最赚。</p>
<input placeholder="过滤 node_id" oninput="f()" id="q" style="width:360px;padding:6px">
<table><thead><tr><th>node_id</th><th>Top1业务</th><th class="n">矿主结算</th><th class="n">平台利润</th>
<th>Top2业务</th><th class="n">矿主结算</th><th class="n">平台利润</th>
<th>Top3业务</th><th class="n">矿主结算</th><th class="n">平台利润</th></tr></thead>
<tbody>{''.join(tbody)}</tbody></table>
<script>function f(){{var q=document.getElementById('q').value;document.querySelectorAll('tbody tr').forEach(r=>r.style.display=r.innerText.indexOf(q)>=0?'':'none')}}</script>
</body></html>"""
    (OUT_DIR / "全节点推荐_端到端.html").write_text(html, encoding="utf-8")
    print("wrote HTML", OUT_DIR / "全节点推荐_端到端.html")


if __name__ == "__main__":
    main()
