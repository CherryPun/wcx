# -*- coding: utf-8 -*-
"""V2 一代（自包含快照）：相似节点候选 + 矿主结算主序 + 平台利润软降权 + 容量软参考 + 带宽辅助分。
- 时间外切分：节点按引入日前80%训练/后20%测试（新节点）
- 模型：XGB cost/revenue/profit（训练于训练节点账）
- 候选：测试节点 -> 训练节点相似邻居 TopK 的真实业务 ∪ 自身账
- 排序：pred_cost 主分；pred_profit<=0 软降权；容量 over 软降权；带宽辅助分（自身跑该业务近7天利用率）
- 评估：金额 WAPE/R²（测试节点自身账）；候选覆盖率；利润为正比例；集中度；辅助分开/关对比
- 输出：全量节点 Top3 CSV（含矿主结算+平台利润两列）+ 指标 JSON，均写本代 _rerun/
- 数据：统一读仓库根 05_shared_data（不复制大文件）；本代独立可复现
"""
from __future__ import annotations

import json
import math
import os
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
ROOT = HERE.parents[1]            # 仓库根 wcx（本文件位于 07_v2_iteration/v2_data/）
DATA = ROOT / "05_shared_data"    # 共享数据统一入口
OUT_DIR = HERE / "_rerun"         # 本代产物
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MIN_SUPPORT = int(os.getenv("V2_MIN_SUPPORT", 20))   # 候选业务需至少出现在训练节点 MIN_SUPPORT 个
KNN_K = int(os.getenv("V2_KNN_K", 15))
TRAIN_RATIO = 0.8
V2_CUT = os.getenv("V2_CUT", "")      # 非空则按 online_day < V2_CUT 分训练/测试（滚动窗口）
W_PROFIT = float(os.getenv("V2_W_PROFIT", 0.8))   # 平台利润<=0 软降权系数
W_CAP = 0.3           # 容量超限软降权系数
W_BW = 0.10           # 带宽辅助分权重
SKIP_OUT = os.getenv("V2_SKIP_OUT", "0") == "1"   # 跳过最终 csv/html 写入（扫描用）
V2_ORIG = os.getenv("V2_ORIG", "0") == "1"          # 诊断用：恢复原主版排序（te早停/含cap/无bw）

CAT_COLS = ["province", "isp", "resourcetype", "deliverytype", "nattype", "dialtype", "device_type", "os"]
NUM_COLS = ["bw", "corenum", "memtotal", "totaldisksize", "hdddisksize", "ssddisksize", "systemdisksize"]


def family_of(business: str, name_map: dict) -> str:
    """七牛虚拟业务按名称诊断归并到家族 token（非生产绑定）。"""
    nm = name_map.get(str(business), "")
    if "七牛CDN-虚拟" in nm:
        return "FQCDNV"
    if "七牛特招-虚拟" in nm:
        return "FQTZV"
    return str(business)


def gate_bad_from(o2: pd.DataFrame, thr: float = 0.9) -> set:
    """门禁：ant 前缀份额 >= thr 的盒子业务不进 nonant 候选（纯函数，入参已按 outcome_days>=7 过滤）。"""
    o2 = o2.copy()
    o2["_ant"] = o2["node_id"].astype(str).str.startswith("ant")
    sh = o2.groupby("business").agg(ant_share=("_ant", "mean")).reset_index()
    return set(sh.loc[sh["ant_share"] >= thr, "business"].astype(str))


def temporal_split(node_first: pd.DataFrame, ratio: float = 0.8, cut_dt=None):
    """时间外切分：节点按最早上线日排序，前 ratio 训练 / 后测试；cut_dt 非空则按 <cut 划分。"""
    node_first = node_first.sort_values("intro_dt").reset_index(drop=True)
    if cut_dt is not None:
        train_ids = set(node_first[node_first["intro_dt"] < cut_dt]["node_id"].astype(str))
        test_ids = set(node_first[node_first["intro_dt"] >= cut_dt]["node_id"].astype(str))
    else:
        cut = int(len(node_first) * ratio)
        train_ids = set(node_first.iloc[:cut]["node_id"].astype(str))
        test_ids = set(node_first.iloc[cut:]["node_id"].astype(str))
    return train_ids, test_ids


def load():
    attrs_path = os.getenv("V2_ATTRS", str(DATA / "nodes_attr_filled.csv"))
    attrs = pd.read_csv(attrs_path, dtype={"node_id": str}, low_memory=False).drop_duplicates("node_id")
    out_path = os.getenv("V2_OUTCOMES", str(DATA / "outcomes_7d_named.csv"))
    outcomes = pd.read_csv(out_path, dtype={"node_id": str, "business": str}, low_memory=False)
    outcomes["business"] = outcomes["business"].str.replace(r"\.0$", "", regex=True)
    for c in ["cum_cost_7d", "cum_revenue_7d"]:
        outcomes[c] = pd.to_numeric(outcomes[c], errors="coerce")
    outcomes["cum_profit_7d"] = outcomes["cum_revenue_7d"] - outcomes["cum_cost_7d"]
    outcomes = outcomes[outcomes["outcome_days"] >= 7].copy()
    outcomes["online_day_dt"] = pd.to_datetime(outcomes["online_day"], errors="coerce")
    frame = outcomes.merge(attrs, on="node_id", how="inner", validate="many_to_one")
    pool = os.getenv("V2_POOL", "")
    if pool in ("ant", "nonant"):
        is_ant = attrs["node_id"].astype(str).str.startswith("ant")
        keep = is_ant if pool == "ant" else ~is_ant
        attrs = attrs[keep].copy()
        frame = frame[frame.node_id.isin(set(attrs["node_id"]))].copy()
    elif os.getenv("V2_NONANT", "0") == "1":   # 向后兼容别名
        attrs = attrs[~attrs["node_id"].astype(str).str.startswith("ant")].copy()
        frame = frame[~frame["node_id"].astype(str).str.startswith("ant")].copy()
    _merge_cfg = os.getenv("V2_MERGE", "")
    if _merge_cfg == "" and os.getenv("V2_POOL", "") == "nonant":
        _merge_cfg = "qiniu_name"   # large 默认开（诊断映射）；显式设 V2_MERGE=0 可关
    if _merge_cfg == "qiniu_name":   # 诊断：按名称合成七牛家族并聚合标签（不当生产绑定）
        _nm = frame.dropna(subset=["business_name"]).drop_duplicates("business")[["business", "business_name"]]
        _nm = {str(b): str(n) for b, n in zip(_nm["business"], _nm["business_name"])}
        frame = frame.copy()
        frame["business"] = frame["business"].astype(str).map(lambda b: family_of(b, _nm))
        frame["online_day_dt"] = pd.to_datetime(frame["online_day"], errors="coerce")
        frame = (frame.groupby(["node_id", "business"], as_index=False)
                 .agg(cum_cost_7d=("cum_cost_7d", "sum"),
                      cum_revenue_7d=("cum_revenue_7d", "sum"),
                      outcome_days=("outcome_days", "max"),
                      online_day=("online_day", "min"),
                      business_name=("business_name", "first")))
        frame["cum_profit_7d"] = frame["cum_revenue_7d"] - frame["cum_cost_7d"]
        frame["online_day_dt"] = pd.to_datetime(frame["online_day"], errors="coerce")
        print("V2_MERGE=qiniu_name 已合成家族并聚合（仅诊断）")
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
    train_ids, test_ids = temporal_split(node_first, ratio=TRAIN_RATIO,
                                         cut_dt=pd.Timestamp(V2_CUT) if V2_CUT else None)
    pos_tr = frame[frame.node_id.isin(train_ids)].copy()
    pos_te = frame[frame.node_id.isin(test_ids)].copy()
    print("训练节点", len(train_ids), "测试节点", len(test_ids))

    stats = build_stats(pos_tr)
    # 无业务例外：门槛统一 MIN_SUPPORT（含 10000074，全库 support 达标即自动进池）
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

    # 早停验证集：训练内随机 valid（不再用测试集，避免时间外偏乐观）
    _n = min(len(tr_feat), len(tr_x))
    _idx = np.arange(_n)
    _rng = np.random.RandomState(RANDOM_STATE)
    _rng.shuffle(_idx)
    _nv = int(len(_idx) * 0.15)
    _vi, _ti = _idx[:_nv], _idx[_nv:]
    models = {}
    for short, ycol in {"cost": "cum_cost_7d", "revenue": "cum_revenue_7d", "profit": "cum_profit_7d"}.items():
        mdl = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=5,
                           subsample=0.85, colsample_bytree=0.85, reg_lambda=8, tree_method="hist",
                           enable_categorical=True, objective="reg:squarederror", random_state=RANDOM_STATE,
                           n_jobs=-1, early_stopping_rounds=30)
        _ev = ((te_feat[feats], te_x[ycol].astype(float)) if V2_ORIG
               else (tr_feat.iloc[_vi][feats], tr_x.iloc[_vi][ycol].astype(float)))
        mdl.fit(tr_feat.iloc[_ti][feats], tr_x.iloc[_ti][ycol].astype(float),
                eval_set=[_ev], verbose=False)
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
    prof = []
    cand_rows = []
    hist_map = frame.groupby("node_id")["business"].apply(lambda s: set(s.astype(str))).to_dict()

    # A 门禁：几乎只在 ant 高频的盒子业务 ID 不进 nonant 候选
    gate_bad = set()
    gate_on = os.getenv("V2_GATE", "1" if os.getenv("V2_POOL", "") == "nonant" else "0") == "1"
    if gate_on and os.getenv("V2_POOL", "") == "nonant":
        _o2 = pd.read_csv(DATA / "outcomes_7d_named.csv", dtype={"node_id": str, "business": str}, low_memory=False)
        _o2["business"] = _o2["business"].astype(str).str.replace(r"\.0$", "", regex=True)
        _o2 = _o2[_o2["outcome_days"] >= 7]
        gate_bad = gate_bad_from(_o2, 0.9)   # ant_share 份额取自 5% 混合账（近似，见 runbook）
        pd.DataFrame({"business": sorted(gate_bad)}).to_csv(OUT_DIR / "gate_bad.csv", index=False, encoding="utf-8-sig")
        print("gate_bad 业务数", len(gate_bad))

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
        if gate_bad:
            cand = [c for c in cand if c not in gate_bad]
        fb = int(len(cand) == 0)
        if not cand:
            cand = ([b for b in pool if b not in gate_bad][:3] if gate_bad else pool[:3])
        n_nbiz = int(neigh_rows["business"].nunique()) if len(neigh_rows) else 0
        prof.append({"node_id": str(nid), "is_test": str(nid) in test_ids,
                     "n_cand": len(cand), "fallback": fb, "n_neigh_biz": n_nbiz})
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
        if V2_ORIG:  # 诊断格：恢复原主版逻辑（含失真容量降权、无带宽辅助、on=off）
            f["cap_pen"] = -W_CAP * f["business"].map(cap_penalty)
            f["score_on"] = f["pred_cost"] + f["profit_pen"] + f["cap_pen"]
            f_on = f.sort_values("score_on", ascending=False).head(3)
            f["score_off"] = f["score_on"]
        else:       # P1 修复版：去失真容量、带宽辅助真正进 score_on
            f["bw_util"] = [bw_map.get((str(nid), str(b)), np.nan) for b in f["business"]]
            _arr = pd.to_numeric(f["bw_util"], errors="coerce").to_numpy(dtype=float)
            _arr = _arr[~np.isnan(_arr)]
            bw_mx = float(_arr.max()) if _arr.size else 0.0
            f["bw_norm"] = pd.to_numeric(f["bw_util"], errors="coerce").fillna(0.0) / (bw_mx if bw_mx > 0 else 1.0)
            f["score_on"] = f["pred_cost"] + f["profit_pen"] + W_BW * f["bw_norm"] * f["cost_norm"]
            f_on = f.sort_values("score_on", ascending=False).head(3)
            f["score_off"] = f["pred_cost"] + f["profit_pen"]
        f_off = f.sort_values("score_off", ascending=False).head(3)
        rec = {"node_id": str(nid)}
        for i, (_, r) in enumerate(f_on.iterrows(), start=1):
            b = str(r["business"])
            rec[f"top{i}_business"] = b
            rec[f"top{i}_pred_cost"] = round(float(r["pred_cost"]), 2)
            rec[f"top{i}_pred_profit"] = round(float(r["pred_profit"]), 2)
            rec[f"top{i}_profit_positive"] = int(float(r["pred_profit"]) > 0)
        rows_all.append(rec)
        cand_rows.append({"node_id": str(nid), "cand": "|".join(cand),
                          "top1": str(rec.get("top1_business", "")), "top2": str(rec.get("top2_business", "")), "top3": str(rec.get("top3_business", ""))})

    if os.getenv("V2_POOL", "") in ("ant", "nonant"):
        _pf = pd.DataFrame(prof)
        _pf.to_csv(OUT_DIR / ("profile_" + os.getenv("V2_POOL") + ".csv"), index=False, encoding="utf-8-sig")
        _cr = pd.DataFrame(cand_rows)
        _cr = _cr[_cr.node_id.isin(set(test_ids))]
        _cr.to_csv(OUT_DIR / ("cand_test_" + os.getenv("V2_POOL") + ".csv"), index=False, encoding="utf-8-sig")
        _t = _pf[_pf.is_test]
        print("候选画像[测试节点] n=", len(_t),
              " 候选数中位=", round(float(_t.n_cand.median()), 1),
              " 空候选兜底率=", round(float(_t.fallback.mean()), 4),
              " 邻居唯一业务均值=", round(float(_t.n_neigh_biz.mean()), 2))

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
    if SKIP_OUT:  # 扫描模式：跳过 HTML 大文件生成
        return
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
