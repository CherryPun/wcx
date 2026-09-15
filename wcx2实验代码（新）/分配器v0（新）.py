# -*- coding: utf-8 -*-
"""[wcx2 新增] 分配器 v0（新）：池×业务份额 + 分段线性凹目标（LP 松弛）

C3-v0（无外部依赖版）：
  - 池 = 省×运营商（v0 代理"机房"粒度）
  - 决策变量 x[p,b] ≥ 0：池 p 分配给业务 b 的 Mbps（连续份额，非 0-1 整节点切换）
  - 约束：Σ_b x[p,b] ≤ pool_capacity[p]；Σ_p x[p,b] ≤ business_ceiling[b]
  - 目标：分段线性**凹**（经验分位切点，边际递减）+ 池偏好项（同业务优先给单位收益高的池）
  - 对照基线：现行"先按分排序再卡容量"（greedy）
  - 指标：容量损失 / planned 溢出 / hold 占比 / 扩容缺口

用法：
  python "分配器v0（新）.py" --self-test                    # 合成用例自检（LP 应不劣于 greedy）
  python "分配器v0（新）.py" --recommendations <csv> --capacities <csv> --output <csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REC = ROOT / "recent_month_large_mainstream_v5_sw_7c2.0（新）" / "v5_node_recommendations.csv"
DEFAULT_CAP = ROOT / "recent_month_large_mainstream_v6_capacity_cap20_curfull（新）" / "v6_business_capacity_summary.csv"
SEGMENT_SLOPES = (1.0, 0.6, 0.3)
POOL_PREFERENCE = 0.1
TOP_K = 3


def solve_lp(items: pd.DataFrame, pool_cap: pd.Series, biz_cap: pd.Series, knots: pd.DataFrame) -> pd.Series:
    """max Σ_b 凹分段价值(X_b) + ε Σ g[p,b] x[p,b]，返回 (pool,business)->Mbps。"""
    from scipy.optimize import linprog

    items = items.reset_index(drop=True)
    npairs, nbiz = len(items), len(biz_cap)
    biz_index = {b: i for i, b in enumerate(biz_cap.index)}
    pool_index = {p: i for i, p in enumerate(pool_cap.index)}
    cols = list(items.columns)

    # 变量：x(0..npairs-1) + t(b, seg1..3)
    nvar = npairs + 3 * nbiz
    c = np.zeros(nvar)
    for i, row in items.iterrows():
        c[i] = -(POOL_PREFERENCE * float(row["g_pb"]))
    for b, bi in biz_index.items():
        k1, k2 = float(knots.loc[b, "k1"]), float(knots.loc[b, "k2"])
        g = float(knots.loc[b, "g_mean"])
        for s, slope in enumerate(SEGMENT_SLOPES):
            c[npairs + 3 * bi + s] = -(slope * g)

    A, b_ub = [], []
    for p, pi in pool_index.items():
        row = np.zeros(nvar)
        for i, it in items.iterrows():
            if it["pool"] == p:
                row[i] = 1.0
        A.append(row); b_ub.append(float(pool_cap[p]))
    for biz, bi in biz_index.items():
        row = np.zeros(nvar)
        for i, it in items.iterrows():
            if it["business"] == biz:
                row[i] = 1.0
        A.append(row); b_ub.append(float(biz_cap[biz]))

    A_eq, b_eq = [], []
    for biz, bi in biz_index.items():
        row = np.zeros(nvar)
        for i, it in items.iterrows():
            if it["business"] == biz:
                row[i] = -1.0
        row[npairs + 3 * bi + 0] = 1.0
        row[npairs + 3 * bi + 1] = 1.0
        row[npairs + 3 * bi + 2] = 1.0
        A_eq.append(row); b_eq.append(0.0)

    bounds = [(0.0, float(items.loc[i, "max_mbps"])) for i in range(npairs)]
    for biz, bi in biz_index.items():
        k1 = float(knots.loc[biz, "k1"]); k2 = float(knots.loc[biz, "k2"])
        bounds += [(0.0, k1), (0.0, max(k2 - k1, 0.0)), (0.0, float(biz_cap[biz]))]

    res = linprog(c, A_ub=np.array(A) if A else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP 未求解成功：{res.message}")
    x = res.x[:npairs]
    out = pd.Series(x, index=pd.MultiIndex.from_frame(items[["pool", "business"]]), name="alloc_mbps")
    return out[out > 1e-6]


def greedy(items: pd.DataFrame, pool_cap: pd.Series, biz_cap: pd.Series) -> pd.Series:
    """现行做法：按单位收益降序，逐条吃满池容量/业务上限。"""
    left_pool = pool_cap.astype(float).to_dict()
    left_biz = biz_cap.astype(float).to_dict()
    out = {}
    for _, row in items.sort_values("g_pb", ascending=False).iterrows():
        take = min(row["max_mbps"], left_pool.get(row["pool"], 0.0), left_biz.get(row["business"], 0.0))
        if take > 1e-6:
            out[(row["pool"], row["business"])] = take
            left_pool[row["pool"]] -= take
            left_biz[row["business"]] -= take
    return pd.Series(out, name="alloc_mbps")


def concave_value(alloc: pd.Series, knots: pd.DataFrame) -> float:
    """同一凹目标下的总价值（用于 LP vs greedy 对照）。"""
    total = 0.0
    biz_level = set(alloc.index.get_level_values(1)) if len(alloc) else set()
    for biz in knots.index:
        x = float(alloc.xs(biz, level=1).sum()) if biz in biz_level else 0.0
        k1, k2, g = float(knots.loc[biz, "k1"]), float(knots.loc[biz, "k2"]), float(knots.loc[biz, "g_mean"])
        total += min(x, k1) * g
        total += min(max(x - k1, 0.0), k2 - k1) * SEGMENT_SLOPES[1] * g
        total += max(x - k2, 0.0) * SEGMENT_SLOPES[2] * g
    return total


def metrics(alloc: pd.Series, pool_cap: pd.Series, biz_cap: pd.Series, demand: pd.Series, kind: str) -> dict:
    per_biz = alloc.groupby(level=1).sum() if len(alloc) else pd.Series(dtype=float)
    per_pool = alloc.groupby(level=0).sum() if len(alloc) else pd.Series(dtype=float)
    total_cap = float(pool_cap.sum())
    used = float(per_pool.sum())
    overflow = float((per_biz.reindex(biz_cap.index).fillna(0) - biz_cap).clip(lower=0).sum())
    gap = float((demand.reindex(biz_cap.index).fillna(0) - biz_cap).clip(lower=0).sum())
    return {
        "方案": kind,
        "总容量(Gbps)": round(total_cap / 1000, 1),
        "已分配(Gbps)": round(used / 1000, 1),
        "容量损失": round(1 - used / total_cap, 4) if total_cap else np.nan,
        "planned溢出(Gbps)": round(overflow / 1000, 3),
        "扩容缺口(Gbps)": round(gap / 1000, 1),
        "命中业务数": int(per_biz.gt(0).sum()),
    }


def build_items(rec: pd.DataFrame, cap: pd.DataFrame, resource_model: str = "headroom") -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """resource_model=headroom：池容量 = 池内**唯一节点**带宽×target − 当前占用（不重复计数）；
       =nodesum：旧口径，池内 top1-3 候选带宽求和（会重复计入同一节点）。"""
    rows = []
    for k in range(1, TOP_K + 1):
        cols = [f"business_top{k}", f"miner_unit_income_top{k}"]
        if not set(cols) <= set(rec.columns):
            continue
        sub = rec[["province", "isp", "build_bandwidth_mbps", *cols]].copy()
        sub.columns = ["province", "isp", "bw", "business", "g"]
        sub["pool"] = sub["province"].astype(str) + "|" + sub["isp"].astype(str)
        sub["k"] = k
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    long["business"] = pd.to_numeric(long["business"], errors="coerce")
    long["g"] = pd.to_numeric(long["g"], errors="coerce")
    long["bw"] = pd.to_numeric(long["bw"], errors="coerce").fillna(0.0)
    cand = long.dropna(subset=["business"]).copy()
    cand["business"] = cand["business"].astype("int64")

    agg = cand.groupby(["pool", "business"]).agg(
        max_mbps=("bw", "sum"),
        g_pb=("g", "mean"),
    ).reset_index()
    caps = cap[cap["pool_level"].eq("business")].copy()
    caps["business"] = pd.to_numeric(caps["business"], errors="coerce")
    caps = caps.dropna(subset=["business"])
    biz_cap = caps.set_index(caps["business"].astype("int64"))["operational_capacity_ceiling_mbps"].astype(float)
    demand = agg.groupby("business")["max_mbps"].sum()

    pool_nodes = rec.assign(
        pool=rec["province"].astype(str) + "|" + rec["isp"].astype(str),
        bw=pd.to_numeric(rec["build_bandwidth_mbps"], errors="coerce").fillna(0.0),
    )
    if resource_model == "headroom" and "current_business" in pool_nodes.columns:
        cur = pd.to_numeric(pool_nodes["current_business"], errors="coerce").notna()
        total = pool_nodes.groupby("pool")["bw"].sum()
        used = pool_nodes[cur].groupby("pool")["bw"].sum()
        pool_cap = (total * 0.7 - used.reindex(total.index).fillna(0.0)).clip(lower=0.0)
    else:
        pool_cap = agg.groupby("pool")["max_mbps"].sum()

    g_mean = agg.groupby("business")["g_pb"].mean()
    knots = pd.DataFrame({"g_mean": g_mean}).reindex(biz_cap.index).fillna(0.0)
    sup = agg.groupby("business")["max_mbps"].sum().reindex(biz_cap.index).fillna(0.0)
    knots["k1"] = sup * 0.5
    knots["k2"] = sup * 0.9
    return agg, pool_cap, biz_cap, demand, knots


def self_test() -> int:
    """合成用例：边际递减下 LP 应优于按单位收益贪心。"""
    items = pd.DataFrame({
        "pool": ["p1", "p2"],
        "business": [1, 1],
        "max_mbps": [300.0, 300.0],
        "g_pb": [1.0, 1.0],
    })
    pool_cap = pd.Series({"p1": 100.0, "p2": 100.0})
    biz_cap = pd.Series({1: 200.0, 2: 200.0})
    knots = pd.DataFrame({"g_mean": [1.0, 0.8], "k1": [50.0, 200.0], "k2": [200.0, 400.0]}, index=[1, 2])
    items_b = pd.DataFrame({
        "pool": ["p1", "p2"],
        "business": [2, 2],
        "max_mbps": [300.0, 300.0],
        "g_pb": [0.8, 0.8],
    })
    items = pd.concat([items, items_b], ignore_index=True)
    lp = solve_lp(items, pool_cap, biz_cap, knots)
    gr = greedy(items, pool_cap, biz_cap)

    def value(alloc: pd.Series) -> float:
        tot = 0.0
        for biz, bi in {1: 0, 2: 1}.items():
            x = float(alloc.xs(biz, level=1).sum()) if biz in alloc.index.get_level_values(1) else 0.0
            k1, k2 = knots.loc[biz, ["k1", "k2"]]
            s = min(x, k1) * knots.loc[biz, "g_mean"]
            s += min(max(x - k1, 0), k2 - k1) * 0.6 * knots.loc[biz, "g_mean"]
            s += max(x - k2, 0) * 0.3 * knots.loc[biz, "g_mean"]
            tot += s
        return tot

    lp_v, gr_v = value(lp), value(gr)
    print(f"[self-test] LP value={lp_v:.1f} | greedy value={gr_v:.1f}")
    print(f"[self-test] LP 分配: {dict(lp.round(1))}")
    print(f"[self-test] greedy 分配: {dict(gr.round(1))}")
    ok = lp_v >= gr_v - 1e-6
    print("PASS  LP 不劣于 greedy" if ok else "FAIL  LP 劣于 greedy")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--recommendations", type=Path, default=DEFAULT_REC)
    ap.add_argument("--capacities", type=Path, default=DEFAULT_CAP)
    ap.add_argument("--output", type=Path, default=ROOT / "分配结果v0（新）.csv")
    ap.add_argument("--resource-model", choices=["headroom", "nodesum"], default="headroom",
                    help="headroom=池容量取(唯一节点带宽×0.7−当前占用)；nodesum=旧口径(候选带宽求和，会重复计数)")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.recommendations.exists() or not args.capacities.exists():
        print("[SKIP] 缺少推荐或容量产物")
        return 0

    rec = pd.read_csv(args.recommendations, low_memory=False)
    cap = pd.read_csv(args.capacities, low_memory=False)
    items, pool_cap, biz_cap, demand, knots = build_items(rec, cap)
    lp = solve_lp(items, pool_cap, biz_cap, knots)
    gr = greedy(items, pool_cap, biz_cap)

    current = rec.dropna(subset=["current_business"]).assign(
        business=lambda d: pd.to_numeric(d["current_business"], errors="coerce")
    ).dropna(subset=["business"])
    hold_pools = set()
    if len(current):
        top = lp.reset_index().groupby("pool").apply(lambda g: g.loc[g["alloc_mbps"].idxmax(), "business"])
        cur = current.assign(pool=lambda d: d["province"].astype(str) + "|" + d["isp"].astype(str)) \
            .groupby(["pool", "business"])["build_bandwidth_mbps"].sum().reset_index() \
            .sort_values("build_bandwidth_mbps", ascending=False).drop_duplicates("pool").set_index("pool")["business"]
        hold_pools = {p for p in top.index if p in cur.index and float(top[p]) == float(cur[p])}

    print("\n=== 分配器 v0 对照（池=省×运营商，业务上限=v6 业务级 ceiling）===")
    print(f"资源模型={args.resource_model} | 池数 {len(pool_cap)} | 候选(池×业务)组合 {len(items)} | 业务上限数 {len(biz_cap)}")
    rows = [metrics(lp, pool_cap, biz_cap, demand, "LP 份额(份额+凹目标)"),
            metrics(gr, pool_cap, biz_cap, demand, "greedy(现做法)")]
    print(pd.DataFrame(rows).to_string(index=False))
    lp_v, gr_v = concave_value(lp, knots), concave_value(gr, knots)
    print(f"目标值（同一凹函数）：LP={lp_v:,.0f} | greedy={gr_v:,.0f} | LP 相对 +{(lp_v - gr_v) / max(gr_v, 1e-9):.2%}")
    print(f"hold 池占比（LP 分配的首选业务 == 当前业务）: {len(hold_pools)}/{len(pool_cap)} = {len(hold_pools)/max(len(pool_cap),1):.1%}")

    out = lp.reset_index().rename(columns={"level_0": "pool", "level_1": "business"})
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[out] -> {args.output.name} ({len(out)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
