# -*- coding: utf-8 -*-
"""[wcx2 新增] 标签可分性诊断（新）

用桌面两张《七牛CDN…》xlsx 的节点级明细，对「单位建设带宽毛利」做方差分解：
  目标 = 毛利金额 / 节点建设带宽Gbps ；eta²(业务) / eta²(机房) / eta²(业务×机房) / 残差
并做安慰剂检验：组内打乱成本调整归属（保持每机房调整合计=0），看实际是否优于随机。

注意：表 B `04_CDN追加承担_节点级` 的 `调整后_毛利金额` 整列为空，
因此修正后毛利一律用恒等式 `毛利金额 − 成本调整金额` 计算，并对 NaN 直接报错。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DESKTOP = Path.home() / "Desktop"
FA = next(p for p in DESKTOP.glob("七牛CDN*.xlsx") if "下钻业务掉量" in p.name)
FB = next(p for p in DESKTOP.glob("七牛CDN*.xlsx") if "成本归因修正" in p.name)


def eta2(y: pd.Series, key: pd.Series) -> float:
    y = pd.Series(np.asarray(pd.to_numeric(y, errors="coerce"), dtype=float))
    assert y.notna().all(), "eta2 收到 NaN：检查目标列"
    gm = y.mean()
    g = y.groupby(pd.Series(np.asarray(key)))
    ss_b = (g.count() * (g.mean() - gm) ** 2).sum()
    return float(ss_b / ((y - gm) ** 2).sum())


def decompose(df: pd.DataFrame, target: pd.Series, tag: str) -> float:
    k = df["业务"].astype(str) + "|" + df["机房名"].astype(str)
    e_b, e_r, e_br = eta2(target, df["业务"]), eta2(target, df["机房名"]), eta2(target, k)
    res = 1 - e_br
    print(f"{tag:<16} n={len(df):<5} eta²业务={e_b:.4f} eta²机房={e_r:.4f} 业务×机房={e_br:.4f} 残差={res:.4f}")
    return res


def load():
    b = pd.concat([pd.read_excel(FB, "03_其他业务成本冲回_节点级"),
                   pd.read_excel(FB, "04_CDN追加承担_节点级")], ignore_index=True)
    b["bw"] = pd.to_numeric(b["节点建设带宽Gbps"], errors="coerce")
    b = b[b["bw"] > 0].copy()
    b["cost_adj"] = pd.to_numeric(b["成本调整金额"], errors="coerce")
    b["u0"] = pd.to_numeric(b["毛利金额"], errors="coerce") / b["bw"]
    b["u1"] = (pd.to_numeric(b["毛利金额"], errors="coerce") - b["cost_adj"]) / b["bw"]
    return b


def main() -> None:
    print("=== 表B 修正前/修正后（全量 03+04）===")
    b = load()
    r0 = decompose(b, b["u0"], "修正前")
    r1 = decompose(b, b["u1"], "修正后")
    print(f"残差变化：{r0:.4f} → {r1:.4f}（相对 {(r1 - r0) / r0 * 100:+.1f}%）")

    print("\n--- 排序影响（全量）---")
    print(f"负单G毛利占比  before={100 * (b['u0'] < 0).mean():.1f}%  after={100 * (b['u1'] < 0).mean():.1f}%")
    flip = ((b["u0"] < 0) != (b["u1"] < 0)).mean() * 100
    print(f"符号翻转        {flip:.1f}%   行秩相关 Spearman={b['u0'].corr(b['u1'], method='spearman'):.4f}")
    gm = b.groupby("业务")[["u0", "u1"]].mean()
    print(f"业务级 平均单G毛利 秩相关={gm['u0'].corr(gm['u1'], method='spearman'):.4f}")

    print("\n--- 安慰剂：组内打乱成本调整归属（保持每机房合计=0）---")
    rng = np.random.default_rng(0)
    k = b["业务"].astype(str) + "|" + b["机房名"].astype(str)
    out = []
    for _ in range(300):
        perm = b.groupby("机房名")["cost_adj"].transform(lambda x: rng.permutation(x.values))
        u = (pd.to_numeric(b["毛利金额"], errors="coerce") - perm) / b["bw"]
        out.append((1 - eta2(u, k), eta2(u, b["业务"])))
    o = np.array(out)
    print(f"安慰剂残差 mean={o[:, 0].mean():.4f} [p5={np.percentile(o[:, 0], 5):.4f} p95={np.percentile(o[:, 0], 95):.4f}] "
          f"| 实际 {r1:.4f} 分位={100 * (o[:, 0] <= r1).mean():.1f}%")
    print(f"安慰剂 eta²业务 mean={o[:, 1].mean():.4f} | 实际 {eta2(b['u1'], b['业务']):.4f}")

    print("\n=== 表A 基线日 / 掉量日 ===")
    a = pd.read_excel(FA, "05_节点级明细")
    a["bw"] = pd.to_numeric(a["节点建设带宽Gbps"], errors="coerce")
    a = a[a["bw"] > 0].copy()
    decompose(a, pd.to_numeric(a["毛利金额_基线日"], errors="coerce") / a["bw"], "基线日 09-12")
    decompose(a, pd.to_numeric(a["毛利金额_掉量日"], errors="coerce") / a["bw"], "掉量日 09-13")


if __name__ == "__main__":
    main()
