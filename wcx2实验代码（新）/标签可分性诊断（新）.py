# -*- coding: utf-8 -*-
"""[wcx2 新增] 标签可分性诊断（新）

用桌面两张《七牛CDN…》xlsx 的节点级明细，对「单位建设带宽毛利」做方差分解：
  目标 = 毛利 / 节点建设带宽Gbps
  eta²(业务) / eta²(机房) / eta²(业务×机房) / eta²(节点) / 残差
并比较「修正前 vs 修正后」（表 B 有 调整后_毛利金额）。
目的：判定平台标签失真主要来自"业务间成本归属错"（可修）还是"节点级真实噪声"（不可修）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DESKTOP = Path.home() / "Desktop"
FA = next(p for p in DESKTOP.glob("七牛CDN*.xlsx") if "下钻业务掉量" in p.name)
FB = next(p for p in DESKTOP.glob("七牛CDN*.xlsx") if "成本归因修正" in p.name)


def eta2(y: pd.Series, key: pd.Series) -> float:
    """一维方差解释度（组间平方和 / 总平方和）。"""
    y = pd.to_numeric(y, errors="coerce")
    m = y.notna()
    y, key = y[m], key[m]
    gm = y.mean()
    g = pd.DataFrame({"y": y, "k": key.values}).groupby("k")["y"]
    ss_between = (g.count() * (g.mean() - gm) ** 2).sum()
    ss_total = ((y - gm) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def decompose(df: pd.DataFrame, target: str, tag: str) -> None:
    df = df.copy()
    df["u"] = df[target] / df["节点建设带宽Gbps"]
    df = df[df["节点建设带宽Gbps"] > 0]
    print(f"\n=== {tag} | n={len(df)} ===")
    e_b = eta2(df["u"], df["业务"])
    e_r = eta2(df["u"], df["机房名"])
    e_br = eta2(df["u"], df["业务"].astype(str) + "|" + df["机房名"].astype(str))
    e_n = eta2(df["u"], df["nodeId"])
    print(f"eta²(业务)         {e_b:6.3f}")
    print(f"eta²(机房)         {e_r:6.3f}")
    print(f"eta²(业务×机房)    {e_br:6.3f}")
    print(f"eta²(节点)         {e_n:6.3f}")
    print(f"残差(1-业务×机房)  {1 - e_br:6.3f}")


# --- 表 B：修正前 vs 修正后（受影响机房/业务）---
b = pd.concat([pd.read_excel(FB, "03_其他业务成本冲回_节点级"),
               pd.read_excel(FB, "04_CDN追加承担_节点级")], ignore_index=True)
if "节点建设带宽Gbps" in b and b["节点建设带宽Gbps"].max() < 100:  # 单位可能是 Gbps→ 归一
    b["节点建设带宽Gbps"] = b["节点建设带宽Gbps"]
decompose(b, "毛利金额", "表B 修正前")
decompose(b, "调整后_毛利金额", "表B 修正后")

# --- 表 A：下钻业务，基线日 vs 掉量日 ---
a = pd.read_excel(FA, "05_节点级明细")
a = a.rename(columns={"节点建设带宽Gbps": "节点建设带宽Gbps"})
decompose(a, "毛利金额_基线日", "表A 基线日(09-12)")
decompose(a, "毛利金额_掉量日", "表A 掉量日(09-13)")

# --- 同业务跨节点离散度（判定归因是否错）---
print("\n=== 表B 各业务 表现 单位带宽毛利 的跨节点离散 ===")
b2 = b[b["节点建设带宽Gbps"] > 0].copy()
b2["u"] = b2["调整后_毛利金额"] / b2["节点建设带宽Gbps"]
g = b2.groupby("业务")["u"].agg(["count", "mean", "std", lambda s: s.quantile(.9) - s.quantile(.1)])
g.columns = ["count", "mean", "std", "p90_p10"]
g["cv"] = (g["std"] / g["mean"].abs()).replace([np.inf, -np.inf], np.nan)
print(g.sort_values("count", ascending=False).round(3).to_string())
