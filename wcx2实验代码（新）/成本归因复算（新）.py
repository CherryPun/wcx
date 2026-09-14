# -*- coding: utf-8 -*-
"""[wcx2 新增] 成本归因口径复算（新）

用桌面《七牛CDN掉量成本归因修正_研发操作数据_*.xlsx》的原始列，独立复算研发的归因规则，
与该表自带的"调整后"列对平，验证口径是否可复现：
  机房级 建议调账额 = max(0, 昨日机房总成本 × 前日CDN成本占比 − 昨日CDN成本)
  业务项级 追加成本 = 机房调账额 × (该项计费带宽下降G / 机房内CDN下降G合计)
  总账约束 各机房调账明细合计 = 0（不改机房总成本、不改大盘）
任一项超容差即退出码 1。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DESKTOP = Path.home() / "Desktop"
F = next(p for p in DESKTOP.glob("七牛CDN*.xlsx") if "成本归因修正" in p.name)
TOL = 0.05
fails: list[str] = []


def check(name: str, got: pd.Series, want: pd.Series, tol: float = TOL) -> None:
    err = (got - want).abs().max()
    ok = err <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:<34} max_abs_err={err:.6f}  n={len(got)}")
    if not ok:
        fails.append(f"{name} err={err:.4f}")


fac = pd.read_excel(F, "01_机房级调账汇总")
biz = pd.read_excel(F, "02_CDN追加承担_业务计费项")
opp = pd.read_excel(F, "03_其他业务成本冲回_节点级")
cdn = pd.read_excel(F, "04_CDN追加承担_节点级")
ops = pd.read_excel(F, "06_研发操作明细")

# --- 机房级 ---
realloc = (fac["昨日_机房成本"] * fac["前日_CDN成本占比"] - fac["昨日_CDN成本"]).clip(lower=0)
check("机房级 建议调账额 公式复算", realloc, fac["建议调账额_转回CDN承担"])
adj_cdn_cost = fac["昨日_CDN成本"] + fac["建议调账额_转回CDN承担"]
check("调整后_CDN成本", adj_cdn_cost, fac["调整后_CDN成本"])
check("调整后_CDN毛利", fac["昨日_CDN收入"] - adj_cdn_cost, fac["调整后_CDN毛利"])
adj_oth_cost = fac["昨日_其他业务成本"] - fac["建议调账额_转回CDN承担"]
check("调整后_其他业务成本=昨日-建议额", adj_oth_cost, fac["调整后_其他业务成本"])

# --- 业务计费项级 ---
mall = biz["建议调账额_转回CDN承担"]
w = biz["计费带宽下降G"] / biz["机房名"].map(biz.groupby("机房名")["计费带宽下降G"].sum())
check("业务项 分摊权重", w, biz["分摊权重"], tol=1e-6)
add = mall * w
check("业务项 追加成本", add, biz["追加成本_由CDN掉量承担"])
adj = biz["昨日_成本"] + add
check("业务项 调整后_成本", adj, biz["调整后_成本"])
check("业务项 调整后_毛利", biz["昨日_收入"] - adj, biz["调整后_毛利"])

# --- 总账约束 ---
check("总账 研发明细合计=0", pd.Series([ops["成本调整金额"].sum()]), pd.Series([0.0]), tol=0.5)
perroom = ops.groupby("机房名")["成本调整金额"].sum()
check("总账 每个机房合计=0", perroom, pd.Series(0.0, index=perroom.index), tol=0.5)
pair = opp.groupby("机房名")["成本调整金额"].sum() + cdn.groupby("机房名")["成本调整金额"].sum()
check("节点级 CDN追加+其他冲回 合计=0", pair, pd.Series(0.0, index=pair.index), tol=0.5)

print("\n--- 影响量级 ---")
print(f"受影响机房数        : {fac['建议调账额_转回CDN承担'].gt(0).sum()}")
print(f"建议转回CDN承担成本 : {fac['建议调账额_转回CDN承担'].sum():,.2f} 元")
print(f"CDN计费带宽下降     : {fac['CDN计费带宽下降G'].sum():,.2f} G")
print(f"调整前CDN毛利       : {fac['昨日_CDN毛利'].sum():,.2f} 元")
print(f"调整后CDN毛利       : {fac['调整后_CDN毛利'].sum():,.2f} 元")

if fails:
    print("\nFAILED: " + "; ".join(fails))
    sys.exit(1)
print("\nALL PASS：归因规则可从原始列复现，总账对平。")
