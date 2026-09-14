# -*- coding: utf-8 -*-
"""[wcx2 新增] 一致性检查（新）

把"刚发生过的 NaN 静默丢样本"这类误差变成 5 秒内可发现的失败，而不是写进交付文档才发现。
两部分：
  1) fixture 自检：合成含 NaN 陷阱的小样本，验证完整性断言会真的失败（pandas 的 .max()/abs() 会跳过 NaN）；
  2) 审计对账：用 RJ 审计文件 node_day_sampling_audit_v3.csv 校验节点日守恒、
     剔除类别与 clean/outcome 行数一致、"一天一个干净业务"语义、七牛 canonical 归并。

退出码 0 = 全部通过；非 0 = 有断言失败（不要用 .max() 当判据）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "recent_month_large_mainstream_v3_daily_weighted（新）"
fails: list[str] = []


def require_complete(s: pd.Series, name: str) -> None:
    """完整性 fail-fast：NaN 一律报错（pandas 的聚合会静默跳过 NaN）。"""
    bad = int(s.isna().sum())
    if bad:
        raise AssertionError(f"{name}: 含 {bad}/{len(s)} 个 NaN —— 拒绝在此列上计算")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<44} {detail}")
    if not ok:
        fails.append(name)


def fixture_self_test() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    # 错误做法：.max() 跳过 NaN，4 行里 1 行缺失却给出"完全一致(max=0)"的结论
    assert (df["a"] - df["b"]).abs().max() == 0.0, "fixture 前提不成立"
    raised = False
    try:
        require_complete(df["a"], "fixture.a")
    except AssertionError:
        raised = True
    check("fixture: 完整性断言能拦住 NaN", raised)


def main() -> None:
    print("=== 1. fixture 自检（含 NaN 陷阱）===")
    fixture_self_test()

    if not V3.exists():
        print("\n[SKIP] 未找到本地 V3 产物目录，跳过审计对账（clone 后属正常）")
        return finish()

    au = pd.read_csv(V3 / "node_day_sampling_audit_v3.csv", low_memory=False)
    out = pd.read_csv(V3 / "multibusiness_outcomes_large_mainstream_v3_daily.csv",
                      usecols=["node_id", "sample_day"], low_memory=False)

    print(f"\n=== 2. 审计对账（audit n={len(au):,}）===")
    require_complete(au["status"], "audit.status")
    require_complete(au["active_mainstream_count"], "audit.active_mainstream_count")

    keys = au[["node_id", "sample_day"]]
    check("node-day 唯一", len(keys) == len(keys.drop_duplicates()),
          f"n={len(keys):,} uniq={len(keys.drop_duplicates()):,}")

    clean = au["status"].eq("clean")
    excl = au["status"].str.startswith("excluded_")
    check("status 只有 clean/excluded_*", bool((clean | excl).all()))
    check("守恒 clean+excluded=总数", int(clean.sum() + excl.sum()) == len(au),
          f"clean={int(clean.sum()):,} excluded={int(excl.sum()):,}")
    check("clean 行无 reason", int(au.loc[clean, "reason"].notna().sum()) == 0)
    check("excluded 行有 reason", int(au.loc[excl, "reason"].isna().sum()) == 0)

    check("clean == outcomes 节点日数", int(clean.sum()) == len(out), f"{int(clean.sum()):,} vs {len(out):,}")

    c = au[clean]
    check("clean 行按 canonical 只含一个业务（一天一个干净业务）",
          bool((c["active_mainstream_count"] == 1).all()),
          f"violations={int((c['active_mainstream_count'] != 1).sum())}")
    check("clean 行无虚拟绑定歧义", int((c["ambiguous_virtual_count"] > 0).sum()) == 0)
    check("clean 行 selected_business 非空", int(c["selected_business"].isna().sum()) == 0)

    check("多业务节点日被整日剔除",
          int(au["status"].eq("excluded_multiple_active_mainstream_businesses").sum())
          == int((au["active_mainstream_count"] > 1).sum()),
          f"excluded={int(au['status'].eq('excluded_multiple_active_mainstream_businesses').sum())} count>1={int((au['active_mainstream_count']>1).sum())}")

    qi = c[c["selected_business"] == 10000280]
    if len(qi):
        canon = qi["active_mainstream_canonical_ids"].astype(str).str.split("|")
        check("七牛 canonical 归并到 10000280 且唯一",
              bool(canon.apply(lambda ids: ids == ["10000280"]).all()),
              f"n={len(qi):,}")

    if len(out):
        keep = set(map(tuple, keys[clean].itertuples(index=False, name=None)))
        ok = set(map(tuple, out.itertuples(index=False, name=None)))
        check("outcomes 节点日集合 == clean 节点日集合", ok == keep,
              f"outcomes={len(ok):,} clean={len(keep):,} 差集={len(ok ^ keep):,}")

    finish()


def finish() -> None:
    if fails:
        print("\nFAILED: " + "; ".join(fails))
        sys.exit(1)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
