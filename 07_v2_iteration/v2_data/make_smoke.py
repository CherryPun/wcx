# -*- coding: utf-8 -*-
"""从本机全库数据生成可入库的冒烟子集（一次性/按需重建）。

- 固定 seed、分层抽样 ~NODES 个 nonant 节点（保业务多样性），
- 输出 v2_data/smoke/smoke_attrs.csv + smoke_outcomes.csv（schema 同 exp_*）。
- 冒烟子集只用于"clone 后跑通流水线"，不承诺复现 21.0。
用法：python make_smoke.py  （需本地 _rerun/exp_attrs.csv 与 exp_outcomes_clean.csv）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RERUN = HERE / "_rerun"
SMOKE = HERE / "smoke"
NODES = 350
SEED = 20260909


def main():
    SMOKE.mkdir(exist_ok=True)
    attrs = pd.read_csv(RERUN / "exp_attrs.csv", dtype={"node_id": str})
    out = pd.read_csv(RERUN / "exp_outcomes_clean.csv", dtype={"node_id": str, "business": str})

    # 只保留满 7 天、且有 >=2 个不同业务的多业务节点（保证训练/测试有账可学）
    o = out[pd.to_numeric(out["outcome_days"], errors="coerce").ge(7)].copy()
    cnt = o.groupby("node_id")["business"].nunique()
    cand = cnt[cnt >= 2].index
    cand = cand[cand.isin(set(attrs["node_id"].astype(str)))]
    rng = np.random.RandomState(SEED)
    pick = pd.Series(sorted(cand)).sample(min(NODES, len(cand)), random_state=rng)
    pick_ids = set(pick.astype(str))

    sa = attrs[attrs["node_id"].astype(str).isin(pick_ids)].copy()
    so = o[o["node_id"].astype(str).isin(pick_ids)].copy()
    sa.to_csv(SMOKE / "smoke_attrs.csv", index=False, encoding="utf-8-sig")
    so.to_csv(SMOKE / "smoke_outcomes.csv", index=False, encoding="utf-8-sig")
    print(f"attrs {len(sa):,} 行 / {sa.node_id.nunique()} 节点")
    print(f"outcomes {len(so):,} 行 / {so.node_id.nunique()} 节点 / 业务 {so.business.nunique()} 个")
    print(f"写 {SMOKE}")


if __name__ == "__main__":
    main()
