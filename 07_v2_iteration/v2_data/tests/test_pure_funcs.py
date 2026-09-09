# -*- coding: utf-8 -*-
"""纯函数单测（无第三方数据依赖，python tests/test_pure_funcs.py 直接跑）。
覆盖：七牛 family 归并、门禁阈值、时间外切分、node_matrix 维度/归一、build_stats。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_e2e import family_of, gate_bad_from, temporal_split, node_matrix, build_stats  # noqa: E402


def test_family_of():
    nm = {"1001": "七牛CDN-虚拟-A", "1002": "七牛特招-虚拟", "1003": "普通客户"}
    assert family_of("1001", nm) == "FQCDNV"
    assert family_of("1002", nm) == "FQTZV"
    assert family_of("1003", nm) == "1003"
    assert family_of("1004", nm) == "1004"     # 不在映射（业务名缺失）不归并
    assert family_of("1001", {}) == "1001"     # 空映射不归并


def test_gate_bad():
    df = pd.DataFrame({
        "node_id": ["ant0001", "ant0002", "ant0003", "node0004", "node0005", "node0006"],
        "business": ["B1", "B1", "B1", "B1", "B2", "B2"],
    })
    bad = gate_bad_from(df, thr=0.9)
    assert "B1" not in bad and "B2" not in bad  # B1 ant 3/4=0.75 <0.9
    df2b = pd.DataFrame({"node_id": ["ant1", "ant2", "ant3", "ant4", "ant5", "ant6"], "business": ["B3"] * 6})
    assert "B3" in gate_bad_from(df2b, 0.9)     # ant 6/6=1.0 -> 进
    assert "B4" not in gate_bad_from(pd.DataFrame({"node_id": ["n1", "n2"], "business": ["B4"] * 2}), 0.9)


def test_temporal_split():
    df = pd.DataFrame({"node_id": ["n1", "n2", "n3", "n4", "n5"],
                       "intro_dt": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])})
    tr, te = temporal_split(df, ratio=0.8)
    assert tr == {"n1", "n2", "n3", "n4"} and te == {"n5"}
    tr2, te2 = temporal_split(df, cut_dt=pd.Timestamp("2026-01-03"))
    assert tr2 == {"n1", "n2"} and te2 == {"n3", "n4", "n5"}
    tr3, te3 = temporal_split(df.iloc[::-1].reset_index(drop=True), ratio=0.8)  # 乱序也按上线时间排
    assert tr3 == {"n1", "n2", "n3", "n4"}


def test_node_matrix():
    n = pd.DataFrame({
        "node_id": [f"n{i}" for i in range(6)],
        "province": ["浙", "浙", "粤", "粤", "京", "京"],
        "isp": ["电信", "联通", "电信", "联通", "电信", "联通"],
        "resourcetype": ["dedicated"] * 6,
        "deliverytype": ["idc"] * 6,
        "nattype": ["公网", "公网", "锥型", "锥型", "对称", "对称"],
        "dialtype": ["静态", "静态", "拨号", "拨号", "拨号", "静态"],
        "device_type": ["d1", "d1", "d1", "d2", "d2", "d2"],
        "os": ["linux"] * 6,
        "bw": [1000, 2000, 1000, 3000, 1000, 4000],
        "corenum": [8, 16, 8, 32, 8, 64],
    })
    X, _enc_cat, _enc_num = node_matrix(n)
    import scipy.sparse as sp
    assert sp.issparse(X) and X.shape[0] == 6 and X.shape[1] > 0   # (节点 × 特征)
    row = np.asarray(np.sqrt(X.multiply(X).sum(axis=1))).ravel()
    assert np.allclose(row, 1.0, atol=1e-4)          # L2 归一
    assert X[0].dot(X[0].T).toarray()[0, 0] > 0.99   # 自身余弦 = 1（相似度可用）
    # 部分数值列全缺失时仍可构造（bw 有值、corenum 全缺失，模拟 jarvis 类字段缺失）
    n2 = n.copy()
    n2["corenum"] = np.nan
    X2, *_ = node_matrix(n2)
    assert X2.shape[0] == 6 and X2.shape[1] > 0


def test_build_stats():
    d = pd.DataFrame({
        "node_id": ["a", "a", "b", "c", "d"],
        "business": ["X", "X", "X", "X", "Y"],
        "cum_cost_7d": [100.0, 200.0, 300.0, 400.0, 999.0],
        "cum_profit_7d": [10.0, 20.0, 30.0, 40.0, 5.0],
    })
    st = build_stats(d).set_index("business")
    assert st.loc["X", "support"] == 3 and st.loc["Y", "support"] == 1
    assert abs(st.loc["X", "cost_mean"] - 250.0) < 1e-6
    assert st.loc["X", "profit_mean_log"] > 0
    assert "cost_mean_log" in st.columns


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npass = 0
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
        npass += 1
    print(f"== {npass}/{len(fns)} tests passed ==")


if __name__ == "__main__":
    main()
