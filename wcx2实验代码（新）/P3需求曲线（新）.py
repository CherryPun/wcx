# -*- coding: utf-8 -*-
"""P3 需求曲线（新）：按业务拟合 traffic ~ D*(1-exp(-S/S0))，输出 D/S0/R²/D 稳定性。
数据：业务日供给流量（新）.csv（本地）；业务范围取主流 allowlist。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "业务日供给流量（新）.csv", dtype={"customerId": str})
d["day"] = pd.to_datetime(d["day"])
d["S"] = pd.to_numeric(d["supply_mbps"], errors="coerce")
d["Traf"] = pd.to_numeric(d["traffic_bps"], errors="coerce") / 1e6
d = d.dropna(subset=["S", "Traf"])
d = d[(d["S"] > 0) & (d["Traf"] > 0)]

allow = ROOT / "mainstream_business_allowlist.csv"
allow_set = set(pd.read_csv(allow, dtype=str)["business"].astype(str)) if allow.exists() else set(d["customerId"])


def f(S, D, S0):
    return D * (1 - np.exp(-S / S0))


rows = []
for biz, g in d[d.customerId.isin(allow_set)].groupby("customerId"):
    if g.day.nunique() < 60 or g.S.std() / max(g.S.mean(), 1e-9) < 0.2:
        continue
    S, Traf = g["S"].to_numpy(float), g["Traf"].to_numpy(float)
    try:
        p, _ = curve_fit(f, S, Traf, p0=[Traf.max() * 1.2, S.mean()], bounds=([0, 1e-6], [np.inf, np.inf]), maxfev=20000)
        D, S0 = float(p[0]), float(p[1])
        pred = f(S, D, S0)
        r2 = 1 - float(((Traf - pred) ** 2).sum() / max(((Traf - Traf.mean()) ** 2).sum(), 1e-9))
    except Exception:
        D = S0 = r2 = float("nan")
    # D 稳定性：前后两半
    Ds = []
    g2 = g.sort_values("day")
    for half in (g2.iloc[: len(g2) // 2], g2.iloc[len(g2) // 2:]):
        try:
            p2, _ = curve_fit(f, half["S"].to_numpy(float), half["Traf"].to_numpy(float),
                              p0=[half["Traf"].max() * 1.2, half["S"].mean()], bounds=([0, 1e-6], [np.inf, np.inf]), maxfev=20000)
            Ds.append(float(p2[0]))
        except Exception:
            Ds.append(np.nan)
    stable = (not np.isnan(Ds).any()) and abs(Ds[1] - Ds[0]) <= 0.2 * max(Ds[0], Ds[1])
    rows.append({"business": biz, "days": int(g.day.nunique()), "D_mbps": round(D, 1), "S0_mbps": round(S0, 1),
                 "r2": round(r2, 3), "D_half1": round(Ds[0], 1), "D_half2": round(Ds[1], 1), "stable": stable})

res = pd.DataFrame(rows).sort_values("r2", ascending=False)
res.to_csv(ROOT / "P3需求曲线参数（新）.csv", index=False, encoding="utf-8-sig")
o = io.StringIO()
o.write("== P3 需求曲线拟合（主流业务，≥60 天且供给 CV≥0.2） ==\n")
o.write("业务数 %d；R²≥0.3 %d；D 稳定 %d\n\n" % (len(res), int((res.r2 >= 0.3).sum()), int(res.stable.sum())))
o.write(res.head(15).to_string(index=False))
txt = o.getvalue()
(ROOT / "P3需求曲线（新）.txt").write_text(txt, encoding="utf-8")
print(txt)
