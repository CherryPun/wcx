# -*- coding: utf-8 -*-
"""模型迭代①（新）：训练样本加"近端时效权重"（半衰期 14 天），不改其他。
输出加权的训练对文件，供 V5 build 使用。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
V3 = HERE / "recent_month_large_mainstream_v3_daily_weighted（新）"
HALF_LIFE_DAYS = 14.0

p = V3 / "v1_training_pairs_large_mainstream_recent_1m.csv"
d = pd.read_csv(p, dtype={"node_id": "string", "business": "string"}, low_memory=False)
d["sample_day"] = pd.to_datetime(d["sample_day"], errors="coerce")
d["sample_weight"] = pd.to_numeric(d["sample_weight"], errors="coerce").fillna(1.0)
end = d["sample_day"].max()
days = (end - d["sample_day"]).dt.days.clip(lower=0)
factor = 0.5 ** (days / HALF_LIFE_DAYS)
# 归一化：使加权系数均值=1，保持总权重尺度
factor = factor / factor.mean()
d["sample_weight"] = d["sample_weight"] * factor
d["recency_factor"] = factor.round(4)

out = HERE / "v1_training_pairs_recency（新）.csv"
d.to_csv(out, index=False, encoding="utf-8-sig")
print("rows", len(d), "->", out)
print("factor min/mean/max:", round(factor.min(), 4), round(factor.mean(), 4), round(factor.max(), 4))
print("end day:", end.date())
