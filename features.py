"""轻量特征编码器（numpy only，不依赖 pandas）。

节点自有属性 + 决策变量 business 编码为特征矩阵：
- 类别特征：训练时见过的取值做 one-hot；未见过→全 0（避免训练/推理分布不一致）。
- 数值特征：直接透传（树模型不需要标准化）。
"""
from __future__ import annotations

import numpy as np


class Encoder:
    def __init__(self) -> None:
        self.cat_map: dict[str, dict] = {}
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.dim: int = 0

    def fit(self, rows: list[dict], cat_cols: list[str], num_cols: list[str]) -> "Encoder":
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.cat_map = {}
        for c in self.cat_cols:
            vals = sorted({r.get(c) for r in rows}, key=lambda x: str(x))
            self.cat_map[c] = {v: i for i, v in enumerate(vals)}
        self.dim = sum(len(m) for m in self.cat_map.values()) + len(self.num_cols)
        return self

    def transform(self, rows: list[dict]) -> np.ndarray:
        X = np.zeros((len(rows), self.dim), dtype=float)
        for ri, r in enumerate(rows):
            col = 0
            for c in self.cat_cols:
                m = self.cat_map[c]
                v = r.get(c)
                if v in m:
                    X[ri, col + m[v]] = 1.0
                col += len(m)
            for c in self.num_cols:
                try:
                    X[ri, col] = float(r.get(c) or 0.0)
                except (TypeError, ValueError):
                    X[ri, col] = 0.0
                col += 1
        return X
