#!/usr/bin/env python3
"""节点最佳业务推荐模型与真实结果评估。"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder


UNK = "__UNK__"


class FeatureEncoder:
    """独热编码低基数类别，频率编码高基数类别，避免类别 ID 的伪顺序。"""

    def __init__(self, cat_features: list[str], num_features: list[str], business_col: str):
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.business_col = business_col
        self.low_card_cols: list[str] = []
        self.high_card_cols: list[str] = []
        self.frequency_maps: dict[str, dict[str, float]] = {}
        self.onehot: OneHotEncoder | None = None

    @staticmethod
    def _as_text(series: pd.Series) -> pd.Series:
        return series.fillna(UNK).astype(str)

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        all_cat_cols = self.cat_features + [self.business_col]
        for column in all_cat_cols:
            values = self._as_text(frame[column])
            cardinality = values.nunique(dropna=False)
            # 业务必须独热；高基数节点属性改用训练集频率编码。
            if column == self.business_col or cardinality <= 128:
                self.low_card_cols.append(column)
            else:
                self.high_card_cols.append(column)
                counts = values.value_counts(normalize=True)
                self.frequency_maps[column] = counts.to_dict()

        if self.low_card_cols:
            try:
                self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                self.onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
            self.onehot.fit(frame[self.low_card_cols].apply(self._as_text))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.low_card_cols and self.onehot is not None:
            parts.append(self.onehot.transform(frame[self.low_card_cols].apply(self._as_text)))

        for column in self.high_card_cols:
            values = self._as_text(frame[column])
            encoded = values.map(self.frequency_maps[column]).fillna(0.0).to_numpy(dtype=float)
            parts.append(encoded.reshape(-1, 1))

        for column in self.num_features:
            numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            parts.append(numeric.to_numpy(dtype=float).reshape(-1, 1))

        if not parts:
            return np.empty((len(frame), 0), dtype=float)
        return np.concatenate(parts, axis=1)


class BestBusinessRecommender:
    """分别预测 7 天成本和 7 天运营利润，再对候选业务排序。"""

    def __init__(self, cat_features, num_features, business_col="business"):
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.business_col = business_col
        self.encoder: FeatureEncoder | None = None
        self.businesses_: list[str] = []
        self.model_cost_: HistGradientBoostingRegressor | None = None
        self.model_op_: HistGradientBoostingRegressor | None = None

    def fit(self, frame: pd.DataFrame, cost_col: str, revenue_col: str):
        train = frame.copy()
        train[self.business_col] = train[self.business_col].fillna(UNK).astype(str)
        self.encoder = FeatureEncoder(self.cat_features, self.num_features, self.business_col)
        self.encoder.fit(train)
        X = self.encoder.transform(train)
        y_cost = pd.to_numeric(train[cost_col], errors="coerce").fillna(0.0).to_numpy()
        y_revenue = pd.to_numeric(train[revenue_col], errors="coerce").fillna(0.0).to_numpy()
        y_op = y_revenue - y_cost

        common = dict(
            max_iter=300,
            learning_rate=0.1,
            l2_regularization=1.0,
            min_samples_leaf=20,
            random_state=42,
        )
        self.model_cost_ = HistGradientBoostingRegressor(**common)
        self.model_op_ = HistGradientBoostingRegressor(**common)
        print("  fitting cost model...")
        self.model_cost_.fit(X, y_cost)
        print("  fitting operator model...")
        self.model_op_.fit(X, y_op)
        self.businesses_ = sorted(train[self.business_col].dropna().astype(str).unique())
        return self

    def recommend(self, nodes_df: pd.DataFrame, businesses=None, top_k: int = 3) -> pd.DataFrame:
        if self.encoder is None or self.model_cost_ is None or self.model_op_ is None:
            raise RuntimeError("Model is not fitted.")
        candidate_businesses = [str(b) for b in (businesses or self.businesses_)]
        base = nodes_df[self.cat_features + self.num_features + ["node_id"]].copy()
        business_df = pd.DataFrame({self.business_col: candidate_businesses})
        base["_join_key"] = 1
        business_df["_join_key"] = 1
        expanded = base.merge(business_df, on="_join_key").drop(columns="_join_key")
        X = self.encoder.transform(expanded)
        cost_pred = self.model_cost_.predict(X)
        op_pred = self.model_op_.predict(X)
        scored = pd.DataFrame({
            "node_id": expanded["node_id"].values,
            "business": expanded[self.business_col].values,
            "pred_cost": cost_pred,
            "pred_op": op_pred,
        })

        output = []
        for node_id, group in scored.groupby("node_id", sort=True):
            cost_top = group.sort_values(
                ["pred_cost", "business"], ascending=[False, True]
            ).head(top_k)
            op_top = group.sort_values(
                ["pred_op", "business"], ascending=[False, True]
            ).head(top_k)
            output.append({
                "node_id": node_id,
                "miner_best": str(cost_top.iloc[0]["business"]),
                "miner_cost": float(cost_top.iloc[0]["pred_cost"]),
                "operator_best": str(op_top.iloc[0]["business"]),
                "operator_profit": float(op_top.iloc[0]["pred_op"]),
                "miner_top3": ";".join(
                    f"{row.business}:{row.pred_cost:.1f}"
                    for row in cost_top.itertuples()
                ),
                "operator_top3": ";".join(
                    f"{row.business}:{row.pred_op:.1f}"
                    for row in op_top.itertuples()
                ),
            })
        return pd.DataFrame(output)

    @staticmethod
    def _regret(best_value: float, selected_value: float) -> float:
        denominator = max(abs(float(best_value)), 1e-9)
        return max(0.0, (float(best_value) - float(selected_value)) / denominator)

    @staticmethod
    def _top_businesses(value: pd.Series, top_k: int = 3) -> list[str]:
        return [str(index) for index in value.nlargest(top_k).index.tolist()]

    def evaluate(
        self,
        nodes_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        businesses=None,
        cost_col="cum_cost_7d",
        revenue_col="cum_revenue_7d",
    ):
        candidate_businesses = [str(b) for b in (businesses or self.businesses_)]
        outcomes = outcomes_df.copy()
        outcomes[self.business_col] = outcomes[self.business_col].astype(str)
        outcomes = outcomes[outcomes[self.business_col].isin(candidate_businesses)]
        recommendations = self.recommend(nodes_df, candidate_businesses, top_k=3).set_index("node_id")

        metrics = defaultdict(int)
        miner_regret: list[float] = []
        operator_regret: list[float] = []
        miner_value_capture: list[float] = []
        operator_value_capture: list[float] = []

        for node_id, group in outcomes.groupby("node_id"):
            if node_id not in recommendations.index:
                continue
            group = group.set_index(self.business_col)
            cost_values = pd.to_numeric(group[cost_col], errors="coerce").fillna(0.0)
            op_values = (
                pd.to_numeric(group[revenue_col], errors="coerce").fillna(0.0)
                - cost_values
            )
            true_miner = str(cost_values.idxmax())
            true_operator = str(op_values.idxmax())
            best_cost = float(cost_values.max())
            best_op = float(op_values.max())
            recommendation = recommendations.loc[node_id]

            miner_top3 = str(recommendation["miner_top3"]).split(";")
            operator_top3 = str(recommendation["operator_top3"]).split(";")
            miner_top3_ids = [item.split(":", 1)[0] for item in miner_top3 if ":" in item]
            operator_top3_ids = [item.split(":", 1)[0] for item in operator_top3 if ":" in item]
            metrics["n_nodes"] += 1
            metrics["miner_hit_rate@1"] += recommendation["miner_best"] == true_miner
            metrics["operator_hit_rate@1"] += recommendation["operator_best"] == true_operator
            metrics["miner_hit_rate@3"] += true_miner in miner_top3_ids
            metrics["operator_hit_rate@3"] += true_operator in operator_top3_ids

            miner_business = str(recommendation["miner_best"])
            operator_business = str(recommendation["operator_best"])
            if miner_business in group.index:
                selected_cost = float(cost_values.loc[miner_business])
                miner_regret.append(self._regret(best_cost, selected_cost))
                miner_value_capture.append(selected_cost / best_cost if best_cost > 0 else 1.0)
                metrics["miner_realized_coverage"] += 1
            if operator_business in group.index:
                selected_op = float(op_values.loc[operator_business])
                operator_regret.append(self._regret(best_op, selected_op))
                operator_value_capture.append(
                    selected_op / best_op if best_op > 0 else (1.0 if selected_op == best_op else 0.0)
                )
                metrics["operator_realized_coverage"] += 1

        n_nodes = metrics["n_nodes"]
        return {
            "n_nodes": n_nodes,
            "miner_hit_rate@1": metrics["miner_hit_rate@1"] / n_nodes if n_nodes else 0.0,
            "operator_hit_rate@1": metrics["operator_hit_rate@1"] / n_nodes if n_nodes else 0.0,
            "miner_hit_rate@3": metrics["miner_hit_rate@3"] / n_nodes if n_nodes else 0.0,
            "operator_hit_rate@3": metrics["operator_hit_rate@3"] / n_nodes if n_nodes else 0.0,
            "miner_realized_coverage": metrics["miner_realized_coverage"] / n_nodes if n_nodes else 0.0,
            "operator_realized_coverage": metrics["operator_realized_coverage"] / n_nodes if n_nodes else 0.0,
            "miner_realized_regret": float(np.mean(miner_regret)) if miner_regret else None,
            "operator_realized_regret": float(np.mean(operator_regret)) if operator_regret else None,
            "miner_value_capture": float(np.mean(miner_value_capture)) if miner_value_capture else None,
            "operator_value_capture": float(np.mean(operator_value_capture)) if operator_value_capture else None,
        }
