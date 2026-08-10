#!/usr/bin/env python3
"""真实数据训练、时间回测和节点业务推荐。"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import BestBusinessRecommender
from schema import (
    BUSINESS_COL,
    NODE_CAT_FEATURES,
    NODE_NUM_FEATURES,
    ONLINE_DAY_COL,
    TARGET_COST,
    TARGET_REVENUE,
)


MIN_SUPPORT = int(os.environ.get("MIN_BUSINESS_SUPPORT", "50"))
TOP_K = 3


def normalize_days(frame: pd.DataFrame, preferred: str) -> pd.Series:
    if preferred in frame.columns:
        return pd.to_datetime(frame[preferred], errors="coerce").dt.normalize()
    if "online_time" in frame.columns:
        return pd.to_datetime(frame["online_time"], errors="coerce").dt.normalize()
    raise ValueError(f"Missing online day column: {preferred}")


def baseline_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame, business: str) -> dict:
    miner_regret = []
    operator_regret = []
    miner_coverage = 0
    operator_coverage = 0
    n_nodes = 0
    miner_hit = 0
    operator_hit = 0

    for _node_id, group in test_df.groupby("node_id"):
        if group.empty:
            continue
        n_nodes += 1
        group = group.set_index(BUSINESS_COL)
        costs = group[TARGET_COST].astype(float)
        ops = group[TARGET_REVENUE].astype(float) - costs
        true_miner_business = str(costs.idxmax())
        true_operator_business = str(ops.idxmax())
        if business == true_miner_business:
            miner_hit += 1
        if business == true_operator_business:
            operator_hit += 1
        if business in group.index:
            miner_coverage += 1
            selected_cost = float(costs.loc[business])
            miner_regret.append(max(0.0, (float(costs.max()) - selected_cost) / max(abs(float(costs.max())), 1e-9)))
            if float(costs.max()) > 0:
                pass
        if business in group.index:
            operator_coverage += 1
            selected_op = float(ops.loc[business])
            best_op = float(ops.max())
            operator_regret.append(max(0.0, (best_op - selected_op) / max(abs(best_op), 1e-9)))

    return {
        "baseline_business": str(business),
        "n_nodes": n_nodes,
        "miner_hit_rate@1": miner_hit / n_nodes if n_nodes else 0.0,
        "operator_hit_rate@1": operator_hit / n_nodes if n_nodes else 0.0,
        "miner_realized_coverage": miner_coverage / n_nodes if n_nodes else 0.0,
        "operator_realized_coverage": operator_coverage / n_nodes if n_nodes else 0.0,
        "miner_realized_regret": float(np.mean(miner_regret)) if miner_regret else None,
        "operator_realized_regret": float(np.mean(operator_regret)) if operator_regret else None,
    }


def main() -> None:
    started = time.time()
    outcomes = pd.read_csv(os.path.join(HERE, "outcomes_raw.csv"))
    nodes = pd.read_csv(os.path.join(HERE, "nodes.csv"), low_memory=False)
    outcomes[BUSINESS_COL] = outcomes[BUSINESS_COL].astype(str)
    for column in (TARGET_COST, TARGET_REVENUE):
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce").fillna(0.0)
    outcomes[ONLINE_DAY_COL] = normalize_days(outcomes, ONLINE_DAY_COL)

    if "online_day" in nodes.columns:
        nodes["online_day"] = pd.to_datetime(nodes["online_day"], errors="coerce").dt.normalize()
    print(f"outcomes: {len(outcomes):,} rows, nodes: {len(nodes):,}")

    node_metadata = nodes.drop(columns=["online_day"], errors="ignore")
    model_df = outcomes.merge(node_metadata, on="node_id", how="inner")
    model_df = model_df.dropna(subset=[ONLINE_DAY_COL]).copy()
    print(
        f"merged model_df: {len(model_df):,} rows, "
        f"{model_df['node_id'].nunique():,} nodes, "
        f"{model_df[BUSINESS_COL].nunique()} businesses"
    )

    days = sorted(model_df[ONLINE_DAY_COL].dropna().unique())
    if len(days) < 2:
        raise RuntimeError("At least two distinct online days are required for a time split.")
    split_index = min(max(int(len(days) * 0.8), 1), len(days) - 1)
    split_day = days[split_index]
    train_all = model_df[model_df[ONLINE_DAY_COL] < split_day].copy()
    test_all = model_df[model_df[ONLINE_DAY_COL] >= split_day].copy()

    candidate_businesses = (
        train_all[BUSINESS_COL]
        .value_counts()
        .loc[lambda counts: counts >= MIN_SUPPORT]
        .index.astype(str)
        .tolist()
    )
    if not candidate_businesses:
        raise RuntimeError("No candidate business meets the configured support threshold.")
    train_df = train_all[train_all[BUSINESS_COL].isin(candidate_businesses)].copy()
    test_outcomes = test_all[test_all[BUSINESS_COL].isin(candidate_businesses)].copy()
    test_node_ids = set(test_outcomes["node_id"])
    test_nodes = nodes[nodes["node_id"].isin(test_node_ids)].reset_index(drop=True)

    print(f"time split: train < {split_day.date()}, test >= {split_day.date()}")
    print(f"train nodes: {train_df['node_id'].nunique():,}; test nodes: {len(test_node_ids):,}")
    print(f"candidate businesses (support >= {MIN_SUPPORT}): {len(candidate_businesses)}")

    recommender = BestBusinessRecommender(NODE_CAT_FEATURES, NODE_NUM_FEATURES, BUSINESS_COL)
    recommender.fit(train_df, TARGET_COST, TARGET_REVENUE)
    metrics = recommender.evaluate(test_nodes, test_outcomes, businesses=candidate_businesses)

    train_best = (
        train_df.loc[train_df.groupby("node_id")[TARGET_COST].idxmax(), BUSINESS_COL]
        .value_counts()
        .idxmax()
    )
    metrics["baseline"] = baseline_metrics(test_all[test_all[BUSINESS_COL].isin(candidate_businesses)], test_outcomes, train_best)
    metrics["split_day"] = str(split_day.date())
    metrics["min_business_support"] = MIN_SUPPORT

    recommendations = recommender.recommend(test_nodes, businesses=candidate_businesses, top_k=TOP_K)
    metadata_columns = [column for column in ["node_id", "online_day", "attribute_snapshot_day", "attribute_snapshot_hour"] if column in nodes.columns]
    recommendations = recommendations.merge(nodes[metadata_columns], on="node_id", how="left")
    recommendations.to_csv(os.path.join(HERE, "recommendations_real.csv"), index=False)

    source_counts = {
        str(key): int(value)
        for key, value in outcomes.get("online_source", pd.Series(dtype=str)).value_counts(dropna=False).items()
    }
    attribute_snapshot = pd.to_datetime(nodes.get("attribute_snapshot_day"), errors="coerce")
    node_online_days = pd.to_datetime(nodes.get("online_day"), errors="coerce")
    snapshot_violations = int((attribute_snapshot > node_online_days).fillna(False).sum())
    node_value_summary = outcomes.groupby("node_id")[[TARGET_COST, TARGET_REVENUE]].sum()
    diagnostics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": "[online_day, online_day + 6]",
        "online_source_counts": source_counts,
        "training_split": f"online_day < {split_day.date()} vs >= {split_day.date()}",
        "sample": "5% deterministic node sample",
        "outcome_rows": int(len(outcomes)),
        "merged_rows": int(len(model_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_outcomes)),
        "train_nodes": int(train_df["node_id"].nunique()),
        "test_nodes": int(test_outcomes["node_id"].nunique()),
        "n_businesses_total": int(model_df[BUSINESS_COL].nunique()),
        "n_candidate_businesses": len(candidate_businesses),
        "attribute_nodes": int(nodes["node_id"].nunique()),
        "attribute_match_rate": round(nodes["node_id"].nunique() / max(outcomes["node_id"].nunique(), 1), 6),
        "attribute_snapshot_violations": snapshot_violations,
        "attribute_missing_feature_cells": int(nodes[NODE_CAT_FEATURES + NODE_NUM_FEATURES].isna().sum().sum()),
        "zero_outcome_rows": int(((outcomes[TARGET_COST] == 0) & (outcomes[TARGET_REVENUE] == 0)).sum()),
        "zero_outcome_nodes": int(
            ((node_value_summary[TARGET_COST] == 0) & (node_value_summary[TARGET_REVENUE] == 0)).sum()
        ),
        "features": NODE_CAT_FEATURES + NODE_NUM_FEATURES,
        "target_miner": TARGET_COST,
        "target_operator": f"({TARGET_REVENUE} - {TARGET_COST})",
        "attribute_snapshot_rule": "latest node_join snapshot not later than online_day",
        "realized_regret_scope": "conditional on recommended business being observed for the test node",
        "elapsed_sec": round(time.time() - started, 1),
    }
    with open(os.path.join(HERE, "metrics_real.json"), "w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics, "diagnostics": diagnostics}, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, "model_diag.txt"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    print(json.dumps({"metrics": metrics, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))
    print(f"recommendations written: {len(recommendations):,} nodes")


if __name__ == "__main__":
    main()
