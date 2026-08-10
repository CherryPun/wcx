"""回测评估：用历史节点验证推荐效果。

指标：
- 命中率@1（hit-rate@1）：推荐的业务是否等于该节点真实最优业务。
- 遗憾（regret）：(真实最优值 - 预测选中值) / 真实最优值，越小越好（0=完美）。
  矿主用成本、运营用利润计算 regret。
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict

from model import BestBusinessRecommender


def backtest(nodes, outcomes, businesses, test_ratio: float = 0.2, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = np.array([n["node_id"] for n in nodes])
    n_test = max(1, int(len(ids) * test_ratio))
    test_ids = set(rng.choice(ids, size=n_test, replace=False))

    train_nodes = [n for n in nodes if n["node_id"] not in test_ids]
    train_out = [o for o in outcomes if o["node_id"] not in test_ids]

    rec = BestBusinessRecommender(businesses).fit(train_nodes, train_out)

    by_node = defaultdict(list)
    for o in outcomes:
        if o["node_id"] in test_ids:
            by_node[o["node_id"]].append(o)

    miner_hit = op_hit = 0
    miner_regret = op_regret = 0.0
    n = 0
    for node in nodes:
        if node["node_id"] not in test_ids:
            continue
        obs = by_node.get(node["node_id"], [])
        if not obs:
            continue
        true_miner = max(obs, key=lambda o: o["cum_cost_7d"])["business"]
        true_op = max(obs, key=lambda o: o["cum_revenue_7d"] - o["cum_cost_7d"])["business"]
        true_miner_val = max(o["cum_cost_7d"] for o in obs)
        true_op_val = max(o["cum_revenue_7d"] - o["cum_cost_7d"] for o in obs)

        pred = rec.recommend(node)
        if pred["miner_best"] == true_miner:
            miner_hit += 1
        if pred["operator_best"] == true_op:
            op_hit += 1
        pm = next(x for x in pred["all"] if x["business"] == pred["miner_best"])
        po = next(x for x in pred["all"] if x["business"] == pred["operator_best"])
        miner_regret += (true_miner_val - pm["cost"]) / true_miner_val if true_miner_val else 0.0
        op_regret += (true_op_val - po["profit"]) / true_op_val if true_op_val else 0.0
        n += 1

    return {
        "n": n,
        "miner_hit_rate": miner_hit / n,
        "operator_hit_rate": op_hit / n,
        "miner_regret": miner_regret / n,
        "operator_regret": op_regret / n,
    }
