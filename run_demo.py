"""端到端演示：模拟数据训练 → 回测 → 新节点推荐 → CSV 切换入口。

运行：
    /Users/nany/.workbuddy/binaries/python/envs/default/bin/python run_demo.py
"""
from __future__ import annotations

import csv
import os

from schema import BUSINESSES, NODE_CAT_FEATURES, NODE_NUM_FEATURES
from mock_data import generate
from model import BestBusinessRecommender
from eval import backtest
from real_data import load_from_csv


def main() -> None:
    print("== 1) 生成模拟数据 ==")
    nodes, outcomes = generate(n_nodes=2000, seed=42)
    print(f"节点数={len(nodes)}  观测数(节点×业务)={len(outcomes)}  候选业务={BUSINESSES}")

    print("\n== 2) 训练 + 回测（历史节点留出 20% 验证）==")
    m = backtest(nodes, outcomes, BUSINESSES, test_ratio=0.2, seed=0)
    print(f"测试节点数={m['n']}")
    print(f"  矿主最佳  命中率@1 = {m['miner_hit_rate']:.3f}   遗憾 = {m['miner_regret']:.4f}")
    print(f"  运营最佳  命中率@1 = {m['operator_hit_rate']:.3f}   遗憾 = {m['operator_regret']:.4f}")

    print("\n== 3) 对新节点做推荐（预测最佳业务）==")
    rec = BestBusinessRecommender(BUSINESSES).fit(nodes, outcomes)
    for nid in ["node_000001", "node_000500", "node_001000"]:
        node = next(x for x in nodes if x["node_id"] == nid)
        r = rec.recommend(node)
        print(f"\n节点 {nid}  [{node['deliverytype']}/{node['isp']}/{node['province']}]")
        print(f"  ★ 矿主最佳业务 = {r['miner_best']}   预测7天成本 = {r['miner_cost']:.0f}")
        print(f"  ★ 运营最佳业务 = {r['operator_best']}   预测7天利润 = {r['operator_profit']:.0f}")
        top3 = sorted(r["all"], key=lambda x: -x["profit"])[:3]
        print("  利润 Top3:", [(x["business"], round(x["profit"])) for x in top3])

    print("\n== 4) CSV 切换入口演示（真实数据走这条）==")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    node_fields = ["node_id"] + NODE_CAT_FEATURES + NODE_NUM_FEATURES
    out_fields = ["node_id", "business", "online_time", "cum_cost_7d", "cum_revenue_7d"]
    with open(os.path.join(data_dir, "nodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=node_fields)
        w.writeheader()
        for n in nodes:
            w.writerow({k: n.get(k) for k in node_fields})
    with open(os.path.join(data_dir, "outcomes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for o in outcomes:
            w.writerow({k: o.get(k) for k in out_fields})

    nodes2, outcomes2 = load_from_csv(
        os.path.join(data_dir, "nodes.csv"), os.path.join(data_dir, "outcomes.csv"))
    print(f"  已写出 data/nodes.csv, data/outcomes.csv；CSV 读回 nodes={len(nodes2)} outcomes={len(outcomes2)}")
    print("\n完成。真实数据：把 data/ 下两份 CSV 换成 Superset 导出（或用 load_from_superset 直连）即可。")


if __name__ == "__main__":
    main()
