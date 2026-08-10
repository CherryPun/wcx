"""真实数据接入层。

当前用模拟数据跑通整条链路；拿到 Superset 凭据后，把真实表字段映射到本模块的
数据契约（nodes / outcomes）即可切换。两条入口：
- load_from_csv(nodes_csv, outcomes_csv)：你导出 CSV 给我时走这条。
- load_from_superset(...)：直接连 Superset SQL Lab 拉数（依赖已补齐的 common/superset_api.py）。

数据契约
--------
nodes  : [{node_id, vendorid, deliverytype, resourcetype, dialtype, nattype,
          scheduleisps, regsource, customermode, province, isp, city,
          device_type, arch_type, isvm, qoskiller_status, sevendayavg95ratio}, ...]
outcomes: [{node_id, business, online_time, cum_cost_7d, cum_revenue_7d}, ...]
  - online_time：节点上线时间（按你的口径 = 首次出现时间）
  - cum_cost_7d / cum_revenue_7d：上线后 7 天窗口内的累积成本 / 收入金额
"""
from __future__ import annotations

import csv
import os
import sys

from schema import NODE_CAT_FEATURES, NODE_NUM_FEATURES

# 真实环境需确认的结算表/字段（文档里只看到 business.billing 计费配置，未含累积金额）。
# 下面用 jarvis.node_business_settlement 作占位表名，拿到真实表后替换即可。
NODE_ATTR_SQL = """
SELECT
  _id                                                   AS node_id,
  nodestaticinfo.vendorid                               AS vendorid,
  nodestaticinfo.nominalinfo.deliverytype               AS deliverytype,
  nodestaticinfo.nominalinfo.resourcetype               AS resourcetype,
  nodestaticinfo.nominalinfo.dialtype                   AS dialtype,
  nodestaticinfo.nominalinfo.nattype                    AS nattype,
  nodestaticinfo.nominalinfo.scheduleisps               AS scheduleisps,
  nodestaticinfo.regsource                              AS regsource,
  nodestaticinfo.customermode                           AS customermode,
  province, isp, city,
  nodestaticinfo.nominalinfo.devicetype                AS device_type,
  nodestaticinfo.nominalinfo.archtype                  AS arch_type,
  nodestaticinfo.nominalinfo.isvm                       AS isvm,
  nodestaticinfo.nodeqoskillerconfig.status            AS qoskiller_status,
  nodeinfo.sevendayavg95ratio                           AS sevendayavg95ratio
FROM jarvis.node_join
WHERE day = {day}
"""

OUTCOME_SQL = """
-- 上线时间 = 节点首次出现时间；7天窗口 = [首次出现, +7d]
-- 累积成本/收入为结算表按 (node_id, business) 在窗口内聚合
SELECT
  node_id,
  business,
  online_time,
  cum_cost_7d,
  cum_revenue_7d
FROM jarvis.node_business_settlement   -- TODO: 替换为真实结算表
WHERE day = {day}
"""


def load_from_csv(nodes_csv: str, outcomes_csv: str):
    nodes, outcomes = [], []
    with open(nodes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["sevendayavg95ratio"] = float(row.get("sevendayavg95ratio", 0) or 0)
            nodes.append(row)
    with open(outcomes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["cum_cost_7d"] = float(row["cum_cost_7d"])
            row["cum_revenue_7d"] = float(row["cum_revenue_7d"])
            row["online_time"] = int(row.get("online_time", 0) or 0)
            outcomes.append(row)
    return nodes, outcomes


def load_from_superset(base_url: str, username: str, password: str,
                       day: int, database_id: str | None = None,
                       superset_common_dir: str | None = None):
    """直连 Superset SQL Lab 拉数。需要 common/superset_api.py（已补齐）。"""
    common_dir = superset_common_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "Downloads", "superset-sql-query-skill", "common")
    sys.path.insert(0, os.path.abspath(common_dir))
    from superset_api import SupersetSQLClient  # type: ignore

    client = SupersetSQLClient(username=username, password=password,
                               base_url=base_url, database_id=database_id)
    node_rows = client.execute_sql(sql=NODE_ATTR_SQL.format(day=day))
    out_rows = client.execute_sql(sql=OUTCOME_SQL.format(day=day))

    nodes = []
    for r in node_rows:
        r["sevendayavg95ratio"] = float(r.get("sevendayavg95ratio") or 0)
        nodes.append(r)
    outcomes = []
    for r in out_rows:
        r["cum_cost_7d"] = float(r["cum_cost_7d"])
        r["cum_revenue_7d"] = float(r["cum_revenue_7d"])
        r["online_time"] = int(r.get("online_time") or 0)
        outcomes.append(r)
    return nodes, outcomes
