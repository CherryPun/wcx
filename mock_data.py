"""模拟数据生成：构造带可学习信号的 (节点属性, 业务) → 7天成本/收入。
真实环境用 real_data.load_from_csv / load_from_superset 替换本模块。
"""
from __future__ import annotations

import numpy as np

from schema import NODE_CAT_FEATURES, NODE_NUM_FEATURES, BUSINESSES

_VENDOR = [77882, 10001, 20002, 30003]
_DELIVERY = ["idc", "pcdn", "egress"]
_RESTYPE = ["server", "box", "vm"]
_DIAL = ["staticNetSingle", "pppoe", "dhcp"]
_NAT = ["public", "fullcone", "symmetric"]
_SCHED = ["telecom", "unicom", "mobile", "multi"]
_REG = ["self", "agent"]
_CUSTOMER = ["direct", "resell"]
_PROV = ["广东", "湖北", "北京", "四川", "浙江"]
_ISP = ["电信", "联通", "移动"]
_CITY = ["广州", "武汉", "北京", "成都", "杭州"]
_DEVICE = ["jarvis.A", "jarvis.B", "jarvis.C"]
_ARCH = ["host", "vm"]
_ISVM = ["true", "false"]
_QOS = ["on", "off"]

_CAT_POOL = {
    "vendorid": _VENDOR, "deliverytype": _DELIVERY, "resourcetype": _RESTYPE,
    "dialtype": _DIAL, "nattype": _NAT, "scheduleisps": _SCHED, "regsource": _REG,
    "customermode": _CUSTOMER, "province": _PROV, "isp": _ISP, "city": _CITY,
    "device_type": _DEVICE, "arch_type": _ARCH, "isvm": _ISVM, "qoskiller_status": _QOS,
}

# 业务对成本/收入的相对权重（制造"矿主最佳"与"运营最佳"通常不同）
_COST_BIZ = {"kuaishou": 1.20, "douyin": 1.10, "tencent_video": 1.00, "iqiyi": 0.90,
             "bilibili": 1.05, "netease": 0.80, "default": 0.70}
_REV_BIZ = {"kuaishou": 1.00, "douyin": 1.15, "tencent_video": 1.25, "iqiyi": 1.10,
            "bilibili": 1.00, "netease": 0.85, "default": 0.60}
_DELIVERY_COST = {"idc": 1000, "pcdn": 600, "egress": 800}
_DELIVERY_REV = {"idc": 1500, "pcdn": 900, "egress": 1100}


def _true_cost(n: dict, b: str) -> float:
    base = _DELIVERY_COST[n["deliverytype"]]
    isp_boost = 1.15 if (n["isp"] == "电信" and b in ("kuaishou", "douyin")) else 1.0
    util = 0.8 + 0.4 * n["sevendayavg95ratio"]
    return base * _COST_BIZ[b] * isp_boost * util


def _true_rev(n: dict, b: str) -> float:
    base = _DELIVERY_REV[n["deliverytype"]]
    prov_boost = 1.20 if (n["province"] in ("广东", "北京") and b in ("tencent_video", "iqiyi")) else 1.0
    util = 0.7 + 0.6 * n["sevendayavg95ratio"]
    return base * _REV_BIZ[b] * prov_boost * util


def generate(n_nodes: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    nodes: list[dict] = []
    outcomes: list[dict] = []
    for i in range(n_nodes):
        nid = f"node_{i:06d}"
        n = {"node_id": nid}
        for c in NODE_CAT_FEATURES:
            n[c] = str(rng.choice(_CAT_POOL[c]))
        n["sevendayavg95ratio"] = round(float(rng.uniform(0.1, 1.0)), 4)
        nodes.append(n)
        online = int(rng.integers(0, 30 * 24 * 3600))  # 上线时间（占位，秒偏移）
        for b in BUSINESSES:
            c = _true_cost(n, b) * float(rng.normal(1.0, 0.05))
            r = _true_rev(n, b) * float(rng.normal(1.0, 0.05))
            outcomes.append({
                "node_id": nid, "business": b, "online_time": online,
                "cum_cost_7d": round(float(c), 2), "cum_revenue_7d": round(float(r), 2),
            })
    return nodes, outcomes
