"""最佳业务推荐的数据契约与特征定义。"""

# 属性来自节点上线日当天的 jarvis.node_join 最新小时快照。
NODE_CAT_FEATURES = [
    "vendorid",
    "deliverytype",
    "resourcetype",
    "dialtype",
    "nattype",
    "scheduleisps",
    "regsource",
    "customermode",
    "province",
    "isp",
    "city",
    "device_type",
    "arch_type",
    "isvm",
    "qoskiller_status",
]

# 上线时可知的静态带宽，不使用上线后 7 天统计值。
NODE_NUM_FEATURES = ["bw"]

BUSINESS_COL = "business"
TARGET_COST = "cum_cost_7d"
TARGET_REVENUE = "cum_revenue_7d"
ONLINE_DAY_COL = "online_day"
ATTRIBUTE_DAY_COL = "attribute_snapshot_day"
