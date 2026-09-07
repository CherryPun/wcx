# 数据字段与来源说明

> 本目录存放主版与诊断共用的成品数据。大文件默认不入库（见根 `.gitignore`），
> 复现前请按“README：数据来源与重建”重新生成，或从内网环境拷贝本目录。

## 1. nodes_attr_filled.csv —— 节点画像（主版/诊断共同输入）

- 来源：jarvis 节点画像 + 宽表上线信息；本机对缺失字段用 node_join/HRR 快照回填后产出。
- 每行一个节点。关键列：

| 列 | 含义 | 备注 |
|---|---|---|
| node_id | 节点 ID | 关联主键 |
| vendorid | 厂商/供应方 ID | 类别 |
| deliverytype | 交付类型 | 如 idc / dedicated / aggregation |
| resourcetype | 资源类型 | 如 dedicated / aggregation / 小盒子 |
| dialtype / nattype | 拨号 / NAT 类型 | 类别 |
| province / isp / city | 省份 / 运营商 / 城市 | isp 需对齐电信/联通/移动口径 |
| device_type / os / arch | 设备 / 系统 / 架构 | 类别 |
| hardwaretype / node_manufacturer / node_model | 硬件 | 类别，覆盖率不完整 |
| idc_id | 机房 ID | 类别 |
| bw | 名义带宽（Mbps） | 数值；缺失率高时模型按缺失处理 |
| corenum / memtotal | CPU 核数 / 内存总量 | 数值（原始单位） |
| totaldisksize / hdddisksize / ssddisksize / systemdisksize | 磁盘容量 | 数值（原始单位） |
| join_cpu_corenumber / join_memtotal | node_join 补充的 CPU/内存 | 与上面冗余，用于回填 |

## 2. outcomes_7d_named.csv —— 节点×业务 7 天账（金额标签）

- 来源：Superset 宽表，按业务首次出现日起 7 个自然日合计；本机先保留满 7 天行。
- 每行一个 (node_id, business) 的 7 天累计。关键列：

| 列 | 含义 |
|---|---|
| node_id | 节点 ID |
| business | 业务 ID（customerId 口径） |
| business_name | 业务中文名（已用名称表补齐） |
| business_online_day | 该业务在该节点的首次出现日 |
| online_day | 节点最早业务上线日（用于时间外切分） |
| business_count | 该节点历史业务数（统计用） |
| cum_cost_7d | 7 天累计 cost_finalAmount（矿主到手，模型主目标） |
| cum_revenue_7d | 7 天累计 revenue_finalAmount |
| outcome_days | 实际有数据天数（满 7 天才进模型） |

## 3. business_capacity_summary.csv —— 业务容量推演（软约束用）

- 来源：近 7 天日 peak95 流量 + 目标利用率 70% 反推容量上限。
- 每行一个业务。关键列：

| 列 | 含义 |
|---|---|
| business / business_name | 业务 ID / 名称 |
| days | 有效天数 |
| sample_nodes | 该业务覆盖节点数 |
| traffic_p75_mbps | 近 7 天日 95 流量的 P75（Mbps） |
| capacity_upper_mbps | 容量上限 ≈ traffic_p75 / 0.7 |
| util_at_capacity | 目标利用率（0.7） |
| current_daily_peak95_sum_mbps | 当前日峰值合计（近似现网负载） |
| can_add_mbps / over_capacity_mbps | 仍可新增 / 超容量（软约束降权依据） |

## 4. bw_daily_7d.csv —— 近 7 天日 95 带宽

- 来源：Superset 宽表近 7 天按 node×business 的 peak95。
- 每行：node_id, day, business, business_name, peak95(bps), cost/revenue, state。
- 用途：带宽辅助分（该节点跑该业务时的利用率），以及容量推演流量来源。

## 口径提醒

- 金额“矿主结算”= cost_finalAmount；利润 = revenue − cost（平台差价）。
- 历史覆盖：2026-03-16 起，约 5% 节点抽样（后缀 00–0c），满 7 天账。
- 与 GitHub 三分支数据不同源、不同窗口，不可混用。
