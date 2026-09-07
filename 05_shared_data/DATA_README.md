# 数据字段与来源说明

> 本目录存放主版与诊断共用的成品数据。大文件默认不入库（见根 `.gitignore`），
> 复现前请按根 README“数据来源与重建”重新生成，或从可访问内网 Superset 的环境拷贝本目录。

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

## 5. cur7d_map.csv —— “当前在跑”列映射（HTML 可视化对照用，非模型输入）

- 用途：`build_final_html.py` 读取，在 `全节点推荐_最终版.html` 的 `node_id` 与 `Top1` 之间渲染“当前在跑（近7天实际）”列。
- 来源：结算宽表 `node_day_ops_wide_full`（数据库 yzh-starrocks / schema test）近 30 日（08-08~09-06）按 5,171 个物理节点分批导出后本地判定。
- 判定口径：物理节点近 30 日内取“最近一段连续满 7 个有效结算日（当日有矿主成本或收入）”，且窗口内必须存在实际收入才填；
  业务名 = 窗口内累计结算最高的业务（收入挂在 customerId=0 的无名行按其同窗口成本最高的有名业务归并）；
  金额为该代表业务窗口 7 天的 `cost_finalAmount`（矿主结算）与平台利润（账上 `profit_profitAmount`，无则用 收入−成本）。
- 不满足（宽表无记录 / 有效结算不足 7 天 / 满 7 天但仅成本保底）则整列置空（HTML 显示 —），业务名与金额同有同空。
- 覆盖率（本批）：可填 1,726 / 空 3,445（无记录 2,206、不足 7 天 1,203、仅成本 36）。
- 本文件与原始导出不入库（`05_shared_data/*.csv` 被 `.gitignore` 排除），重跑需按上表口径从宽表重新生成。

## 口径提醒

- 金额“矿主结算”= cost_finalAmount；利润 = revenue − cost（平台差价）。
- 历史覆盖：2026-03-16 起，约 5% 节点抽样（后缀 00–0c），满 7 天账。
