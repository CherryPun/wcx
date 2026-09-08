# 节点画像 V1 字段字典

## 目标

V1 字段字典服务两个场景：

1. 新交付节点：输入 `node_id` 后形成节点画像，输出 Top3 推荐业务。
2. 存量节点纠偏：识别当前业务与推荐业务是否一致，输出候选调整节点、原因和风险。

当前业务决策口径：

```text
节点一次只交付一个业务
输出 Top3 推荐
综合目标 = 0.5 * 矿主收益分 + 0.5 * 平台利润分
无已知业务硬准入规则
最终由人结合原因与风险决策
```

配套机器可读版本见 [node_profile_v1_fields.json](/Users/rj/Desktop/machine-test/node_profile_v1_fields.json)。

## 字段分层

V1 不把 `multibusiness_nodes.csv` 的 240 个字段全部纳入模型，而是分层使用：

| 分层 | 用途 | 新节点推荐 | 存量纠偏 | 缺失处理 |
|---|---|---|---|---|
| 身份字段 | join、追溯、展示 | 不进模型 | 不进模型 | `node_id` 必填 |
| 共性画像 | 冷启动推荐主特征 | 主模型输入 | 主模型输入 | 类别=unknown，数值=中位数+缺失标记 |
| 网络形态 | 推荐增强、风险解释 | 可选增强 | 可用 | 缺失标记为信息不足 |
| 网络质量/健康风险 | 风险说明、人工审核 | 不做冷启动主特征 | 可用 | 缺失不视为健康 |
| 业务收益/当前业务 | 标签、当前业务识别、纠偏 | 不能作为输入 | 可用 | 按窗口和状态口径处理 |

## 1. 身份字段

| 字段 | 中文名 | 覆盖率 | V1 用途 | 是否进模型 |
|---|---|---:|---|---|
| `node_id` | 节点ID | 100.0% | join key、查询入口 | 否 |
| `online_day` | 上线日/业务首次日 | 100.0% | 历史窗口对齐 | 否 |
| `attribute_day` | 属性快照日 | 100.0% | 防止目标泄漏 | 否 |
| `attribute_snapshot_mode` | 属性快照口径 | 100.0% | 训练口径追溯 | 否 |
| `vendorid` | 供应商/矿主ID | 68.8% | 画像特征、展示 | 是 |

说明：

- `node_id` 只用于查询和关联，不能进入训练。
- `attribute_day` 当前口径为 `online_day - 1 day`，用于避免把业务启动后的状态泄漏进推荐模型。

## 2. 共性画像字段

这些字段是 V1 新节点推荐模型的主干，要求相对稳定、交付时可查、覆盖率相对较好。

| 字段 | 中文名 | 覆盖率 | 类型 | 缺失处理 |
|---|---|---:|---|---|
| `province` | 省份 | 69.0% | 类别 | unknown |
| `city` | 城市 | 69.0% | 类别 | unknown |
| `isp` | 运营商 | 69.0% | 类别 | unknown |
| `resourcetype` | 资源类型 | 69.1% | 类别 | unknown |
| `deliverytype` | 交付类型 | 68.8% | 类别 | unknown |
| `device_type` | 设备类型 | 68.9% | 类别 | unknown |
| `node_manufacturer` | 设备厂商 | 67.2% | 类别 | unknown |
| `node_model` | 设备型号 | 68.3% | 类别 | unknown |
| `isvm` | 是否虚机 | 69.1% | 类别/布尔 | unknown |
| `bw` | 名义带宽 | 68.0% | 数值 | 中位数+缺失标记 |
| `actualbandwidth` | 实测带宽 | 69.2% | 数值 | 中位数+缺失标记 |
| `corenum` | CPU 核数 | 69.2% | 数值 | 中位数+缺失标记 |
| `memtotal` | 内存总量 | 69.2% | 数值 | 中位数+缺失标记 |
| `totaldisksize` | 磁盘总量 | 69.2% | 数值 | 中位数+缺失标记 |
| `hdddisksize` | HDD 容量 | 69.2% | 数值 | 中位数+缺失标记 |
| `ssddisksize` | SSD 容量 | 69.2% | 数值 | 中位数+缺失标记 |
| `systemdisksize` | 系统盘容量 | 69.2% | 数值 | 中位数+缺失标记 |

V1 训练建议：

```text
类别字段：OneHot 或低频合并/频率编码
数值字段：只用训练集统计量填补，并增加 missing flag
```

注意：

- `actualbandwidth` 覆盖率高，但需要确认它在“新交付节点推荐”时是否已经可查。如果交付时不可查，则新节点模型降级使用 `bw`。
- `node_manufacturer`、`node_model` 需要后续做大小写和同义归一，比如 `Inspur` 与 `INSPUR`。

## 3. 网络形态字段

这些字段对业务承接能力和风险很重要，但当前覆盖率偏低，V1 先作为增强字段和解释字段，不作为硬准入规则。

| 字段 | 中文名 | 覆盖率 | V1 用途 | 缺失处理 |
|---|---|---:|---|---|
| `nattype` | NAT 类型 | 24.7% | 可选模型增强、风险解释 | unknown + 信息不足 |
| `dialtype` | 拨号类型 | 24.7% | 可选模型增强、风险解释 | unknown + 信息不足 |
| `dial_is_normal` | 拨号是否正常 | 23.4% | 风险解释、存量审核 | unknown，不判健康 |
| `dial_ipv6_enable` | 拨号 IPv6 是否开启 | 23.4% | 可选增强、展示 | unknown |
| `dial_on_physical_nic` | 是否物理网卡拨号 | 23.3% | 风险解释 | unknown |
| `dial_account_count` | 拨号账号数 | 22.4% | 风险解释、存量审核 | 中位数+缺失标记 |
| `scheduleisps` | 调度运营商 | 当前大节点约 44.0% | 可选模型增强、风险解释 | unknown + 信息不足 |
| `scheduleisps_text` | 首个调度目标运营商 | 当前大节点约 44.0% | 判断本网/异网并拆分目标运营商 | 空列表按本网；非空只取原始顺序第一个 |
| `analysis_transprovrate` | 跨省调度比例 | 当前大节点 100.0% | 可选模型增强、条件分段 | 中位数+缺失标记 |
| `network_schedule_type` | 网络调度类型 | 派生字段 | 本网本省/本网出省/异网本省/异网出省 | 中间比例保留为混合省份 |
| `join_isbantransprov` | 是否禁止跨省 | 当前大节点约 30.3% | 可选模型增强、风险解释 | true 重点提示，空值不当作正常 |
| `join_isipv6schedule` | 是否 IPv6 调度 | 当前大节点约 0.7% | 可选模型增强、展示 | unknown |

V1 原则：

```text
缺失不能当作正常；
有异常可以提示风险；
没有业务准入文档前，不用这些字段做硬性业务阻断。
```

## 4. 网络质量与健康风险字段

这些字段多数是运行期观测，不适合作为新节点冷启动主特征，但适合用于存量节点风险说明和人工审核。

| 字段 | 中文名 | 覆盖率 | 风险方向 | V1 用途 |
|---|---|---:|---|---|
| `quality_retransrate` | 质量重传率 | 18.7% | 越高越差 | 风险说明、存量纠偏 |
| `quality_pinglossrate` | 质量丢包率 | 13.6% | 越高越差 | 风险说明、存量纠偏 |
| `quality_rtt` | 质量 RTT | 20.0% | 越高越差 | 风险说明、存量纠偏 |
| `join_net_tcpretransrate` | node_join TCP 重传率 | 2.6% | 越高越差 | 展示/备用 |
| `join_net_pinglossrate` | node_join 丢包率 | 0.9% | 越高越差 | 展示/备用 |
| `join_net_pingrtt` | node_join RTT | 2.6% | 越高越差 | 展示/备用 |
| `prom_retrans_ratio` | Prometheus 重传率 | 8.7% | 越高越差 | 实时风险说明 |
| `cpu_load1_per_core` | 单核 CPU 负载 | 23.7% | 越高越差 | 运行风险 |
| `mem_used_ratio` | 内存使用率 | 23.7% | 越高越差 | 运行风险 |
| `disk_used_ratio` | 磁盘使用率 | 23.3% | 越高越差 | 运行风险 |
| `maxioutil` | 最大磁盘 IO 利用率 | 17.9% | 越高越差 | 运行风险 |
| `smart_available_spare` | SMART 可用 spare | 0.4% | 越低越差 | 有值展示 |
| `smart_bad_block_count` | SMART 坏块数 | 0.0% | 越高越差 | 有值展示 |
| `smart_case_temperature` | SMART 温度 | 0.4% | 越高越差 | 有值展示 |
| `smart_critical_warning` | SMART critical warning | 0.4% | 越高越差 | 有值展示 |
| `zfs_pool_online` | ZFS 池在线 | 0.0% | 0 更差 | 当前不可用 |

V1 原则：

```text
风险字段不参与新节点冷启动主推荐；
风险字段缺失时输出“信息不足”，不输出“健康”；
SMART/ZFS 覆盖太低，不做 V1 硬规则。
```

## 5. 业务收益与当前业务字段

这些字段来自 `node_day_ops_wide_full` 或当前项目聚合产物，用于训练标签、当前业务识别和存量纠偏，不能作为新节点推荐输入。

| 字段 | 中文名 | 来源 | 用途 |
|---|---|---|---|
| `business` / `customerId` | 业务ID | `node_day_ops_wide_full.customerId` | 候选业务、当前业务 |
| `business_name` / `customerName` | 业务名称 | `node_day_ops_wide_full.customerName` | 展示 |
| `day` | 日期 | `node_day_ops_wide_full.day` | 当前业务窗口 |
| `state` | 在线状态 | `node_day_ops_wide_full.state` | 当前业务推断 |
| `peak95` | 日95带宽 | `node_day_ops_wide_full.peak95` | 当前业务强度/排序 |
| `cost_finalAmount` / `cum_cost_7d` | 成本/7天累计成本 | 宽表聚合 | 矿主目标 |
| `revenue_finalAmount` / `cum_revenue_7d` | 收入/7天累计收入 | 宽表聚合 | 平台利润目标 |
| `cum_profit_7d` | 7天累计利润 | `revenue - cost` | 运营目标 |
| `outcome_distinct_days` | 7天窗口有记录天数 | 宽表聚合 | 观测充分性 |

当前业务 V1 推断建议：

```text
优先使用最近 3 天 state='online' 的业务；
若多个业务并存，按出现天数、peak95、revenue_finalAmount 依次排序；
输出 current_business_confidence，低置信度交给人工判断。
```

## V1 输出结构

单节点查询建议输出：

```text
基础画像：身份、地域、运营商、资源类型、设备、硬件容量
网络能力：带宽、NAT、拨号、IPv6
网络质量：RTT、丢包、重传、CPU/内存/磁盘风险
推荐结果：综合 Top3、矿主 Top3、运营 Top3
当前业务：当前业务ID/名称、推断置信度
纠偏判断：当前业务是否等于 Top1，是否在 Top3
风险说明：缺失字段、异常字段、健康风险
```

## 后续待确认

1. `actualbandwidth` 在新节点交付时是否一定可查。
2. 当前业务是否以 `node_day_ops_wide_full` 最近 3 天推断为准，还是有更权威的部署/绑定系统。
3. `node_join` 是否能稳定提供 NAT/拨号字段，当前低覆盖是否因为采集口径还是样本问题。
4. 业务侧是否后续会补充硬准入规则，如省份、运营商、NAT、带宽限制。
