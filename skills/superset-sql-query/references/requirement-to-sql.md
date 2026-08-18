# Requirement To SQL

## 目标

当用户不给 SQL，而是直接给业务需求时，按固定流程把需求翻译成可执行的 Superset SQL，并返回结果。

## 固定流程

1. 明确查询对象

- 节点当前/历史事实优先查 `jarvis.node_analysis_data`
- 静态画像、运行画像、7 天汇总优先查 `jarvis.node_join`
- 大盘切片、维度分布、聚合画像优先查 `jarvis.node_analysis_summary`
- 拨号账号、接口、拨号状态、连通性问题优先查 `jarvis.dial_acc`
- 两类信息都要时，先以 `node_analysis_data` 筛节点，再补 `node_join`

补充判断：

- 要查“某个节点 / 某批节点”时，优先从 `node_analysis_data` 开始
- 要查“某类节点整体分布 / 大盘画像”时，优先考虑 `node_analysis_summary`
- 要查“节点为什么在线但业务异常、是不是拨号有问题”时，优先考虑 `dial_acc`
- 要查“控制面怎么看这个节点、是否需要重部署、QoS 是否生效”时，补 `node_join`

2. 明确时间范围

- 没给时间时，默认先问“今天 / 昨天 / 本周 / 最近 7 天”中的一个明确口径
- `node_analysis_data.day` 用 `'YYYYMMDD'`
- `node_join.day` 用 `YYYYMMDD`

3. 明确筛选维度

常见维度：

- `nodeid`
- `customersummary`
- `purchasername` / `purchaserid`
- `vendorid`
- `province` / `city` / `isp`
- `state`
- `tags`

4. 先做最小可验证查询

- 先 `LIMIT 5/10`
- 先查字段和样本，确认口径没跑偏
- 再放大到正式导出

5. 正式查询时保留可解释字段

不要只返回 `nodeid`，通常至少带：

- 时间字段
- 业务字段
- 状态字段
- 地域字段
- 用户关心的指标字段

6. 结果交付

- 小结果：直接总结 + 列重点行
- 大结果：落 `CSV/JSON`
- 需要持续复用：把 SQL 落到文件

## 默认规则

### 1. 业务切换

如果用户说“从 A 切到 B”：

- 先按天去重，每个节点每天取最后一条
- 再比较相邻日期或一周路径
- 结果至少返回：
  `nodeid / source_business / switched_business / first_switch_day / latest_business`

### 2. 状态变化

如果用户说“离线 / 恢复 / 掉量”：

- 先看 `state`
- 再补 `node_join.nodeinfo.status`
- 如果两者冲突，标记“状态口径不一致”

### 3. 利用率

如果用户说“利用率高低 / 是否低利用率”：

- 不用 `online_ratio` 代替利用率
- 优先使用：
  `yesterdayp95bw / yesterdaysnapnetbenchbandwidth`
  `dbyp95bw / dbysnapnetbenchbandwidth`
  `dby2p95bw / dby2snapnetbenchbandwidth`
- 长周期补充：
  `node_join.nodeinfo.sevendayavg95ratio`

### 4. 质量异常

优先字段：

- `retrans`
- `v4pingloss`
- `v6pingloss`
- `cpuutilp70statsabnormal`
- `pinglossp5statsabnormal`
- `retranp5statsabnormal`
- `node_join.nodeinfo.qualityinfosummary.*`

### 4.1 网络评分

如果用户说“质量好节点 / 网络评分 / 压测质量”：

- 采用固定评分逻辑：
  - `avg_retrans <= 0.05` 记 `100`
  - `avg_retrans <= 0.15` 记 `70`
  - 其他记 `40`
  - `avg_pingloss <= 5` 记 `100`
  - `avg_pingloss <= 15` 记 `70`
  - 其他记 `40`
  - `yesterdaySnapNetBenchBandwidth / yesterdaySnapBandwidth >= 0.90` 记 `100`
  - `>= 0.80` 记 `70`
  - 其他记 `40`
- 总分：
  `(retrans_score + pingloss_score + udp_satisfy_score) / 3`
- 口径约定：
  - `网络评分 = 100` 认定为“质量好节点”
  - `网络评分 < 100` 不认定为“质量好节点”

通常还会同时返回：

- `bw_usage_ratio`
- `udp_actual_satisfy_pct`
- `retrans_pct`
- `pingloss_pct`

### 5. 策略 / 限速 / 控制面

优先字段：

- `netbenchlimitbandwidth`
- `outlimitbandwidthbynode`
- `tags`
- `node_join.nodestaticinfo.nodeqoskillerconfig.status`
- `node_join.nodestaticinfo.nodeudpqoskillerconfig.status`
- `node_join.nodeinfo.combineddata.businessstatus`

### 6. 大盘切片 / 汇总画像

如果用户说“整体情况怎样 / 哪类节点占比高 / 哪个业务在哪些省份多”：

- 优先查 `node_analysis_summary`
- 它是聚合切片表，不追单节点
- 核心字段通常是：
  `customersummary / province / isp / tcpnattype / udpnattype / dialtype / os / state / count`

适合输出：

- 节点数分布
- 带宽分布
- 某业务在不同地域/网络属性下的画像

### 7. 拨号异常 / 账号异常

如果用户说“拨号有没有问题 / PPPoE 是否正常 / 账号是否异常”：

- 优先查 `dial_acc`
- 核心不是业务，而是 `accounts[]`
- 重点字段：
  `connectstatus / dialstatus / error / netdevname / pppinterface / pingresult / ipv6pingresult`

适合输出：

- 整机拨号是否正常：`dialisnormal`
- 哪条账号/接口失败了
- 失败报错是什么
- IPv4 / IPv6 连通性是否正常

## 常见需求翻译

### “帮我查本周从腾讯直连独立切到其他业务的节点”

思路：

- 每节点每天取最后一条
- 识别源业务集合
- 找首次切出日和最新业务
- 区分：
  - 仍在其他业务
  - 切成空业务
  - 已切回源业务

### “按采购统计昨天离线节点”

思路：

- 先筛昨天离线
- 按 `purchasername / purchaserid` 聚合
- 返回：
  `节点数 / 主要业务 / 主要省份 / 主原因`

### “看某个业务的大盘分布”

思路：

- 优先查 `node_analysis_summary`
- 按业务、地域、运营商、NAT、系统等维度切片
- 返回：
  `customersummary / province / isp / state / count / bandwidth / actualbandwidth`

### “看这台节点是不是拨号有问题”

思路：

- 优先查 `dial_acc`
- 先按 `_id + day + hour` 取最近一条
- 展开 `accounts[]`
- 返回：
  `account / connectstatus / dialstatus / error / netdevname / pppinterface / pingresult`

### “看看这批节点为什么流失”

思路：

- 先筛连续在线 -> 昨天 mostly outline
- 再补：
  - 业务变化
  - 利用率变化
  - 限速变化
  - 质量异常
  - `node_join` 静态离线 / 重部署 / QoS 状态

### “筛质量好且利用率低的节点”

思路：

1. 先用 `node_analysis_data` 计算网络评分
2. 只保留：
   - `total_score = 100`
   - `bw_usage_ratio <= 阈值`
3. 如需更稳，可再补 `node_join.sevendayavg95ratio`

默认阈值如果用户未指定，可先用：

- `bw_usage_ratio <= 30`

### “先看大盘，再下钻到节点”

思路：

1. 先用 `node_analysis_summary` 看整体分布和切片
2. 确定重点业务 / 重点省份 / 重点运营商
3. 再回到 `node_analysis_data` 查具体节点
4. 如有必要，再补 `node_join` 或 `dial_acc`

## 输出风格

- 先给结论，再给 SQL 或文件路径
- 结果过大时不在消息里贴全部明细
- 如果口径有歧义，优先先跑小样本验证
- 如果是大盘分析，优先先给聚合结果，不急着贴全量节点
- 如果是排障分析，优先给“最可能原因 + 对应关键字段”
- 如果是质量筛选，优先先给 `score + 利用率 + 节点清单`
