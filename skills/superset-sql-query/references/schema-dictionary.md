# Schema Dictionary

## 表角色

### `jarvis.node_analysis_data`

定位：

- 节点时点快照事实表
- 适合做状态变化、业务变化、利用率变化、质量异常分析

主键/分区：

- `nodeid`
- `day` `varchar`
- `hour` `varchar`
- `time` `bigint`

最常用字段：

- 身份与归属：
  `nodeid`, `nodetype`, `vendorid`, `purchasername`, `purchaserid`
- 地域：
  `province`, `city`, `isp`
- 状态：
  `state`, `stage`, `tags`
- 业务：
  `customersummary`
- 带宽/限制：
  `bandwidth`, `actualbandwidth`, `netbenchlimitbandwidth`, `outlimitbandwidthbynode`
- 利用率相关：
  `yesterdayp95bw`, `dbyp95bw`, `dby2p95bw`
  `yesterdaysnapnetbenchbandwidth`, `dbysnapnetbenchbandwidth`, `dby2snapnetbenchbandwidth`
- 质量相关：
  `retrans`, `v4pingloss`, `v6pingloss`
  `cpuutilp70statsabnormal`, `pinglossp5statsabnormal`, `retranp5statsabnormal`
- 部署相关：
  `lastdeploystate`, `lastdeploydesc`, `lastdeploytime`

字段理解：

- `customersummary`：这个时点节点主业务
- `state`：这个时点的节点状态
- `netbenchlimitbandwidth`：压测/限制后的带宽口径
- `outlimitbandwidthbynode`：当前对外输出限制
- `yesterdayp95bw / snap_limit`：更适合算真实利用率

### `jarvis.node_join`

定位：

- 节点补充画像表
- 适合补静态属性、运行画像、7 天质量汇总、重部署与控制面信息

主键/分区：

- `_id`
- `day` `integer`
- `hour` `integer`

顶层字段：

- `_id`
- `nodeinfo`
- `nodestaticinfo`
- `nodestaticinfo.nodetype`
- `day`
- `hour`

## `node_join` 重点字段

### `nodeinfo.*`

适合补运行画像：

- `nodeinfo.status`
- `nodeinfo.devmark`
- `nodeinfo.runinfo.mode`
- `nodeinfo.runinfo.bizsyncmode`
- `nodeinfo.runinfo.lbmode`
- `nodeinfo.combineddata.businessstatus`
- `nodeinfo.netbenchresults.actualbw`
- `nodeinfo.netbenchresults.limitbw`
- `nodeinfo.sevendayavg95ratio`
- `nodeinfo.qualityinfosummary.pinglossrate`
- `nodeinfo.qualityinfosummary.retransrate`
- `nodeinfo.qualityinfosummary.cpuutil`
- `nodeinfo.needredeployreason`

理解：

- `nodeinfo.status`：运行态/控制面看到的状态
- `sevendayavg95ratio`：7 天平均 95 利用率口径
- `needredeployreason`：是否建议重部署

### `nodestaticinfo.*`

适合补静态配置：

- `nodestaticinfo.stage`
- `nodestaticinfo.vendorid`
- `nodestaticinfo.regsource`
- `nodestaticinfo.customermode`
- `nodestaticinfo.nominalinfo.deliverytype`
- `nodestaticinfo.nominalinfo.resourcetype`
- `nodestaticinfo.nominalinfo.dialtype`
- `nodestaticinfo.nominalinfo.nattype`
- `nodestaticinfo.nominalinfo.scheduleisps`
- `nodestaticinfo.offline.type`
- `nodestaticinfo.offline.reason`
- `nodestaticinfo.deploystate`
- `nodestaticinfo.nodeqoskillerconfig.status`
- `nodestaticinfo.nodeudpqoskillerconfig.status`
- `nodestaticinfo.tags`

理解：

- `offline.*`：静态侧记录的离线原因
- `nodeqoskillerconfig / nodeudpqoskillerconfig`：QoS / 策略控制
- `nominalinfo.*`：节点理论属性和调度属性

### `jarvis.node_analysis_summary`

定位：

- 节点汇总切片表
- 更像把同一时刻、同一组维度下的一批节点做聚合后的结果
- 适合做大盘分布、维度切片、画像统计，不适合直接追单节点

主键/分区：

- 无单节点主键
- `day` `bigint`
- `hour` `bigint`
- `time` `bigint`

最常用维度字段：

- `nodetype`
- `customersummary`
- `nodecustmertype`
- `nodedeliverytype`
- `province`
- `isp`
- `tcpnattype`
- `udpnattype`
- `dialtype`
- `os`
- `arch`
- `isvm`
- `isroot`
- `state`
- `resourcetype`
- `supplysidedeliverytype`

最常用指标字段：

- `count`
- `bandwidth`
- `actualbandwidth`
- `netbenchlimitbandwidth`
- `outbandwidthbynode`
- `outlimitbandwidthbynode`
- `pinglossp5bizbw`
- `pinglossp10bizbw`
- `uptraffictotal`
- `downtraffictotal`
- `retrans`
- `v4pingloss`
- `v6pingloss`
- `maxioutil`
- `diskdivbw`
- `cpuutilp70statsabnormal`
- `pinglossp5statsabnormal`
- `pinglossp10statsabnormal`
- `retranp5statsabnormal`
- `udpactualbandwidth`
- `udpnetbenchlimitbandwidth`

字段理解：

- `count`：当前这个维度切片下命中的节点数
- 这张表里很多数值字段更像“聚合后的切片指标”，适合拿来做分布和大盘统计
- 它没有 `nodeid`，所以不适合直接做单节点排查
- 如果要追单节点变化，还是优先查 `node_analysis_data`

使用建议：

- 看业务/地域/网络属性分布
- 看某个业务在某个省份、运营商、NAT 类型下的大盘画像
- 做 BI 看板、切片统计、聚合趋势

### `jarvis.dial_acc`

定位：

- 拨号账号/拨号链路状态表
- 重点不是业务带宽，而是节点上拨号账号、接口、拨号状态、错误信息

主键/分区：

- `_id`
- `day` `integer`
- `hour` `integer`

顶层字段：

- `_id`
- `accounts`
- `createat`
- `lastupdatenano`
- `updateat`
- `lbtype`
- `dialisnormal`
- `edittime`
- `ipv6enable`
- `version`
- `dialonphysicalnic`
- `lbtypev6`
- `day`
- `hour`

`accounts[]` 里最重要的字段：

- `ip`
- `mac`
- `account`
- `connectstatus`
- `dialstatus`
- `error`
- `netdevname`
- `pppinterface`
- `ipv6`
- `pingresult`
- `ipv6pingresult`
- `bras`
- `gateway`
- `mask`
- `dialonphysicalnic`
- `ipv6dhcp`
- `multidial`

字段理解：

- `_id`：通常对应节点 ID
- `accounts`：一台节点上的拨号账号列表，是这张表的核心
- `connectstatus / dialstatus / error`：看拨号是否成功、失败原因是什么
- `netdevname / pppinterface`：看拨号挂在哪个接口上
- `pingresult / ipv6pingresult`：看拨号成功后连通性是否正常
- `dialisnormal`：整机拨号是否整体正常
- `ipv6enable / lbtype / lbtypev6`：更偏拨号与负载方式配置

使用建议：

- 查节点拨号是否正常
- 查拨号账号是否缺失、异常、报错
- 查某个接口/PPPoE 线路是否拨通
- 排查“节点在线但业务异常”时的拨号层问题

## 关联规则

通常用：

```sql
node_analysis_data.nodeid = node_join._id
```

按天 join 时注意类型：

```sql
CAST(node_join.day AS varchar) = node_analysis_data.day
```

## 推荐分析模式

### 1. 纯事实分析

只用 `node_analysis_data`

适合：

- 查节点当前业务
- 查业务切换
- 查昨天离线
- 查带宽和利用率变化

### 2. 归因增强

`node_analysis_data` 先筛节点，再补 `node_join`

适合：

- 为什么掉线
- 是否需要重部署
- QoS 是否生效
- 7 天画像是否长期低利用率
- 状态口径是否打架

## 高频业务问题映射

- “切业务”：
  `customersummary`
- “低利用率 / 高利用率”：
  `p95 / snap_limit`, `sevendayavg95ratio`
- “质量问题”：
  `retrans`, `pingloss`, `qualityinfosummary.*`
- “策略问题”：
  `tags`, `netbenchlimitbandwidth`, `nodeqoskillerconfig.*`
- “部署问题”：
  `lastdeploystate`, `lastdeploydesc`, `needredeployreason`
