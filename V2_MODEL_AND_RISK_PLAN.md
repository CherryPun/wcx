# V2 新节点泛化模型与风险规则方案

## 目标

V1 已经跑通离线闭环：

```text
节点画像 + 历史节点-业务收益 -> Top3 推荐 -> 存量纠偏 -> 单节点审计
```

V2 的重点是解决真正新交付节点的问题：节点没有历史业务收益时，也能根据节点画像推荐 Top3，并给出原因和风险。

## 当前已实现

已落地 `v2_ranking_model.py`，当前版本是无外部 ML 依赖的画像分段排序模型：

```text
模型类型：profile_segment_ranker
训练数据：v1_training_pairs.csv
业务名称：v1_business_name_map_enriched.csv
风险画像：v1_business_risk_profile.csv
字段契约：node_profile_v1_fields.json
```

排序主信号：

```text
rank_score =
  0.60 * 匹配分段历史最佳业务胜率
+ 0.20 * 分段冠军证据
+ 0.15 * 预估最佳概率
+ 0.05 * 预估综合收益分
```

这样做的原因是：新节点没有自身业务收益，最可信的泛化信号是“相似画像节点上，哪个业务历史上最常成为最佳业务”，收益均值只做辅助兜底，避免被少数全局高收益业务过度吸走。

## 当前产物

全量构建命令：

```bash
python3 v2_ranking_model.py build \
  --business-map v1_business_name_map_enriched.csv
```

`v1_business_name_map_enriched.csv` 以 `node_day_ops_wide` 中非空 `customerName` 为基础，并用 Superset 验收规则客户清单补齐高频缺名业务。

生成文件：

```text
v2_ranking_model.json
v2_node_recommendations.csv
v2_model_metrics.json
```

校验：

```bash
python3 v2_ranking_model.py check
python3 -m unittest -v test_v2_ranking_model.py
```

单节点推荐：

```bash
python3 v2_ranking_model.py recommend-node <node_id> --json
```

批量节点推荐：

```bash
python3 v2_ranking_model.py recommend-batch \
  --node-ids <node_id_1>,<node_id_2> \
  --json
```

单条件/多条件筛选后逐节点推荐：

```bash
python3 v2_ranking_model.py recommend-filter \
  --where province=安徽 \
  --where isp=移动 \
  --limit 20
```

条件组合汇总推荐：

```bash
python3 v2_ranking_model.py recommend-segments \
  --where province=安徽 \
  --where isp=移动 \
  --group-by city,resourcetype,nattype,dialtype,bw_bucket \
  --min-nodes 2 \
  --limit 20
```

`recommend-segments` 的语义是：先按 `--where` 找到节点池，再按 `--group-by` 对节点画像做分组，输出每个画像组合下占比最高的 Top1 推荐业务、覆盖节点数、业务占比、平均分和风险节点数。默认优先读取已生成的 `v2_node_recommendations.csv`，因此适合运营分析；如果要实时按模型重算，加 `--score-live`。

前端条件推荐报告：

```bash
python3 v2_ranking_model.py export-condition-report \
  --min-nodes 10
```

生成文件：

```text
v2_frontend_condition_business_report.html
v2_frontend_condition_business_report.csv
v2_frontend_condition_business_report.json
v2_frontend_condition_business_report_data.js
```

这份数据不依赖节点 ID。它把节点画像条件组合聚合成可检索记录，默认包含 7 个粒度：

`v2_frontend_condition_business_report.html` 是静态前端报告，可直接在浏览器打开；页面数据来自同目录下的 `v2_frontend_condition_business_report_data.js`。

```text
province,isp
province,isp,resourcetype
province,isp,city,resourcetype
province,isp,resourcetype,nattype,dialtype
province,isp,resourcetype,deliverytype,device_type
province,isp,resourcetype,bw_bucket,actualbandwidth_bucket
province,isp,city,resourcetype,nattype,dialtype,bw_bucket
```

主要字段：

```text
segment_level: 条件粒度名称
condition_key / condition_text: 前端检索用条件
node_count: 该条件组合覆盖的历史节点数
confidence_level / confidence_reason: 推荐置信度
business_top1/top2/top3: 推荐业务 ID
business_name_top1/top2/top3: 推荐业务名称；缺失时兜底为 business_id:<ID>
vote_share_top1/top2/top3: Top3 加权投票占比
top1_share_top1/top2/top3: 成为节点 Top1 的占比
high_risk_rate_top1/top2/top3: 高风险占比
medium_risk_rate_top1/top2/top3: 中风险占比
```

正式全量构建后的离线验证指标：

```text
test_nodes: 5323
hit@1: 0.2574
hit@3: 0.4770
nDCG@3: 0.5072
top1_observed_coverage: 0.7451
observed_regret: 0.3438
```

Top1 业务分布：

```text
10000251: 2991
10000182: 1967
10000234: 276
10000149: 89
```

输出字段重点：

```text
v2_business_top1 / top2 / top3
v2_business_name_top1 / top2 / top3
v2_rank_score_top1 / top2 / top3
v2_expected_score_top1 / top2 / top3
v2_expected_best_rate_top1 / top2 / top3
v2_reason_top1 / top2 / top3
v2_risk_level_top1 / top2 / top3
v2_risk_reasons_top1 / top2 / top3
```

## 新节点模型思路

建议做两阶段排序：

```text
候选召回 -> 精排打分 -> Top3 + 原因 + 风险
```

第一阶段：候选召回

```text
按省份、运营商、资源类型、NAT、拨号类型、带宽档位召回历史上表现好的业务
补充全局稳定业务
过滤历史样本太少的业务
```

第二阶段：精排模型

```text
训练样本粒度：node_id + business
标签：0.5 * 矿主收益归一分 + 0.5 * 平台利润归一分
模型：LightGBM/CatBoost Ranking 优先，其次 RandomForest/ExtraTrees baseline
输出：每个业务的预估综合分、矿主分、平台分、置信度
```

推荐特征：

```text
节点基础：province/city/isp/resourcetype/deliverytype/device_type/isvm
网络形态：nattype/dialtype/IPv6/拨号账号数
资源能力：bw/actualbandwidth/corenum/memtotal/totaldisksize/ssd/hdd
网络质量：重传/丢包/RTT/Prometheus retrans
稳定性：在线时长、离线次数、昨日/前日在线率
业务统计：业务全局收益均值、分省运营商收益、样本量、波动
交叉特征：business x province、business x isp、business x resource_type、business x bw_bucket
```

验证方式：

```text
V2 baseline 当前仍按节点随机切分；正式模型建议继续升级为时间切分
用早期数据训练，后期数据验证
指标看 hit@1、hit@3、nDCG@3、regret、Top1 预估收益误差
按省份/运营商/资源类型分层看效果
```

## 风险规则建议

由于目前没有业务准入文档，V2 不建议一上来做硬阻断。先做软风险分层：

```text
low: 可推荐
medium: 可推荐但需要人工看风险
high: 不建议自动切换，只进入人工复核
```

全局节点风险：

```text
高重传
高丢包
高 RTT
磁盘使用率高
CPU/内存压力高
IO 利用率高
SMART/ZFS 异常
关键画像字段缺失
```

业务历史可行域风险：

```text
对每个业务统计历史 Top 表现节点的画像范围
如果新节点明显落在该业务历史成功样本范围之外，标记 medium/high
例如：
  该业务历史成功节点 p95 重传 <= 5%，当前节点 15%
  该业务历史成功节点 p05 带宽 >= 1000Mbps，当前节点 300Mbps
  该业务历史很少在某运营商/省份成功
```

当前业务/切换风险：

```text
当前业务来自历史推断还是 Superset 最近窗口
当前业务不在 Top3
Top1 与当前业务分差是否足够大
推荐业务样本量是否充足
收益是否稳定，是否只靠少量天数冲高
```

## 推荐落地顺序

1. 补齐 Superset 业务名称映射。
2. 固定当前业务历史口径，并允许 Superset 最近窗口覆盖。
3. 生成业务级历史可行域画像。
4. 训练 V2 排序模型。
5. 在 `audit-node` 中展示 V2 模型分、历史收益分、风险原因。
6. 先人工评审，不自动切业务。
