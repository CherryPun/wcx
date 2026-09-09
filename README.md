# 多业务节点最佳业务推荐

本项目使用真实节点业务数据，基于节点画像和日粒度业务收益，推荐大节点更适合的主流业务：

- 矿主目标：单日成本/矿主收益表现更好的业务。
- 运营目标：单日收入减成本后的平台利润更好的业务。
- 当前主目标：矿主收益和平台利润各占 50%。

## 最新量化闭环（截至 2026-09-08）

本轮同时训练最近 31 天主模型与最近 14 天挑战模型，两者使用相同的
`2026-09-05` 至 `2026-09-08` 测试集。挑战模型未通过覆盖度、推荐集中度和
低可信占比门槛，因此正式推荐继续使用 31 天模型；14 天窗口只用于近期趋势
监控和容量预测。

```text
31 天训练数据：280,981 个干净节点日，12,063 个节点，26 个观测业务
31 天候选业务：21 个
31 天测试：矿主 R²=0.128，平台 R²=0.117
31 天可信度：低；82.9% 节点低可信，91.5% 的 Top1 平台利润区间跨零
14 天测试：矿主 R²=0.156，平台 R²=0.104
14 天问题：Top1 最大集中度 75.7%，低可信节点 95.7%
容量：1,891 个四层容量池，1,454 个证据充分，340 个规划调整，53 个容量阻断
价格：71 个业务+运营商组合，6,834 个不混合计价类型/计价项的节点日主签名
```

最新报告：

```text
recent_31d_large_mainstream_v5_hybrid_20260908/v5_business_recommendation_report.html
recent_31d_large_mainstream_v6_capacity_20260908/v6_capacity_report.html
recent_31d_large_mainstream_v6_capacity_20260908/v6_node_report.html
recent_31d_large_mainstream_price_report_20260908/business_price_report.html
recent_v5_dual_window_comparison_20260908/v5_dual_window_comparison.md
最新量化报告_20260908.md
```

完整重建使用明确日期窗口，避免旧目录决定训练日期：

```bash
python3 build_v3_daily_business_training.py \
  --candidate-json recent_31d_large_20260908/large_candidate_node_ids_recent_1m.json \
  --output-dir recent_31d_large_mainstream_v3_daily_weighted_20260908 \
  --current-nodes current_non_idc_large_scan_20260909/current_online_inservice_non_idc_large_nodes_20260909.csv \
  --start-day 2026-08-09 --end-day 2026-09-08 --refresh

python3 v5_hybrid_recommendation.py build \
  --pairs recent_31d_large_mainstream_v3_daily_weighted_20260908/v1_large_mainstream_outputs/v1_training_pairs.csv \
  --historical-profiles historical_node_profiles_20260909/current_non_idc_large_node_attrs_20260909.csv \
  --latest-pressure-profiles latest_node_pressure_profiles_20260909.csv \
  --current-nodes current_non_idc_large_scan_20260909/current_online_inservice_non_idc_large_nodes_20260909.csv \
  --current-business current_non_idc_large_scan_20260909/current_business_from_wide_20260908_20260908.csv \
  --source-summary recent_31d_large_mainstream_v3_daily_weighted_20260908/v3_daily_training_summary.json \
  --output-dir recent_31d_large_mainstream_v5_hybrid_20260908 \
  --start-day 2026-08-09 --end-day 2026-09-08

python3 build_business_price_report.py \
  --facts recent_31d_large_mainstream_v3_daily_weighted_20260908/multibusiness_outcomes_large_mainstream_v3_daily.csv \
  --output-dir recent_31d_large_mainstream_price_report_20260908
```

## 当前闭环入口

V1 用于历史节点收益闭环和存量纠偏，V2 用于新节点画像泛化推荐。

```bash
# 历史收益闭环：构建推荐、当前业务推断、纠偏候选
python3 v1_recommendation_pipeline.py build \
  --business-map v1_business_name_map_enriched.csv
python3 v1_recommendation_pipeline.py check
python3 v1_recommendation_pipeline.py audit-node <node_id> --json

# 新节点泛化推荐：输入 node_id 输出 Top3 + 原因 + 风险
python3 v2_ranking_model.py build \
  --business-map v1_business_name_map_enriched.csv
python3 v2_ranking_model.py check
python3 v2_ranking_model.py recommend-node <node_id> --json

# 批量节点推荐
python3 v2_ranking_model.py recommend-batch \
  --node-ids <node_id_1>,<node_id_2> \
  --json

# 单条件/多条件筛选节点后逐节点推荐
python3 v2_ranking_model.py recommend-filter \
  --where province=安徽 \
  --where isp=移动 \
  --limit 20

# 条件组合汇总：省份+运营商 -> 其他画像组合 -> 推荐业务
python3 v2_ranking_model.py recommend-segments \
  --where province=安徽 \
  --where isp=移动 \
  --group-by city,resourcetype,nattype,dialtype,bw_bucket \
  --min-nodes 2 \
  --limit 20

# 前端条件推荐报告：不依赖 node_id，导出条件组合 -> Top3 业务
python3 v2_ranking_model.py export-condition-report \
  --min-nodes 10
```

大节点专用数据和模型已单独输出，后续大节点分析请优先使用这组文件，避免小盒子画像混入：

```bash
# 大节点单/多条件推荐
python3 v2_ranking_model.py recommend-filter \
  --where node_size_type=large_node \
  --nodes multibusiness_nodes_large.csv \
  --model v2_large_outputs/v2_ranking_model.json \
  --limit 20

# 大节点条件推荐报告
python3 v2_ranking_model.py export-condition-report \
  --nodes multibusiness_nodes_large.csv \
  --model v2_large_outputs/v2_ranking_model.json \
  --recommendations v2_large_outputs/v2_node_recommendations.csv \
  --min-nodes 1 \
  --output-csv v2_frontend_condition_business_report_large.csv \
  --output-json v2_frontend_condition_business_report_large.json \
  --output-data-js v2_frontend_condition_business_report_large_data.js
```

当前已生成的 V2 正式产物：

```text
v2_ranking_model.json
v2_node_recommendations.csv
v2_model_metrics.json
v2_frontend_condition_business_report.html
v2_frontend_condition_business_report.csv
v2_frontend_condition_business_report.json
v2_frontend_condition_business_report_data.js
```

大节点专用产物：

```text
multibusiness_nodes_large.csv
v1_training_pairs_large.csv
v2_large_outputs/v2_ranking_model.json
v2_large_outputs/v2_node_recommendations.csv
v2_large_outputs/v2_model_metrics.json
v2_frontend_condition_business_report_large.html
v2_frontend_condition_business_report_large.csv
v2_frontend_condition_business_report_large.json
v2_frontend_condition_business_report_large_data.js
```

最近一个月大节点日粒度产物已输出到 `recent_month_large_1d/`，训练窗口为 `sample_day 2026-08-02` 至 `2026-09-01`。这版会先按 Superset 最新节点快照分片找大节点候选，再用项目内 `node_size_type=large_node` 口径二次过滤，避免小盒子混入：

```bash
python3 -u build_recent_large_training.py \
  --output-dir recent_month_large_1d \
  --outcome-window-days 1 \
  --workers 4 \
  --candidate-chunk-size 800 \
  --min-business-support 1

python3 v2_ranking_model.py recommend-filter \
  --where province=江苏 \
  --where isp=电信 \
  --nodes recent_month_large_1d/multibusiness_nodes_large_recent_1m.csv \
  --model recent_month_large_1d/v2_large_recent_outputs/v2_ranking_model.json \
  --limit 20
```

最近一个月大节点 1d 关键产物：

```text
recent_month_large_1d/multibusiness_nodes_large_recent_1m.csv
recent_month_large_1d/multibusiness_outcomes_large_recent_1m.csv
recent_month_large_1d/v1_training_pairs_large_recent_1m.csv
recent_month_large_1d/v2_large_recent_outputs/v2_ranking_model.json
recent_month_large_1d/v2_large_recent_outputs/v2_node_recommendations.csv
recent_month_large_1d/v2_large_recent_outputs/v2_model_metrics.json
recent_month_large_1d/v2_frontend_condition_business_report_large_recent_1m.html
recent_month_large_1d/recent_large_training_summary.json
```

主流业务 allowlist 已固化到 `mainstream_business_allowlist.csv`。当前口径只包含已确认的字节、百度、优酷、快手、腾讯、爱奇艺、B站主流业务及七牛相关业务；审计中发现但未确认属于主流范围的变体不进入训练。虚拟/签约业务绑定审计结果记录在 `business_binding_audit/`，七牛手工确认绑定关系记录在 `business_virtual_binding_map.csv`。使用主流业务版入口：

```bash
python3 -u build_mainstream_large_training.py \
  --source-dir recent_month_large_1d \
  --output-dir recent_month_large_mainstream_1d \
  --min-business-support 1

python3 v2_ranking_model.py recommend-node <node_id> \
  --nodes recent_month_large_mainstream_1d/multibusiness_nodes_large_mainstream_recent_1m.csv \
  --model recent_month_large_mainstream_1d/v2_large_mainstream_outputs/v2_ranking_model.json \
  --json

# 现网重匹配：存量节点先看当前真实收益，模型推荐只作为辅助
python3 -u scan_current_non_idc_large_mismatches.py

# 主流业务完整统计：自动读取最新现网扫描；非主流业务只记录排除量
python3 -u build_mainstream_data_statistics.py
```

主流业务版关键产物：

```text
mainstream_business_allowlist.csv
business_virtual_binding_map.csv
business_binding_audit/business_virtual_binding_map_full_audit_202608.csv
business_binding_audit/mainstream_allowlist_expansion_selected_202608.csv
business_binding_audit/mainstream_allowlist_expansion_excluded_202608.csv
recent_month_large_mainstream_1d/multibusiness_outcomes_large_mainstream_recent_1m.csv
recent_month_large_mainstream_1d/v1_training_pairs_large_mainstream_recent_1m.csv
recent_month_large_mainstream_1d/v2_large_mainstream_outputs/v2_ranking_model.json
recent_month_large_mainstream_1d/v2_large_mainstream_outputs/v2_node_recommendations.csv
recent_month_large_mainstream_1d/v2_frontend_condition_business_report_large_mainstream_recent_1m.html
recent_month_large_mainstream_1d/mainstream_large_training_summary.json
current_non_idc_large_scan/current_non_idc_large_recommendations_20260902.csv
current_non_idc_large_scan/current_non_idc_large_mismatch_candidates_20260902.csv
mainstream_data_statistics/mainstream_data_statistics_report_20260902.md
mainstream_data_statistics/mainstream_data_statistics_summary_20260902.json
```

V2 主流业务大节点 1d 离线验证结果：`hit@1=0.2760`，`hit@3=0.5571`，`nDCG@3=0.4632`。完整说明见 `业务推荐逻辑与训练说明.md`、`V1_RECOMMENDATION_PIPELINE.md` 和 `V2_MODEL_AND_RISK_PLAN.md`。

单位带宽收益率模型已作为对照版输出到 `recent_month_large_mainstream_1d_unit_bw/`，目标为 `cost/profit ÷ effective_bandwidth_mbps`，V2 排序权重更偏 `expected_score`：

```bash
python3 -u build_mainstream_large_training.py \
  --source-dir recent_month_large_1d \
  --output-dir recent_month_large_mainstream_1d_unit_bw \
  --target-mode unit_bandwidth \
  --min-business-support 1

python3 -u scan_current_non_idc_large_mismatches.py \
  --output-dir current_non_idc_large_scan_unit_bw \
  --model recent_month_large_mainstream_1d_unit_bw/v2_large_mainstream_outputs/v2_ranking_model.json
```

收益率版 V2 离线验证结果：`hit@1=0.2600`，`hit@3=0.5218`，`nDCG@3=0.3730`，`observed_regret=0.0397`。该版本先作为人工复核和策略对照，不直接覆盖默认 1d 模型。

收益率版前端收益报告可直接生成并打开：

```bash
python3 -u build_business_recommendation_profit_report.py
open business_recommendation_profit_report_unit_bw.html
```

报告产物：

```text
business_recommendation_profit_report_unit_bw.html
business_recommendation_profit_report_unit_bw.csv
business_recommendation_profit_report_unit_bw_data.js
```

含调度字段版已纳入 `scheduleisps`、`analysis_transprovrate`、`join_isbantransprov`、`join_isipv6schedule`，用于区分调度运营商、跨省比例、禁跨省和 IPv6 调度能力：

模型会进一步派生 `network_schedule_type`：`本网本省`、`本网出省`、`异网本省`、`异网出省`。`scheduleISPs` 为空表示本网，并归一为节点自身运营商；非空时严格按原始顺序只取第一个运营商，其余值不参与训练和容量分池。`transProvRate` 为空或 0 表示本省、100 表示出省，1 至 99 视为异常值并从训练中剔除。

当前默认现网扫描使用：

```bash
python3 scan_current_non_idc_large_mismatches.py
python3 build_business_recommendation_profit_report.py
```

默认模型目录为 `recent_month_large_mainstream_1d_unit_bw_network_scope/`，默认扫描目录为 `current_non_idc_large_scan_network_scope/`。

```bash
python3 -u build_mainstream_large_training.py \
  --source-dir recent_month_large_1d \
  --output-dir recent_month_large_mainstream_1d_unit_bw_sched \
  --target-mode unit_bandwidth \
  --min-business-support 1

python3 -u scan_current_non_idc_large_mismatches.py \
  --output-dir current_non_idc_large_scan_unit_bw_sched \
  --model recent_month_large_mainstream_1d_unit_bw_sched/v2_large_mainstream_outputs/v2_ranking_model.json
```

新版现网报告：

```text
business_recommendation_profit_report_unit_bw_sched.html
business_recommendation_profit_report_unit_bw_sched.csv
current_non_idc_large_scan_unit_bw_sched/current_condition_business_report_unit_bw_sched_20260902.html
current_non_idc_large_scan_unit_bw_sched/current_condition_business_report_unit_bw_sched_20260902.csv
```

调度字段版 V2 离线验证结果：`hit@1=0.2676`，`hit@3=0.5271`，`nDCG@3=0.3765`，`observed_regret=0.0392`。

## 数据口径

- 业务数据：Superset 数据库 `yzh-starrocks` 的 `test.node_day_ops_wide_full`。
- 业务名称：优先使用 `node_day_ops_wide` 中非空 `customerName`，并用 Superset 验收规则客户清单补齐；虚拟业务会结合 `signId/signName`、`vendorSuggestCustomersName`、`virtualCustomersName` 做审计，输出为 `v1_business_name_map_enriched.csv` 和 `business_binding_audit/`。
- 节点属性：Superset 数据库 `jf-jarvis` 的 `jarvis.node_analysis_data`、`jarvis.node_join` 和 `jarvis.dial_acc`；调度字段来自 `node_join.nodestaticinfo.nominalinfo.scheduleisps`、`node_analysis_data.transprovrate`、`node_join.nodestaticinfo.nominalinfo.isbantransprov` 和 `isipv6schedule`。
- 运行健康指标：VictoriaMetrics/Prometheus，用于健康准入，不进入收益预测模型。
- 存量纠偏：取消“历史分段最优覆盖数”主导切换的口径；当前业务真实利润和单位利润优于推荐目标业务现网中位值时，保留当前业务。
- 当前 1d 历史范围：`2026-08-02` 至 `2026-09-01`。
- 节点筛选：至少运行 3 个不同业务，业务首次出现日期至少分布在 3 天，且至少有一条 `state='online'` 记录。
- 当前样本：节点 ID 后缀 `00` 至 `0c`，约 5.08%，不是全量节点推断。
- 属性时间：业务首次出现日前一天的最新可用小时，避免把业务启动后的观测泄漏到目标窗口。

## 特征边界

收益预测模型只使用节点固有属性：

- 地域、运营商、机房、设备型号、硬件型号、架构、NAT 和拨号配置。
- 节点部署配置、合作类型、IPv6/UPnP 配置和管理属性。
- 名义带宽、CPU 核数、内存总量、总盘/HDD/SSD/系统盘容量。
- 业务与名义带宽、CPU/内存/磁盘容量的交互项。

以下字段不进入收益预测模型，但保留在节点数据和健康契约中：

- 历史在线时长、前一天在线率、实际带宽和带宽利用率。
- 运行中 CPU/内存/磁盘使用率、磁盘 IO、RTT、丢包和重传。
- SMART、ZFS、Prometheus 运行时指标以及业务流量。

健康指标的处理规则是：指标不合格则阻断对应业务，指标缺失则转人工检查，不将缺失值当作正常值放行。

## 模型

- 推荐分类器：`ExtraTreesClassifier`
- 参数：`n_estimators=200`、`max_depth=18`、`min_samples_leaf=5`、`max_features='sqrt'`、`n_jobs=-1`
- 辅助金额模型：`Ridge(alpha=10.0, solver='lsqr')`，目标使用有符号对数变换
- 验证方式：按节点切分训练集和测试集，并使用随机种子 `42, 52, 62, 72, 82` 验证稳定性

最新严格固有属性模型的测试集命中率：

- 矿主最佳业务：17.96%
- 运营最佳业务：18.35%

金额辅助模型测试集 R²：成本 `-0.2986`，利润 `-0.1664`，暂不能认为金额预测有效。完整限制和显著性结果见 `多业务节点最佳业务重建报告.md`。

## 运行环境

使用项目环境运行：

```bash
/Users/nany/.workbuddy/binaries/python/envs/default/bin/python rebuild_multibusiness_model.py
```

脚本只执行查询和本地建模，Superset 凭据从 `/Users/nany/.codex/.superset_env` 读取，不在项目中保存密码。

## 主要文件

- `rebuild_multibusiness_model.py`：数据读取、特征边界、健康准入、训练和评估。
- `项目进展总结_20260810.md`：当前项目进展和结果摘要。
- `多业务节点最佳业务重建报告.md`：完整分析报告。
- `multibusiness_metrics.json`：机器可读指标、字段边界和健康规则。
- `multibusiness_nodes.csv`：节点属性快照和健康观测字段。
- `multibusiness_outcomes.csv`：旧版节点-业务 7 天成本、收入、利润结果；当前主流大节点模型优先使用 `recent_month_large_mainstream_1d/` 下的日粒度数据。
- `multibusiness_recommendations.csv`：训练集/测试集推荐结果和健康阻断理由。
- `multibusiness_eligible_pairs.csv`：合格节点-业务首次出现明细。
- `multibusiness_prometheus_probe.json`：Prometheus 指标探查结果。

# machine-test

## V3.1 日粒度加权主流业务模型（当前口径）

V3 不再把同一节点同一天的多个 `customerId` 当成可比较业务。当天业务由
`state='online' AND stage='inService'` 的主流业务行确定；七牛系列虚拟 ID
统一为 `10000280 / 七牛CDN-ZJ月95`。同日仍有多个有效主流业务、存在未归属
金额或建设带宽不一致时，整天不进训练并写入审计表。

NiuLink 虚拟业务目录通过环境变量鉴权同步，鉴权值不写入项目：

```bash
NIULINK_AUTH='<authorization>' python3 fetch_niulink_virtual_business_bindings.py
```

虚拟业务只有在当天财务行、同日真实业务或业务建议字段能唯一指向一个主流真实业务时才归属；一对多且无法唯一判断的节点日直接剔除。专线和汇聚目前都可参与推荐，不设硬准入限制。

连续重复样本的训练权重为：

```text
单条权重 = 1 / 同节点同业务连续有效天数
```

因此同一业务连续运行 30 天与连续运行 1 天都只贡献 1 个连续段的总权重，避免稳定运行时间长的业务仅凭重复天数支配模型。

网络调度口径固定为：

```text
scheduleISPs 为空 = 调度到节点自身运营商（本网）
scheduleISPs 非空 = 按原始顺序只取第一个运营商
transProvRate 为空或 0 = 本省
transProvRate 100 = 出省
transProvRate 1~99 = 异常值，整节点日不进训练并写入审计表
```

精确运营商路径会保留，例如 `移动->电信` 与 `移动->联通` 是两个不同分段；
空 `scheduleISPs` 会先归一为节点自身运营商，再与显式填写本网的记录合并。

收益目标固定为：

```text
矿主单位收益 = cost_finalAmount / buildBandwidth
平台单位利润 = (revenue_finalAmount - cost_finalAmount) / buildBandwidth
综合分 = 0.5 * 归一化矿主单位收益 + 0.5 * 归一化平台单位利润
```

V2 排序只使用期望综合收益，旧版 `source_rate/champion/expected_best_rate`
投票权重均为 0。`hit@1/hit@3` 只表示推荐是否等于历史实际选择，不代表因果上的
最佳业务准确率。

完整重建：

```bash
python3 -u build_v3_daily_business_training.py \
  --output-dir recent_month_large_mainstream_v3_daily_weighted \
  --min-business-support 30

python3 -u scan_current_non_idc_large_mismatches.py \
  --output-dir current_non_idc_large_scan_v3_daily \
  --model recent_month_large_mainstream_v3_daily_weighted/v2_large_mainstream_outputs/v2_ranking_model.json
```

当前关键产物：

```text
recent_month_large_mainstream_v3_daily_weighted/v3_daily_training_summary.json
recent_month_large_mainstream_v3_daily_weighted/node_day_sampling_audit_v3.csv
recent_month_large_mainstream_v3_daily_weighted/multibusiness_outcomes_large_mainstream_v3_daily.csv
recent_month_large_mainstream_v3_daily_weighted/v2_large_mainstream_outputs/v2_ranking_model.json
recent_month_large_mainstream_v3_daily_weighted/business_recommendation_profit_report_v3_daily.html
current_non_idc_large_scan_v3_daily/current_business_sampling_audit_20260901_20260901.csv
current_non_idc_large_scan_v3_daily/current_non_idc_large_mismatch_candidates_20260902.csv
```

## V4 双收益结果模型与可信度报告

V4 不再学习“历史上最常被选中的业务”，而是对每个候选业务分别预测两个日粒度目标：

```text
矿主单位收益 = cost_finalAmount / 建设带宽
平台单位利润 = (revenue_finalAmount - cost_finalAmount) / 建设带宽
综合分 = 50% 归一化矿主预测 + 50% 归一化平台预测
```

模型画像严格限定为省份、运营商、调度类型、CPU 档位、NAT 类型、内存档位、磁盘档位、
IPv6 能力、建设带宽档位和整体丢包压测满意度。CPU 使用率、运行时流量、城市、资源类型等
字段不进入模型。验证严格按时间切分，训练、区间校准、最终测试
依次后置，并输出每个业务的 MAE、R²、90% 区间覆盖率、历史分配概率和 Doubly Robust
偏差修正诊断。DR 仅用于观察性诊断，不代表因果证明。

整体丢包压测满意度来自 Superset `jarvis.node_join.nodeinfo.netbenchresults`。每个节点只取
最新分区中最新小时、最新上报的一次有效 TCP 压测，不按训练日期展开。多线路节点按
`Σ line.limitbw / Σ line.expectedbw * 100%` 计算整体满意度并限制在 `0~100%`；原始汇总比例
仅保留用于审计，因为节点建设带宽变更后该比例可能超过 100%。缺失节点保持未知，不使用
`pinglossp5avgbw` 或日常质量丢包率替代。

按当前业务规则，最新压测值作为节点静态画像回填到该节点的全部历史收益样本。收益标签仍按
时间切分验证，但压测特征并非逐历史日回放，因此这部分验证存在未来画像信息的限制，不能
解释为严格因果效果或直接支持自动切换业务。

完整重建与校验：

```bash
python3 -u fetch_latest_node_pressure.py \
  --snapshot-day 2026-09-04 \
  --output latest_node_pressure_profiles.csv

python3 -u v4_outcome_recommendation.py build \
  --latest-pressure-profiles latest_node_pressure_profiles.csv \
  --output-dir recent_month_large_mainstream_v4_outcome_latest_pressure

python3 v4_outcome_recommendation.py check \
  --output-dir recent_month_large_mainstream_v4_outcome_latest_pressure

python3 -m unittest -v \
  test_fetch_latest_node_pressure.py \
  test_v4_outcome_recommendation.py
```

当前产物：

```text
latest_node_pressure_profiles.csv
recent_month_large_mainstream_v4_outcome_latest_pressure/v4_business_recommendation_report.html
recent_month_large_mainstream_v4_outcome_latest_pressure/v4_training_summary.json
recent_month_large_mainstream_v4_outcome_latest_pressure/v4_business_training_metrics.csv
recent_month_large_mainstream_v4_outcome_latest_pressure/v4_node_recommendations.csv
recent_month_large_mainstream_v4_outcome_latest_pressure/v4_review_switch_candidates.csv
recent_month_large_mainstream_v4_outcome_latest_pressure/v4_temporal_validation_predictions.csv
```

只有 Top1 点估计平台利润为正且 90% 下界不为负的高可信节点才可进入“优先人工复核”；
其余节点会降级为“谨慎人工复核”或“暂不推荐执行”。任何可信度等级都不允许自动切业务。
现网报告会合并最近一天的权威当前业务快照，分别标记当前已是 Top1、当前非 Top1、
当前业务不在候选集和当前业务未知；只有“当前非 Top1”且模型未拦截的记录才进入人工换业务复核集合。

## V5 公平对照与存量增强模型

完整的数据获取、日级清洗、字段口径、算法、验证、决策和可信度说明见
[`节点业务推荐模型V5全流程说明.md`](节点业务推荐模型V5全流程说明.md)。

V5 在完全相同的日粒度数据、业务候选集和时间窗口上，对以下模型进行公平比较：

- V4 分层收益基线。
- 只使用十项节点画像的 XGBoost 收益模型。
- 使用稳定倾向权重修正历史业务分配偏差的 XGBoost 模型。
- 面向存量节点的业务无关节点残差修正；新节点自动回退到画像模型。

模型选择只看 `2026-08-26` 至 `2026-08-28` 校准期，`2026-08-29` 至
`2026-09-01` 保留为最终时间外测试。节点残差修正必须让校准期 RMSE 至少下降 3%
才启用。倾向权重截断在 `0.25~4.0`，防止极低历史分配概率放大噪声。

调度类型只接受以下四类：

```text
本网本省、本网出省、异网本省、异网出省
```

同时包含本网与异网的混合调度记录不会被静默归类，本轮有 133 个节点日被 V5 排除。

当前完整训练结果：

```text
训练节点日：279,310
有效连续段：17,832.29
历史节点：11,808
候选业务：20

矿主模型：XGBoost + 存量节点残差
矿主 R²：0.2825 -> 0.3786
矿主 RMSE：0.04704 -> 0.04378（下降 6.94%）

平台模型：保留 V4 分层基线，不启用节点残差
平台 R²：0.0263 -> 0.0263
平台 RMSE：0.04917 -> 0.04917
```

平台利润的时间外解释能力仍然很低，且 86.5% 节点的 Top1 平台利润 90% 区间跨零，
因此 V5 总体可信度仍标记为低，不允许自动切业务。当前业务与模型 Top1 不一致的
4,152 个节点写入观察清单；只有 Top1 综合得分的 90% 下界高于当前业务 90% 上界的
节点才写入严格换业务复核清单，本轮严格清单为 0 个。

安装、训练和校验：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-v5.txt

.venv/bin/python v5_hybrid_recommendation.py build \
  --output-dir recent_month_large_mainstream_v5_hybrid

.venv/bin/python v5_hybrid_recommendation.py check \
  --output-dir recent_month_large_mainstream_v5_hybrid

.venv/bin/python -m unittest -v test_v5_hybrid_recommendation.py
```

主要产物：

```text
recent_month_large_mainstream_v5_hybrid/v5_business_recommendation_report.html
recent_month_large_mainstream_v5_hybrid/v5_training_summary.json
recent_month_large_mainstream_v5_hybrid/v5_model_comparison.csv
recent_month_large_mainstream_v5_hybrid/v5_node_recommendations.csv
recent_month_large_mainstream_v5_hybrid/v5_current_not_top1_watchlist.csv
recent_month_large_mainstream_v5_hybrid/v5_review_switch_candidates.csv
```

## V6.2 分层业务容量感知与批量约束分配

V6.2 在 V5 节点收益推荐之后增加四层容量约束，防止把大量节点无约束地集中到同一个业务或局部
原运营商、调度类型、目标运营商、省份组合。
完整口径、算法、真实结果和限制见
[`业务容量感知推荐模型V6说明.md`](业务容量感知推荐模型V6说明.md)。
面向业务同学的简化解释见
[`业务容量推演说明.md`](业务容量推演说明.md)。

容量流量按 `peak95 -> analyzePeak95 -> buildBandwidth * peak95Ratio / 100` 依次回退；
七牛虚拟业务同一节点日只取一次有效流量，不累加重复虚拟行。默认以最近 14 个有效日的总流量
P75 除以 70% 目标利用率，分别得到业务整体、业务+原运营商、业务+原运营商+调度类型+目标运营商、
业务+省份+原运营商+调度类型+目标运营商的建设带宽上限。细层证据不足时回退到最近的有效父层。

本轮 31 天真实数据包含 505,615 条原始记录、279,442 个干净节点日，流量可用率 99.0%。
如果将可识别节点全部切到 V5 Top1，会有 4 个业务合计超出约 2,059.15 Gbps。462 个节点进入
全局优化，四层容量约束后保留 335 个切换复核方案，其中 Top1/Top2/Top3 分别为 307/17/11；
全部 1,454 个证据充分容量池均未超限。该结果不允许自动切业务。

运行与校验：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-v5.txt

.venv/bin/python v6_capacity_aware_allocation.py build \
  --output-dir recent_month_large_mainstream_v6_capacity

.venv/bin/python v6_capacity_aware_allocation.py check \
  --output-dir recent_month_large_mainstream_v6_capacity

.venv/bin/python -m unittest -v \
  test_v3_daily_business_training.py \
  test_v6_capacity_aware_allocation.py
```

前端报告：

```text
recent_month_large_mainstream_v6_capacity/v6_capacity_report.html
recent_month_large_mainstream_v6_capacity/v6_node_report.html
recent_month_large_mainstream_v6_capacity/v6_capacity_review_nodes.csv
recent_month_large_mainstream_v6_capacity/v6_capacity_blocked_nodes.csv
```
