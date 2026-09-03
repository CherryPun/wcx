# V1 节点最佳业务推荐闭环

## 目标

V1 先在本地历史产物上跑通完整闭环：

```text
节点画像 + 节点-业务 7 天收益
-> 综合目标
-> 推荐 Top3
-> 存量节点纠偏候选
```

决策口径：

```text
一个节点只交付一个业务
输出 Top3，由人结合原因与风险决策
综合目标 = 0.5 * 节点内归一化矿主成本分 + 0.5 * 节点内归一化平台利润分
平台利润 = cum_revenue_7d - cum_cost_7d
```

## 构建

```bash
python3 v1_recommendation_pipeline.py build
```

生成文件：

```text
v1_training_pairs.csv
v1_business_score_model.json
v1_node_recommendations.csv
v1_existing_node_correction_candidates.csv
v1_model_metrics.json
v1_business_name_map.csv
v1_business_risk_profile.csv
```

如果已有真实当前业务表：

```bash
python3 v1_recommendation_pipeline.py build \
  --current-business current_business.csv
```

如果已有外部业务名称映射：

```bash
python3 v1_recommendation_pipeline.py build \
  --business-map business_map.csv
```

小样本 smoke build：

```bash
python3 v1_recommendation_pipeline.py build \
  --limit-nodes 500 \
  --output-dir /tmp/machine_test_v1_smoke \
  --min-business-support 3
```

## 校验

```bash
python3 v1_recommendation_pipeline.py check
```

或运行单元 smoke test：

```bash
python3 -m unittest -v test_v1_recommendation_pipeline.py
```

## 单节点推荐

```bash
python3 v1_recommendation_pipeline.py recommend-node <node_id>
```

JSON 输出：

```bash
python3 v1_recommendation_pipeline.py recommend-node <node_id> --json
```

默认口径：

```text
如果该节点在 multibusiness_outcomes.csv 里已有节点-业务收益记录：
  返回 observed_node_outcome，即直接按该节点历史 7 天综合收益排序 Top3
如果没有收益记录：
  返回 model_segment_fallback，即按省份/运营商/资源类型等分段模型推荐 Top3
```

强制只看模型兜底：

```bash
python3 v1_recommendation_pipeline.py recommend-node <node_id> --model-only --json
```

临时补业务名映射：

```bash
python3 v1_recommendation_pipeline.py recommend-node <node_id> \
  --business-map business_map.csv \
  --json
```

多条件查询：

```bash
python3 v1_recommendation_pipeline.py recommend-filter \
  --where province=江苏 \
  --where isp=电信 \
  --where resourcetype=aggregation \
  --limit 20
```

## 运营审计入口

单节点完整审计：

```bash
python3 v1_recommendation_pipeline.py audit-node <node_id>
```

JSON 输出：

```bash
python3 v1_recommendation_pipeline.py audit-node <node_id> --json
```

返回内容：

```text
profile: 节点关键画像和网络质量
current_business: 当前业务、来源、置信度、当前分数
recommendation.top3: 推荐 Top3、分数、收益分项、原因
decision: 建议动作、分差、原因
risk: 风险等级、风险原因、缺失字段
```

拉取存量纠偏候选：

```bash
python3 v1_recommendation_pipeline.py correction-candidates \
  --action review_switch_candidate \
  --limit 50
```

只看高风险：

```bash
python3 v1_recommendation_pipeline.py correction-candidates \
  --risk-level high \
  --limit 50
```

## 当前业务接入

如果从 Superset `test.node_day_ops_wide_full` 导出最近 7 天宽表 CSV，可以先转成标准当前业务表：

```bash
python3 v1_recommendation_pipeline.py derive-current-business node_day_ops_wide.csv \
  --output current_business.csv \
  --lookback-days 7 \
  --min-active-days 3
```

推断规则：

```text
1. 优先过滤 state=online
2. 过滤 customerId 为空或 <=0
3. 每个节点按业务聚合最近 N 天
4. 优先选择出现天数最多的业务
5. 再按最近日期、peak95 汇总、收入汇总排序
```

标准当前业务表字段：

```text
node_id
current_business
current_business_name
current_business_day
current_business_source
current_business_confidence
```

然后重新构建：

```bash
python3 v1_recommendation_pipeline.py build \
  --current-business current_business.csv
```

如果 `current_business.csv` 只覆盖部分节点，已覆盖节点优先使用外部当前业务，未覆盖节点会回退到历史推断口径。

## 业务名称映射

从现有收益表导出已知业务名称：

```bash
python3 v1_recommendation_pipeline.py export-business-map \
  --output v1_business_name_map.csv
```

当前已通过 Superset 宽表补到 75 个业务 ID 的有效名称。仍缺失的业务 ID 已输出到：

```text
v1_business_name_missing_after_superset.csv
```

缺失的业务名需要业务维表或人工补一张映射表，字段可用：

```text
business,business_name
```

也兼容：

```text
customerId,customerName
```

## 主要输出

`v1_node_recommendations.csv`：

```text
node_id
recommendation_mode
combined_top1 / combined_top2 / combined_top3
combined_score_top1 / combined_score_top2 / combined_score_top3
recommended_miner_score_top1 / recommended_operator_score_top1
recommended_cum_cost_7d_top1 / recommended_cum_revenue_7d_top1 / recommended_cum_profit_7d_top1
recommended_support_top1
recommendation_reason_top1
combined_top3_detail
risk_level
risk_reasons
missing_profile_field_count
missing_profile_fields
```

`v1_existing_node_correction_candidates.csv`：

```text
node_id
current_business
current_business_name
current_business_source
current_business_confidence
combined_top1 / combined_top2 / combined_top3
current_eq_top1
current_in_top3
current_business_score
current_business_score_source
recommended_minus_current_score
suggested_action
reason_summary
risk_level
risk_reasons
```

`v1_business_risk_profile.csv`：

```text
business
business_name
support_nodes
combined_score_mean
cum_cost_7d_mean / cum_revenue_7d_mean / cum_profit_7d_mean
关键资源/质量字段的 p05/p50/p90/p95/max
```

用途：

```text
作为业务级软准入和风险规则的历史画像。
例如某业务历史最佳节点的 quality_retransrate_p95 很低，
而待推荐节点重传明显超过该范围，就应标记人工复核风险。
```

`suggested_action` 口径：

```text
keep_top1: 当前业务等于 Top1
keep_in_top3_review: 当前业务在 Top3 内，可人工复核是否保持
review_switch_candidate: 当前业务不在 Top3，且 Top1 分数优势达到阈值
review_current_business_not_in_model: 当前业务不在候选模型或收益表不可评分
current_business_unknown: 当前业务代理为空
observe: 当前不在 Top3，但分差未达到切换阈值
```

## 当前业务口径

当前业务按历史口径定义。V1 纠偏清单默认使用：

```text
每个节点 business_online_day 最新的业务
```

输出里会标记：

```text
current_business_source = historical_from_multibusiness_outcomes_latest_business_online_day
current_business_confidence = historical_inferred
```

如果后续要用 Superset 宽表最近 3/7 天重算当前业务，可以通过 `derive-current-business` 生成 `current_business.csv` 后传给 `build --current-business`。

## V1 模型说明

当前产物里有两层逻辑：

```text
observed_node_outcome: 存量节点优先直接使用该节点已观测的节点-业务 7 天收益 Top3
model_segment_fallback: 新节点或无收益节点使用透明分段模型兜底
```

当前 `v1_business_score_model.json` 是透明 baseline：

```text
smoothed_segment_business_score
```

它按省份、运营商、资源类型等分段统计业务综合收益，并用全局业务均值平滑。它不是最终多算法版本，作用是先把训练样本、综合目标、Top3 推荐、纠偏输出和测试闭环跑通。

`v1_model_metrics.json` 里的 hit@1/hit@3 评估的是 `model_segment_fallback` 的泛化能力，不代表 `observed_node_outcome` 口径。当前模型兜底能力偏弱，生产推荐前需要继续引入更多特征和算法。

后续模型可以在同一训练样本上替换为：

```text
ExtraTrees / RandomForest / LightGBM / CatBoost / 排序模型
```

## 限制

1. 当前数据是约 5.08% 节点采样，不是全量生产数据。
2. 当前业务默认使用历史推断口径，不等于线上实时部署业务。
3. V1 baseline 指标不代表最终模型效果。
4. 如果业务从未在相似分段出现，推荐会回退到全局业务分数。
5. 风险字段会提示缺字段、高重传、高丢包、高 RTT、高磁盘/内存/CPU/IO 等风险，但暂不做硬阻断。
6. 部分业务 id 缺少业务名称映射，V1 会清理 `\N` 占位符，但不能自动补出名称。
