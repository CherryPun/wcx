# 多业务节点最佳业务推荐

本项目使用真实节点业务数据，基于节点上线前可确定的固有属性，预测节点上线后 7 天内更适合的业务：

- 矿主最佳业务：7 天累计成本最高的业务。
- 运营最佳业务：7 天累计收入减累计成本最高的业务。

## 数据口径

- 业务数据：Superset 数据库 `yzh-starrocks` 的 `test.node_day_ops_wide_full`。
- 节点属性：Superset 数据库 `jf-jarvis` 的 `jarvis.node_analysis_data`、`jarvis.node_join` 和 `jarvis.dial_acc`。
- 运行健康指标：VictoriaMetrics/Prometheus，用于健康准入，不进入收益预测模型。
- 历史范围：`2026-03-16` 至 `2026-08-04`。
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
- `multibusiness_outcomes.csv`：节点-业务 7 天成本、收入、利润结果。
- `multibusiness_recommendations.csv`：训练集/测试集推荐结果和健康阻断理由。
- `multibusiness_eligible_pairs.csv`：合格节点-业务首次出现明细。
- `multibusiness_prometheus_probe.json`：Prometheus 指标探查结果。

# machine-test
