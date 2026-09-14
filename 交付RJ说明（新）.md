# 交付 RJ 说明（新）

> 交付方：wcx2 分支（基于 RJ `e541ac2`）。用途：供 RJ 评审是否采纳本代改动与结论。

## 1. 建议采纳的改动（核心 1 项 + 3 个开关）

**核心：V5 训练启用近端时效权重，并固定确定性线程。**

| 项 | 值 | 说明 |
|---|---|---|
| `V5_RECENCY_HALF_LIFE` | 默认 **7** | 半衰期 7 天 |
| `V5_RECENCY_WEIGHT_CAP` | 默认 **2.0** | 权重上限（单线程重扫的帕累托点；旧 1.5 集中度过高已弃） |
| `V5_NTHREAD` | 默认 **1** | **确定性**（RJ 原为 6，同配置两次 Top1 差异 573/2288） |
| 其他开关 | `V5_TREND`/`V5_BUSINESS_BALANCE`/`V5_CALIB` 默认 **关** | 已实验且被否决，保留接口 |

- **回退**：`$env:V5_RECENCY_HALF_LIFE="0"` → 回到 RJ 基线行为；
- **若 RJ 暂不接受默认变更**：可先保持默认关，改为"按需启用"（代码已支持）。

## 2. 证据（单线程；同窗口 2026-08-10~09-09；同点全量 current）

**V5 时间外（bw ≥ 500 Mbps，默认评估口径；n=33,484）：**
| 指标 | RJ 基线 | 采用配置 | 变化 |
|---|---:|---:|---|
| 矿主 RMSE / R² | 0.03537 / 0.17862 | **0.03410 / 0.23697** | RMSE −3.6%，R² **+0.058** |
| 平台 RMSE / R² | 0.02823 / 0.04435 | 0.02835 / 0.03627 | RMSE +0.4%，R² −0.008 |
| 90% 区间覆盖（矿主/平台） | — | 0.895 / 0.904 | 达标 |
| Top1 集中度 max / HHI | 0.5044 / 0.3550 | **0.4178 / 0.2520** | 均下降 |

**同窗口全样本口径（`--min-build-bandwidth 0`）**：矿主 RMSE 0.27810→0.20020（−28%）、R² 0.01965→0.02532；平台 RMSE 0.24255→0.19225、R² 0.00218→0.00690。

> ⚠️ **口径解读（重要）**：全样本的"−28%"**主要由小带宽尾部样本驱动**（分母小、单位值极端）。
> 换成大节点口径（bw≥500）后，采用配置的收益主要是**矿主 R² 提升（+0.058）与集中度下降**，
> RMSE 仅 −3.6%，且**平台侧略降**。建议以 bw≥500 口径为准，并把收益表述改为
> "提高大节点上的解释度与降低集中度"，而非"RMSE 降 28%"。

**V6 容量：**
| 口径 | 基线 | 采用配置 |
|---|---:|---:|
| 规划切换 / 拦截 | 94 / 63 | 206 / 87 |
| 全切溢出 | 6,265 Gbps | **4,139 Gbps（-34%）** |
| planned 溢出 | 0 | 0 |

**工程修复**：V5 非确定性（同配置 Top1 差 573）已修复；单测 7/7 通过（含新增 3 个时效权重单测）。

> ⚠️ **指标口径（重要）**：上表为**全样本**（含建设带宽低至 50 Mbps 的节点），小分母极端值会主导 SSE。
> 按建设带宽分层重算（见 `指标口径对账（新）.md`）：**bw ≥ 500 Mbps** 时矿主 RMSE/R² = 0.0341/0.237、平台 = 0.0284/0.0363，
> 与 RJ 文档 §10.3（矿主 0.0470/0.2825、平台 0.0492/0.0263）同量级。**建议 RJ 侧评估统一声明带宽下限**。

> ⚠️ **压测特征**：白名单含 `packet_loss_satisfaction_bucket`，但本轮回填覆盖率 = **0.0**
> （`v5_training_summary.json → latest_pressure_training_coverage`），故**本次指标不含静态压测泄漏**。
> 若启用该特征（RJ 口径按"最新一次 netbench"复用整窗），才会引入未来信息风险（`节点业务推荐模型V5全流程说明.md` §5.7）。

## 3. 复现所需数据（**未随仓库分发**）

| 数据 | 来源 | 说明 |
|---|---|---|
| 全库账（满7天） | Superset `yzh-starrocks.test.node_day_ops_wide_full` | `fd` 上限≈2026-08-30，7 天窗口累计到宽表最新 |
| 节点画像 | 宽表行属性构造（province/isp/deliverytype/resourcetype/bw）；jarvis 部分缺失 | 见 `EXEC_expansion_runbook.md` |
| 账/画像/压力/节点 |
| 训练对（pairs） | 由 V3 日粒度构建 | `v1_training_pairs_large_mainstream_recent_1m.csv` |
| 压力画像 | `fetch_latest_node_pressure.py`（jarvis netbench） | `latest_node_pressure_profiles.csv` |
| 当前节点/当前业务 | 现网扫描 + 同点全量 current | `current_business_同点全量（新）.csv` |
| allowlist / 业务名映射 | `mainstream_business_allowlist.csv`、`v1_business_name_map_enriched.csv` | RJ 既有 |

复现链：`build_recent_large_training → build_mainstream_large_training → build_v3_daily_business_training → v5_hybrid_recommendation → v6_capacity_aware_allocation`。

## 4. 变更文件

- 修改（不能改名，见 `修改文件清单（改）.md`）：`v5_hybrid_recommendation.py`、`test_v5_hybrid_recommendation.py`、`README.md`、`.gitignore`；
- 新增（（新））：见 `正式交付索引（新）.md`。

## 5. 结论
- **结论与建议改动可交付评审**。
