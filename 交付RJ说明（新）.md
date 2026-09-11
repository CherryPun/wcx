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

**V5 时间外：**
| 指标 | RJ 基线 | 采用配置 | 变化 |
|---|---:|---:|---|
| 矿主 RMSE / R² | 0.27810 / 0.01965 | **0.20020 / 0.02532** | RMSE -28% |
| 平台 RMSE / R² | 0.24255 / 0.00218 | **0.19225 / 0.00690** | RMSE -21% |
| Top1 集中度 max / HHI | 0.5044 / 0.3550 | **0.4178 / 0.2520** | 均下降 |

**V6 容量：**
| 口径 | 基线 | 采用配置 |
|---|---:|---:|
| 规划切换 / 拦截 | 94 / 63 | 206 / 87 |
| 全切溢出 | 6,265 Gbps | **4,139 Gbps（-34%）** |
| planned 溢出 | 0 | 0 |

**工程修复**：V5 非确定性（同配置 Top1 差 573）已修复；单测 7/7 通过（含新增 3 个时效权重单测）。

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
