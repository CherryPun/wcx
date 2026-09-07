# wcx

给矿主节点的业务推荐：候选只从**相似节点真实跑过的业务**中取，按**矿主预测 7 天结算**为主序，平台利润与容量只做软约束，输出 Top1~3 与预测金额，定位为**候选/参考，不承诺最赚**。

数据口径与 HRR 同源（7 天累计、5% 抽样、满 7 天账），验证使用时间外切分，与 yzx/RJ 的目标与产物均不同。

## 目录关系

```text
01_temporal_validation/   时间外切分 + 冷启动基线诊断（结论的事实依据）
02_main_recommendation/   主版推荐（v1：direct 成本）+ 结果 CSV/指标 + HTML/分布图生成脚本
03_variant_ablation/      v2(log1p) 与带宽辅助/金额误差消融（对照，非主版）
04_report/                报告、交付说明、关键数字口径稿、交付索引、容量口径说明
05_shared_data/           数据字段说明（DATA_README；成品 CSV 不入库，见 .gitignore）
06_html/                  主版全节点推荐总览 + 推荐分布可视化（纯 SVG，离线可看）
requirements.txt          运行依赖
.gitignore                数据/临时产物默认排除
run_verify_all.py         一键复现校验（重跑 4 步并核对关键数字）
```

### 严谨性说明

- 本仓库**不含任何他人分支渲染的结果文件**。仓库内所有 CSV/JSON/HTML
  均由本仓库脚本读取 `05_shared_data` 生成，可用 `run_verify_all.py` 复现。
- `01`/`02`/`03` 中的 JSON 即各步骤的可复现指标快照；`_rerun/`（运行产物）不入库。
- 无效或实验性方案（放大惩罚、单位带宽参照、受控容量模拟等）的**结论**
  保留在 `04_report`；相应脚本与中间产物已移出仓库并归档，避免混入无出处的数据。

## 主结果（测试 = 时间外较晚上线的新节点）

| 指标 | 数值 |
|---|---:|
| 真实最优业务进 Top1 | 35.9% |
| 真实最优业务进 Top3 | 65.4% |
| Top1 平台利润为正 | 100% |
| 矿主结算预测 WAPE | ≈0.82 |

详细口径见 `04_report/关键数字口径稿.md`。

> **预测口径声明**：推荐金额是模型对“换到该业务后、按同类节点典型给量”的 7 天预测，
> 模型输入**不含该节点平时的实际带宽利用率/可达量**。对当前给量不足、空载或刚切换业务的节点，
> 预测结算可能明显高于实际到手（现有“当前在跑(近7天实际)”列可与预测对照）；请按“可争取上限”理解，不承诺兑现。

## 一键复现校验

```bash
python run_verify_all.py
```

依次重跑 01 诊断、01 时间外、02 主版、HTML 生成，并校验关键数字是否落在容差内。输出 PASS / FAIL。

## 运行说明

- 主版、01 诊断、03 消融脚本均已改为**在交付目录直接运行**：数据读 `05_shared_data`，结果写各模块 `_rerun/`，不覆盖正式产物。
- 直接运行（本机 Anaconda Python）：

```bash
python 01_temporal_validation/stage_p0_temporal.py
python 01_temporal_validation/stage_diag_coldstart.py
python 02_main_recommendation/stage_e2e.py
python 02_main_recommendation/build_final_html.py
python 03_variant_ablation/stage_bw_ab.py
python 03_variant_ablation/stage_cost_improve.py
```

- `03_variant_ablation/stage_e2e_v2.py`（v2 对照）可 import，完整重跑耗时较长；结论已固化在 `04_report/关键数字口径稿.md`。
- 依赖：pandas / numpy / scipy / scikit-learn / xgboost。

## 数据来源与重建

数据与 GitHub 三个分支**不同源、不同窗口，不可混用**。均为本机从内网 Superset 现拉并处理，历史范围 2026-03-16 起、约 5% 节点抽样（后缀 00–0c）、满 7 天账。字段口径见 `05_shared_data/DATA_README.md`。

复现需要的 4 份数据：

| 文件 | 来源 |
|---|---|
| `05_shared_data/nodes_attr_filled.csv` | jarvis 节点画像（node_analysis_data / node_join / dial_acc），上线前可用字段 + 缺失回填 |
| `05_shared_data/outcomes_7d_named.csv` | Superset 宽表按业务首次出现日起 7 天合计（满 7 天行） |
| `05_shared_data/business_capacity_summary.csv` | 近 7 天日 95 流量 + 目标利用率 70% 反推容量 |
| `05_shared_data/bw_daily_7d.csv` | 近 7 天宽表日 peak95 |

> 大文件默认不入库（见 `.gitignore`）。若数据目录为空，需在可访问内网 Superset 的环境下重建；
> 生成链路线索：宽表金额与带宽取 `test.node_day_ops_wide_full`，节点画像取 `jarvis.*`（详见各 fetch/build 脚本说明及 `04_report/方案定稿报告_*.md`）。

## 复现路径（克隆后）

1. 安装依赖：`pip install -r requirements.txt`
2. 准备数据（见上）：放置 `nodes_attr_filled.csv`、`outcomes_7d_named.csv`、
   `business_capacity_summary.csv`、`bw_daily_7d.csv` 到 `05_shared_data/`
3. 一键校验：`python run_verify_all.py`（重跑 4 步并核对关键数字，PASS/FAIL）
4. 或分步运行：见下方“运行说明”

## 数据

- 主版与诊断使用 `05_shared_data` 下 4 份成品数据；字段说明见 `05_shared_data/DATA_README.md`。
- 数据来源为 Superset 现拉宽表 + jarvis 节点画像回填，只在本机使用，不 push。
