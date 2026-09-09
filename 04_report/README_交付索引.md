# wcx 交付索引

> 本索引说明仓库目录结构与各文件职责：代码按模块组织，数据集中在 `05_shared_data`，报告在 `04_report`。

## 目录

```
wcx/
├── README.md                      总览与运行说明
├── 01_temporal_validation/        时间外切分 + 冷启动基线诊断
│   ├── stage_p0_temporal.py       时间外 8/2 切分验证脚本
│   ├── stage_diag_coldstart.py    冷启动五基线诊断脚本
│   ├── final_top3_temporal_metrics.json
│   └── coldstart_baseline_metrics.json
├── 02_main_recommendation/        主版（v1）
│   ├── stage_e2e.py               主版端到端（相似节点候选 + 矿主结算主序）
│   ├── final_top3_e2e.csv         5171 节点 Top1~3 + 矿主结算/平台利润
│   ├── e2e_metrics.json           主版指标
│   ├── build_final_html.py        HTML 生成脚本（含“当前在跑”实际对照列）
│   └── build_distribution_chart.py 推荐分布条形/环状图生成
├── 03_variant_ablation/           变体与消融
│   ├── stage_e2e_v2.py            v2(log1p+neighbor) 对照
│   ├── stage_cost_improve.py      金额误差三变体拆解
│   ├── stage_bw_ab.py             带宽辅助分 开/关 对比
│   ├── final_top3_e2e_v2.csv / e2e_v2_metrics.json
│   ├── bw_aux_ab_metrics.json
│   └── cost_improve_metrics.json
├── 04_report/                     报告与交付文档
│   ├── README_交付索引.md
│   ├── 方案定稿报告_矿主优先候选推荐.md
│   ├── 最终交付说明_矿主优先推荐.md
│   ├── 关键数字口径稿.md
│   ├── 项目过程说明.md
│   ├── 各文件作用说明.md
│   ├── 容量口径限制与正确做法.md
│   └── 数据体量与属性覆盖说明.md
├── 05_shared_data/                主版与诊断共用成品数据
│   ├── nodes_attr_filled.csv
│   ├── outcomes_7d_named.csv
│   ├── business_capacity_summary.csv
│   ├── bw_daily_7d.csv
│   └── DATA_README.md             字段口径说明
├── 06_html/                       主版全节点推荐总览
│   ├── 全节点推荐_最终版.html
│   └── 推荐业务分布_可视化.html    横向条形（Top10 业务）+ 饼图（Top10 占比 87%）
├── 07_v2_iteration/               V2 交付：README.md（入口/结果）+ outputs（终版页/复核）+ v2_data
├── requirements.txt              运行依赖
├── .gitignore                     数据/临时产物默认排除
└── run_verify_all.py              一键复现校验（重跑 4 步并核对关键数字）
```

## 已做但无效的改进

- **放大惩罚（stage_e2e_ampfix.py）**：按训练期“推荐占比/真实最优占比”对高放大业务扣分，
  测试命中 Top1 35.9%→32.4%、Top3 65.4%→47.0%，集中度几乎未降（Top1 占比 67%→66%）。
  结论：数据中高占比业务本身是真实大头，惩罚只伤命中不散扎堆，**不作为主版**。

## 不复用的（弃用/过渡产物，不在交付内）

- 早期/对照候选与重排脚本：`score_prefer_history*`、`score_tight*`、`score_all_candidates*`、
  `run_cost_balance_experiment*`、`v4_*`、`recommendations_before/after*`、`add_current_business*`
- 旧 HTML/查询服务：`serve_live_lookup.py`、`node_lookup*`、`overview.html`、`node_search*`、`weekly_report*`
- 中间阶段：`stage2_build_samples*`、`stage3_dual_tower*`（双塔召回在冷启动下未被证明，仅研究过程）

## 运行说明

- 主版与 01 诊断脚本已改成在交付目录直接运行：数据读 `05_shared_data`，结果写各模块 `_rerun/`，不覆盖正式产物。
- 已实测可复现：`stage_e2e.py`（02）、`stage_p0_temporal.py` / `stage_diag_coldstart.py`（01）输出与正式数字一致。
- `03_variant_ablation` 与 `02` 的 HTML 生成脚本仍引用原工作区路径（研究对照用），复现前请把头部 `TASK`/`DATA` 指向本目录。

