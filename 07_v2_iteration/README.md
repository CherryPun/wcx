# V2（全库扩样定稿）交付总览

> 定位：主版 v1 之后的一代，交付 **large = 非 ant（大节点/专线+汇聚）推荐**。5% 上算法冻结，本轮为全库扩样复核后的定稿。
> 产品：给大节点推荐“同池相似节点真实跑过的业务”（候选/参考，不自动切）；矿主 7 天结算主序 + 平台利润软约束。

## 1. 最终结果（时间外，n=2,994；训练 11,972 / 画像交集 14,966）

| 指标 | 值 |
|---|---|
| 家族粒 hit@1 / @3 | **21.0% / 51.7%** |
| 金额 WAPE / R² | **0.41 / 0.68** |
| ID 粒（全库） | 未单独重跑（= `V2_MERGE=0` 一次可得）；5% 参考 18.3/48.8 |
| 候选池 / 空候选 | 61 / 3（≈0） |
| ant | 弱产品单独页，不参与 large |

- 交付基线细节与全部否决/结案记录见 `V2执行结果与定稿.md`（顶部为最新口径）。
- 配置：`V2_POOL=nonant` + `V2_MERGE=qiniu_name`（默认开，诊断映射，sign 探针确认无可再合并）＋门禁（盒子业务不进 large）＋无 `10000074` 例外。

## 2. 目录结构

```
07_v2_iteration/
├─ README.md                本文件（交付入口）
├─ README_V2迭代方案.md      V2 方案与执行顺序
├─ V2执行结果与定稿.md       交付基线/最终结果/结案记录
├─ V2最优方案.md             收敛建议
├─ outputs/                  终版可视化/复核/指标（可查看）
│  ├─ 全节点_large_fullpool.html
│  ├─ 全节点_ant.html
│  ├─ 推荐分布_大节点_large.html
│  ├─ 推荐分布_小节点_ant.html
│  ├─ large_review_fullpool.csv   （2,994 测试复核表）
│  ├─ exp_metrics.json
│  └─ B_large_amplification.txt
└─ v2_data/                 一代工作区（脚本+数据重建）
   ├─ stage_e2e.py           模型流水线（env 参数化，见 README）
   ├─ README.md              运行环境/默认配置
   ├─ EXEC_expansion_runbook.md  全库数据重建与四项复核执行单
   ├─ analyze_attr_coverage.py / build_fullpool.py / eval_layer.py 等（诊断/交付脚本）
   └─ _rerun/                运行产物（不入库）
```

## 3. 复现（数据就绪后）

```powershell
# 数据重建（含第 0 步规模 SQL）→ EXEC_expansion_runbook.md
# 指向全库账/画像后，零配置重跑：
$env:V2_POOL="nonant"; $env:V2_MERGE="qiniu_name"
$env:V2_OUTCOMES="<全库非ant满7账.csv>"; $env:V2_ATTRS="<全库非ant画像.csv>"
python v2_data/stage_e2e.py     # 产物写 v2_data/_rerun/，指标见 e2e_metrics.json
```

## 4. 口径与诚实边界

- 指标口径：家族粒命中按内部 token（FQCDNV/FQTZV）计算（`exp_metrics.json` 的 top1_distribution 含 token）；对外 HTML 与 `large_review_fullpool.csv` 均已映射为具体代表 ID（10000292/10000281），；复核表另保留家族一致性列 tb_fam/top1_fam（token，用于核对家族粒命中）。
- 一键校验：仓库 `run_verify_all.py` 仅复现 v1 历史口径；V2/全库验收走 `v2_data/EXEC_expansion_runbook.md`（数据重建）与 `v2_data/_rerun/exp_metrics.json`。


- “在跑=有账”不等于有量；ant 为弱产品（约 8%），勿与 large 混排；混训 36%、big 13.8% 不作交付。
- 画像 42% 缺 jarvis 字段以缺失处理；七牛虚拟为诊断归并（非官方绑定）；`10000074` 全库 support≥20 已自动进池。
- Git 仅跟踪：脚本 + md + outputs 小件（不交 log / _rerun / 大 CSV）。
