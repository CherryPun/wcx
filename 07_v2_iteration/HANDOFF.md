# 会话交接说明（给新窗口助手）

> 用途：直接接手上一个超长会话的工作。先读本文件 + `07_v2_iteration/README.md` + `V2执行结果与定稿.md` 顶部“交付基线”。

## 项目一句话
给大节点做业务推荐（v1 主版 → V2 全库 large），5% 上算法已冻结，全库扩样复核后定稿。仓库 `CherryPun/wcx`（个人，仅 master，不建分支）。

## 已定稿（冻结，勿再调参）
- large = `V2_POOL=nonant`（node 非 ant）+ `V2_MERGE=qiniu_name`（默认开=诊断映射；sign 探针确认无可再合并）；`V2_GATE` 在 nonant 默认开启（盒子业务门禁）；无 `10000074` 例外（全库 support=1,041≥20 已自动进池）。
- K=15（冻结，勿切 10）；MIN_SUPPORT=20；早停=训练内 15% valid（不用测试集）；容量罚分不进排序；ant 弱产品单独页。
- 数字（clean 全库，时间外）：家族粒 hit1 **21.0 / hit3 51.7**，金额 WAPE **0.41 / R² 0.68**，n=2,994（训练 11,972 / 画像交集 14,966）。
- 参考/对照（勿当交付）：v1 主版混合 35.9/65.4/WAPE0.82（ant 占 96% 主导，作废口径）；5% large 25.6/51.2（n=82）；ant 独立约 8%。

## 文件地图
- `07_v2_iteration/README.md`：交付总览（最终数字/边界）。
- `V2执行结果与定稿.md`：顶部=交付基线；1–9 节=5% 历史执行记录；第 8/10 节已改定稿态。
- `V2最优方案.md`：顶部=全库定稿；正文=5% 收敛过程（已标注历史）。
- `outputs/`：全节点_large_fullpool.html（含当前在跑列）、全节点_ant.html、推荐分布_大节点/小节点 html、large_review_fullpool.csv（n=2,994；含家族一致性列 token+具体 ID）、exp_metrics.json（clean）、B_large_amplification.txt（全库 n=2,994）。
- `v2_data/`：stage_e2e.py（env：V2_POOL/MERGE/GATE/OUTCOMES/ATTRS/ORIG/SKIP_OUT/CUT）；README（默认配置）；EXEC_expansion_runbook.md（数据重建/复现，含第 0 步规模 SQL）；各 build_*.py（当前列页/分布/复核）与分析脚本。
- 数据：5% 资产在 `05_shared_data/`（不入库）；全库账 `_rerun/exp_outcomes_clean.csv`、画像 `exp_attrs.csv`（不入库，重建看 runbook）。Superset 凭据在桌面 `superset账号.txt`；查询器 `skills/superset-sql-query/common/run_superset_sql.py`（python=D:\Users\anaconda3\python.exe）。

## 已知项（勿当 bug 反复调查）
- Top2/3 在候选<3 的节点会缺（页面显示 —/nan），记录在“已知展示项”，不修复。
- `exp_metrics.json` top1_distribution 为内部家族 token（FQCDNV/FQTZV），家族粒指标所需；对外 HTML 与复核表 top1_business_id 已映射具体代表 ID。
- `run_verify_all.py` 只复现 v1；V2 验收走 runbook + exp_metrics。
- 文件命名/数字均已统一到 clean 版；如重跑数据后需重生成 outputs 页（脚本见 v2_data）。

## 下一步候选（需要新会话时）
1. （可选）ID 粒全库 = 一次 `V2_MERGE=0` 重跑，补 ID 粒对外行。
2. 产品/展示打磨（如当前在跑低量标注、复核表浏览器化）。
3. 后续实验（非本轮）：可达量校准、日粒度标签口径——需先扩数据/另立专项。
4. Git：只提交到 master；提交范围=脚本+md+outputs 小件；不交 log/_rerun/大 CSV。
