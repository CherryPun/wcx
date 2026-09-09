# v2_data —— V2 一代工作区（自包含）

> 交付口径：**large = `V2_POOL=nonant`（node_id 非 ant）+ `V2_MERGE=qiniu_name`（默认开，诊断映射）**；无 `10000074` 例外。
> 全库终版指标（时间外 n=2,994）：家族粒 hit1 21.0 / hit3 51.7，金额 WAPE 0.41 / R² 0.68（5% 参考：ID 18.3/48.8、家族 25.6/51.2, n=82）。ant 弱产品单独页。
> 详细交付基线见 `../V2执行结果与定稿.md` 顶部“交付基线（全库扩样定稿）”。

## 运行环境（扩样/复核统一用这套，勿再调参）

```powershell
$env:V2_POOL="nonant"; $env:V2_MERGE="qiniu_name"   # merge 也可不设（nonant 默认开）
$env:V2_SKIP_OUT="1"                                  # 扫描模式（metrics 仍写 _rerun/e2e_metrics.json）
python stage_e2e.py                                   # 产物在 _rerun/
```

- K：沿用默认 15（`V2_KNN_K` 不设）；MIN_SUPPORT=20；早停=训练内 15% valid；容量不进排序；盒子门禁在 nonant 下默认生效。
- 数据路径 env：`V2_ATTRS`（画像 csv）、`V2_OUTCOMES`（账 csv，5% 默认读仓库根 `05_shared_data/`）。
- 全库扩样只需把 `V2_OUTCOMES/V2_ATTRS` 指向新拉的“全库非 ant 满 7 天账/画像”，其它配置零改动。

## 关闭归并（对照用）

```powershell
$env:V2_MERGE="0"; python stage_e2e.py   # 输出 ID 粒（无家族合成）
```

## 跑法文件

- `EXEC_expansion_runbook.md`：全库扩样第一步与四项复核的执行单（数据就绪后按步骤复现）。
