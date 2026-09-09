# 扩样执行单（全库非 ant 满 7 天账复核；数据重建与复现步骤）

> **状态：已执行并定稿（2026-09-08）。** 全库扩样已跑完（全库非 ant 25,860→画像交集 14,966→时间外 n=2,994），
> 终版数字见 `../README.md` 与 `../V2执行结果与定稿.md`。本文件作为复现/存档执行单保留；其中“待办”表述不再代表当前待办。

> 背景：5% 抽样模型已冻结（分池+七牛按名归并+无 74 例外）。扩样只验证“全库是不是同一回事”，不是改模型。
> 重要：**不要用“当日活跃 12,845”当规模**——那是口径错误。必须用与 5% 的 409 同一把尺子（满 7 天账）。

## 0) 第一步先数规模（跑这一条再决定要不要继续）

用与 `outcomes_7d_named.csv` 一致的生成口径数“全库非 ant 满 7 天账节点数”：

```sql
-- node×customer 的“业务首次出现日”= 该 pair 在宽表的最早 day
WITH first_day AS (
  SELECT nodeId, customerId, MIN(day) AS fd
  FROM node_day_ops_wide_full
  WHERE nodeId NOT LIKE 'ant%'
  GROUP BY nodeId, customerId
),
win AS (
  SELECT f.nodeId, f.customerId, f.fd,
         COUNT(DISTINCT w.day) AS ndays,
         SUM(COALESCE(w.cost_finalAmount,0))   AS cost7,
         SUM(COALESCE(w.revenue_finalAmount,0)) AS rev7
  FROM first_day f
  JOIN node_day_ops_wide_full w
    ON w.nodeId=f.nodeId AND w.customerId=f.customerId
   AND w.day BETWEEN f.fd AND DATE_ADD(f.fd, INTERVAL 6 DAY)  -- StarRocks 亦支持 days_add(fd, 6) 备选
  GROUP BY f.nodeId, f.customerId, f.fd
)
SELECT COUNT(*) AS full7_pairs, COUNT(DISTINCT nodeId) AS full7_nodes,
       COUNT(DISTINCT customerId) AS customers
FROM win WHERE ndays >= 7;
```

- **实测规模（2026-09-08 已跑）**：full7_pairs **49,868** / full7_nodes **25,860** / customers **141**（≈5% 的 409 的 63×）。
- 预期备注：比初估“几千”大得多，属正常量级；25,860 节点的 jarvis 画像需**新拉**（主版 `nodes_attr_filled` 只覆盖 5% 样本、其中非 ant 仅 409），画像获取与 as-of 对齐是全库扩样的主要成本项。

## 1) 账（对齐 5% 的 outcomes）

- 按上 SQL 导出全库非 ant 满 7 天对 → `V2_OUTCOMES` 指向的 csv，列名对齐 `outcomes_7d_named.csv`：node_id / business(=customerId) / business_name(可空) / business_online_day(=fd) / online_day(节点引入日，另取) / outcome_days(>=7) / cum_cost_7d / cum_revenue_7d。
- 节点级 `online_day`：取该 node 全历史最早 day（控制面/宽表均可）。

## 2) 画像

- 这批 node_id 的 jarvis 快照（as-of 能做就做；做不到用现回填逻辑），记录关键字段覆盖率后再跑；画像缺失节点数要报。

## 3) 环境（零改动）

```powershell
$env:V2_POOL="nonant"; $env:V2_MERGE="qiniu_name"
$env:V2_ATTRS="<全库非ant画像.csv>"; $env:V2_OUTCOMES="<全库非ant满7账.csv>"
python stage_e2e.py
```
- 不调 K / MIN_SUPPORT / 早停 / 门禁；无 74 例外。

## 4) 四项复核表（同一 82 的尺子换成全库时间外）

1. 家族粒 hit1/3 = 21.0/51.7（n=2,994；5% 参考 25.6/51.2，已脱离 ±5pp 判为抽样向上偏）；
2. 腾讯汇聚 / 快手严选放大比是否仍 ≥3×；
3. 七牛虚拟系真优进 Top3 = 72.7%（sign 校验无可再合并，记录为弱项）；
4. `10000074` 全库满7 node=1,041（训练 907）≥20 → 自动进池，无需例外。

## 5) 翻车规则（不回调参）

| 情形 | 动作 |
|---|---|
| 四项接近 | 结案，交付数字换成全库，配置不动 |
| 仅 2 仍 ≥3× 且测试集 >150 | 才允许一轮排序实验（同节点单位带宽重排或轻先验），一轮试完即停 |
| 3 明显掉 | 名字归并不够 → 用宽表 `signName/virtualCustomersName` 补一张诊断映射，再跑一次，不再迭代 |
| 4 ≥20 | 74 自动进池，不需例外 |
| 家族粒 hit1 21.0 | ≥18，未触发改口；若复跑跌破 18 再对外改口 |

## 产出（回填到仓库再谈提交）

- 全库两行数字：ID 粒 ｜ 家族粒（必写 n）
- 更新两份 md（第 8/10 节），重生成 `outputs/全节点_large_fullpool.html` 与 `推荐分布_大节点_large.html`（全库）与 large 专用复核表（当前在跑 vs Top1/3 + 家族是否一致）；ant 页保持弱产品声明。
- Git 只交：脚本 + 两份 md + 小 json + 两页 HTML（不交 log / 大 CSV）。
