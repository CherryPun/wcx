# 跨日期多业务节点最佳业务推荐重建报告

## 结论

本次使用宽表样本节点 26,623 个，节点级切分为训练 21,298 个、测试 5,325 个。训练集为拟合内评估，测试集为节点留出评估；不能把训练集指标当作泛化效果。
测试集临时排序矿主命中率@1为 17.77%，运营命中率@1为 18.45%；在同一健康约束下，矿主基线命中率为 10.43%，运营基线命中率为 12.06%。
相对健康约束基线的矿主差值 7.35%，p=0.0004998，95% CI [6.65%, 8.06%]；运营差值 6.39%，p=0.0004998，95% CI [5.73%, 7.06%]。两项差值均为正，且95% CI均未跨0，当前固定节点留出评估下呈现统计显著的正向预测提升；这不是因果验证，也不能替代健康指标完整后的上线前审核。

## 数据口径

- 宽表：`test.node_day_ops_wide_full`，历史范围 `2026-03-16` 至 `2026-08-04`。
- 筛选：`customerId > 0`，至少 3 个不同业务、业务首次出现日至少 3 个不同日期，且至少一条记录 `state='online'`；业务首次出现日不晚于 `2026-07-29`，保证7天窗口可计算。
- 样本：节点 ID 后缀 `00,01,02,03,04,05,06,07,08,09,0a,0b,0c`，约 5.08%；不是全量节点推断。
- 目标窗口：每个节点-业务从该业务首次出现日开始的 7 天，成本为 `cum_cost_7d`，运营目标为 `cum_revenue_7d - cum_cost_7d`。
- 窗口观测日数：最少 1 天，中位数 2 天，最多 7 天；其中 286,163 行少于 7 个观测日，缺失日按宽表没有记录处理为目标累计中的零贡献，不能解释为真实业务运行状态。
- 原始属性异常：无法读取的 `node_analysis_data` 分区为 [{'day': '20260627', 'hour': '05'}]；这些小时被排除，没有进行数值补填，节点若没有其他可用小时则对应属性保持缺失。
- 模型：推荐模型：ExtraTrees 多分类器，分别预测节点历史成本最佳业务和运营利润最佳业务；Ridge 仅作为金额回归辅助诊断。训练集为拟合内评估，测试集为节点级留出评估。CPU、内存、磁盘、网络质量等 Superset 属性按节点首次上线日取当天最新小时；当前只能取得的 VM SMART/ZFS/Prometheus 指标不进入历史收益模型，保留用于运行时健康准入。
- 推荐分类目标：矿主为训练节点历史 `cum_cost_7d` 最大业务，运营为训练节点历史 `cum_profit_7d` 最大业务；CSV 中 `miner_model_probability`/`operator_model_probability` 是分类器概率，不是金额。Ridge 金额预测仅用于辅助 R²/误差诊断。
- 原始结果 294,836 行，业务数 135；候选业务 85 个，训练支持阈值为 30 行。

## 使用字段

类别字段：vendorid、deliverytype、resourcetype、dialtype、nattype、scheduleisps、regsource、customermode、province、isp、city、device_type、arch_type、isvm、qoskiller_status、os、arch、node_state、node_stage、node_status、hardwaretype、node_manufacturer、node_model、node_analysis_stage、idc_id、idc_name、analysis_nodecustmertype、analysis_nodedeliverytype、analysis_tcpnattype、analysis_udpnattype、analysis_cgroupversion、analysis_lastdeploystate、analysis_lastdeployscenario、analysis_cooperationtype、analysis_supply_side_delivery_type、analysis_isroot、analysis_issupportipv6、analysis_upnpstate、dial_is_normal、dial_ipv6_enable、dial_on_physical_nic、dial_lbtype、dial_lbtype_v6、join_ismanaged、join_isminorisp、join_isbantransprov、join_isipv6schedule、join_natforwardenable、join_idcbindtype、join_deploystate。

原始数值字段：bw、bandwidth、actualbandwidth、netbenchlimitbandwidth、cpuload1、cpuload5、cpuload15、corenum、cpuuser、cpusystem、cpunice、cpuidle、cpuiowait、cpuirq、cpusoftirq、cpusteal、memtotal、memused、memfree、membufferd、memcached、join_cpu_load1、join_cpu_load5、join_cpu_load15、join_cpu_corenumber、join_cpu_totalcores、join_cpu_totalphysicals、join_cpu_totalthreads、join_cpu_idle、join_memtotal、join_memused、join_memfree、join_membuffers、join_memcached、mem_used_ratio、retrans、v4pingloss、v6pingloss、v4pingavgrtt、v4pingbestrtt、v4pingworstrtt、v6pingavgrtt、v6pingbestrtt、v6pingworstrtt、totaldisksize、totaldiskused、hdddisksize、hdddiskused、ssddisksize、ssddiskused、systemdisksize、systemdiskused、disk_used_ratio、hdd_used_ratio、ssd_used_ratio、system_disk_used_ratio、maxioutil、quality_cpuutil、quality_retransrate、quality_pinglossrate、quality_rtt、sevendayavg95ratio、join_disks_totalsize、join_disks_total_size、join_disks_total_used、join_disks_total_usedrate、join_disks_hdd_size、join_disks_hdd_used、join_disks_hdd_usedrate、join_disks_ssd_size、join_disks_ssd_used、join_disks_ssd_usedrate、join_disks_system_size、join_disks_system_used、join_disks_system_usedrate、join_net_pingrtt、join_net_tcpretransrate、join_net_pinglossrate、join_actualbw、join_limitbw、join_actualbw_achieverate、join_actualbw_rtt、join_actualbw_tcpretr_rate、join_limitbw_achieverate、join_netbench_bandwidth、join_netcardbw、join_procbw、join_diskdivbw、cpu_load1_per_core、smart_available_spare、smart_bad_block_count、smart_case_temperature、smart_critical_warning、zfs_pool_online、zfs_dataset_read、zfs_dataset_write、prom_retrans_ratio、analysis_transprovrate、analysis_usb_bw、analysis_bw_num、analysis_serviceduration、analysis_yesterday_p95_bw、analysis_yesterday_online_duration、analysis_yesterday_evening_online_duration、analysis_yesterday_outline_count、analysis_yesterday_evening_outline_count、analysis_yesterday_snapshot_bw、analysis_yesterday_snapshot_netbench_bw、analysis_dby_online_duration、analysis_dby_evening_online_duration、analysis_dby_outline_count、analysis_dby_evening_outline_count、analysis_dby_p95_bw、analysis_dby_snapshot_bw、analysis_dby_snapshot_netbench_bw、analysis_dby2_p95_bw、analysis_dby2_online_duration、analysis_dby2_outline_count、analysis_dby2_evening_outline_count、analysis_dby2_snapshot_netbench_bw、analysis_out_bandwidth_total、analysis_out_bandwidth_node、analysis_out_limit_bandwidth_node、analysis_out_bandwidth_udp_node、analysis_out_limit_bandwidth_udp_node、analysis_out_bandwidth_udp_total、analysis_udp_actual_bandwidth、analysis_udp_netbench_limit_bandwidth、analysis_conn_succeed_bw_num、analysis_abnormal_line_num、analysis_ping_loss_stats_total、analysis_ping_loss_p5_abnormal、analysis_ping_loss_p10_abnormal、analysis_retrans_stats_total、analysis_retrans_p5_abnormal、analysis_cpuutil_stats_total、analysis_cpuutil_p70_abnormal、analysis_ipv4_packet_count、analysis_ipv6_packet_count、analysis_disk_div_bw、join_uptime、join_duration_time、join_has_errors、join_iptable_abnormal、join_biz_bw、join_yesterday_avg_peak_ratio、join_quality_retrans_abnormal、join_quality_retrans_total、join_quality_pingloss_p5_abnormal、join_quality_pingloss_p5_total、join_quality_pingloss_p10_abnormal、join_quality_pingloss_p10_total、join_quality_cpuutil_p70_abnormal、join_quality_cpuutil_p70_total、dial_account_count、yesterday_online_ratio、dby_online_ratio、analysis_retrans_abnormal_ratio、analysis_pingloss_abnormal_ratio、analysis_cpuutil_abnormal_ratio、analysis_actual_bw_ratio、analysis_udp_actual_bw_ratio、analysis_out_limit_bw_ratio。

历史收益模型使用数值字段：bw、bandwidth、actualbandwidth、netbenchlimitbandwidth、cpuload1、cpuload5、cpuload15、corenum、cpuuser、cpusystem、cpunice、cpuidle、cpuiowait、cpuirq、cpusoftirq、cpusteal、memtotal、memused、memfree、membufferd、memcached、join_cpu_load1、join_cpu_load5、join_cpu_load15、join_cpu_corenumber、join_cpu_totalcores、join_cpu_totalphysicals、join_cpu_totalthreads、join_cpu_idle、join_memtotal、join_memused、join_memfree、join_membuffers、join_memcached、mem_used_ratio、retrans、v4pingloss、v6pingloss、v4pingavgrtt、v4pingbestrtt、v4pingworstrtt、v6pingavgrtt、v6pingbestrtt、v6pingworstrtt、totaldisksize、totaldiskused、hdddisksize、hdddiskused、ssddisksize、ssddiskused、systemdisksize、systemdiskused、disk_used_ratio、hdd_used_ratio、ssd_used_ratio、system_disk_used_ratio、maxioutil、quality_cpuutil、quality_retransrate、quality_pinglossrate、quality_rtt、sevendayavg95ratio、join_disks_totalsize、join_disks_total_size、join_disks_total_used、join_disks_total_usedrate、join_disks_hdd_size、join_disks_hdd_used、join_disks_hdd_usedrate、join_disks_ssd_size、join_disks_ssd_used、join_disks_ssd_usedrate、join_disks_system_size、join_disks_system_used、join_disks_system_usedrate、join_net_pingrtt、join_net_tcpretransrate、join_net_pinglossrate、join_actualbw、join_limitbw、join_actualbw_achieverate、join_actualbw_rtt、join_actualbw_tcpretr_rate、join_limitbw_achieverate、join_netbench_bandwidth、join_netcardbw、join_procbw、join_diskdivbw、cpu_load1_per_core、analysis_transprovrate、analysis_usb_bw、analysis_bw_num、analysis_serviceduration、analysis_yesterday_p95_bw、analysis_yesterday_online_duration、analysis_yesterday_evening_online_duration、analysis_yesterday_outline_count、analysis_yesterday_evening_outline_count、analysis_yesterday_snapshot_bw、analysis_yesterday_snapshot_netbench_bw、analysis_dby_online_duration、analysis_dby_evening_online_duration、analysis_dby_outline_count、analysis_dby_evening_outline_count、analysis_dby_p95_bw、analysis_dby_snapshot_bw、analysis_dby_snapshot_netbench_bw、analysis_dby2_p95_bw、analysis_dby2_online_duration、analysis_dby2_outline_count、analysis_dby2_evening_outline_count、analysis_dby2_snapshot_netbench_bw、analysis_out_bandwidth_total、analysis_out_bandwidth_node、analysis_out_limit_bandwidth_node、analysis_out_bandwidth_udp_node、analysis_out_limit_bandwidth_udp_node、analysis_out_bandwidth_udp_total、analysis_udp_actual_bandwidth、analysis_udp_netbench_limit_bandwidth、analysis_conn_succeed_bw_num、analysis_abnormal_line_num、analysis_ping_loss_stats_total、analysis_ping_loss_p5_abnormal、analysis_ping_loss_p10_abnormal、analysis_retrans_stats_total、analysis_retrans_p5_abnormal、analysis_cpuutil_stats_total、analysis_cpuutil_p70_abnormal、analysis_ipv4_packet_count、analysis_ipv6_packet_count、analysis_disk_div_bw、join_uptime、join_duration_time、join_has_errors、join_iptable_abnormal、join_biz_bw、join_yesterday_avg_peak_ratio、join_quality_retrans_abnormal、join_quality_retrans_total、join_quality_pingloss_p5_abnormal、join_quality_pingloss_p5_total、join_quality_pingloss_p10_abnormal、join_quality_pingloss_p10_total、join_quality_cpuutil_p70_abnormal、join_quality_cpuutil_p70_total、dial_account_count、yesterday_online_ratio、dby_online_ratio、analysis_retrans_abnormal_ratio、analysis_pingloss_abnormal_ratio、analysis_cpuutil_abnormal_ratio、analysis_actual_bw_ratio、analysis_udp_actual_bw_ratio、analysis_out_limit_bw_ratio。

健康契约字段：CPU负载/核、内存使用率、磁盘使用率、重传、IPv4/IPv6丢包、node_join重传/丢包、磁盘IO利用率、SMART spare/坏块/温度/critical warning、ZFS池状态、Prometheus重传。

## 训练与测试效果

| 集合 | 节点数 | 行数 | 矿主命中率@1 | 运营命中率@1 | 矿主覆盖率 | 运营覆盖率 | 矿主遗憾 | 运营遗憾 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 训练集拟合内 | 21,291 | 235,293 | 17.78% | 18.36% | 57.16% | 57.97% | 43.28% | 47.05% |
| 测试集节点留出 | 5,323 | 59,001 | 17.77% | 18.45% | 57.35% | 58.43% | 43.43% | 48.71% |

| 集合 | 成本MAE | 成本RMSE | 成本R² | 利润MAE | 利润RMSE | 利润R² | 自动通过节点 | 需人工检查节点 | 无可行业务节点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 训练集拟合内 | 3.40512 | 80.2286 | -0.001596791759491012 | 1.29905 | 30.0152 | -0.0001820007127701917 | 0 | 16,297 | 4,994 |
| 测试集节点留出 | 2.65678 | 61.715 | -0.001584374291943469 | 1.15842 | 27.9687 | -7.923889228078629e-05 | 0 | 4,087 | 1,236 |

训练集矿主基线业务为 `10000251`，运营基线业务为 `10000251`。
覆盖率只表示临时排序业务在该节点历史结果中实际出现过，遗憾和价值捕获率仅在可观测业务上计算；这不是对从未运行过业务的因果效果估计。

## 健康约束

CPU、内存、磁盘、SMART、ZFS、重传都纳入字段契约。硬阈值见 `multibusiness_metrics.json`；业务特定可行域由训练集该业务历史节点的上分位数生成。由于没有业务方提供的合格阈值和业务-资源要求映射，这些规则是可追溯的初始规则，不是已证实的准入标准。硬性不合格会阻断业务；指标缺失只进入人工检查，不会被填成健康。

Prometheus 直接按 `node_id` 探查结果为 `mapping_usable=True`；这些 VM 指标取运行时快照，实际节点覆盖率：SMART spare 0.42%、坏块 0.00%、温度 0.00%、critical warning 0.40%、Prometheus重传 8.93%、ZFS池 0.00%。当前自动通过节点为训练 0、测试 0；`miner_best`/`operator_best` 只有自动通过时才填值，人工检查结果保存在 `*_provisional` 字段。
非正容量/核数异常计数：{'corenum': 12123, 'memtotal': 12129, 'totaldisksize': 12217, 'hdddisksize': 18143, 'ssddisksize': 17975, 'systemdisksize': 12221}；负值哨兵计数：{'retrans': 25, 'v4pingloss': 24, 'v6pingloss': 24, 'quality_retransrate': 0, 'quality_pinglossrate': 0, 'join_net_tcpretransrate': 83, 'join_net_pinglossrate': 42, 'join_net_pingrtt': 42}。这些值在派生健康特征中按缺失处理，原始值仍保留在节点 CSV 中。

## 文件

- `multibusiness_eligible_pairs.csv`：符合资格的节点-业务首次日期明细。
- `multibusiness_outcomes.csv`：节点-业务 7 天成本、收入、利润结果。
- `multibusiness_nodes.csv`：node_analysis_data 与 node_join 属性快照及缺失值。
- `multibusiness_prometheus_probe.json`：Prometheus 指标映射探查结果。
- `multibusiness_recommendations.csv`：训练集和测试集推荐结果及健康阻断理由。
- `multibusiness_metrics.json`：机器可读指标、字段和规则。
