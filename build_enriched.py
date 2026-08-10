"""融合 node_join 静态属性 + node_analysis_data 静态库存属性 -> nodes_enriched.csv

说明（已核实）：
- node_analysis_data(day=20260805) 仅覆盖样本节点的 19%（5,205/27,492），未覆盖节点新列为 NaN。
- hardwaretype 在源数据中全为空 -> 丢弃；保留真正有值的 os / arch 作为新类别特征。
- 数值：bw_rated(额定带宽), corenum(核数), memtotal(内存), totaldisksize/hdddisksize/ssddisksize/systemdisksize(磁盘)
"""
import os, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
nodes = pd.read_csv(os.path.join(HERE, "nodes.csv"))
na = pd.read_csv(os.path.join(HERE, "node_analysis_static_sample.csv")).rename(columns={"nodeid": "node_id"})

extra_cat = ["os", "arch"]                       # 新类别（node_join 无），android/linux, armeabi/x86_64
extra_num = ["bw_rated", "corenum", "memtotal", "totaldisksize",
             "hdddisksize", "ssddisksize", "systemdisksize"]

merged = nodes.merge(na[["node_id"] + extra_cat + extra_num], on="node_id", how="left")
covered = merged[extra_num[0]].notna().sum()      # 用有值的列衡量覆盖
print(f"融合后节点数: {len(merged)}，node_analysis 覆盖: {covered} ({covered/len(merged):.1%})")

for c in extra_num:
    merged[c] = pd.to_numeric(merged[c], errors="coerce")
for c in extra_cat:
    merged[c] = merged[c].astype("string").str.strip()

merged.to_csv(os.path.join(HERE, "nodes_enriched.csv"), index=False)
print("已保存 nodes_enriched.csv；新增特征:", extra_cat + extra_num)
print("os 分布:", merged["os"].value_counts(dropna=False).to_dict())
print("arch 分布:", merged["arch"].value_counts(dropna=False).to_dict())
