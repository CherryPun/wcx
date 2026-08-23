# hrr_node_business_precdict

HRR 的节点—业务反事实金额预测实验。目标是在每个节点只运行少量业务的情况下，利用节点特征、业务先验、节点历史和矩阵补全，预测该节点未运行过业务的 7 天成本与收入。

## 当前结论

验证采用节点内部 90% 训练、10% 业务遮挡，共 8,673 条完整 7 天记录，其中训练 7,806 条、验证 867 条。

| 目标 | MSE | MAE | R² | WAPE |
|---|---:|---:|---:|---:|
| 成本 | 117,087.18 | 86.92 | 0.6399 | 56.88% |
| 收入 | 203,341.71 | 106.97 | 0.5321 | 61.12% |

最终方案是非负、权重和为 1 的历史增强融合模型。业务遮挡仅用于验证，不是协同过滤本身；矩阵分解是融合模型中的一个候选信号。

## 文件说明

- `run_counterfactual_matrix_completion.py`：90/10 节点内遮挡、XGBoost/业务先验/矩阵分解基线训练。
- `optimize_counterfactual_blends.py`：按 WAPE/MSE 搜索受约束融合权重。
- `run_counterfactual_history_augmentation.py`：加入节点及业务历史特征，产生当前最终结果。
- `export_counterfactual_validation_errors.py`：导出每条验证记录的金额差和百分比误差。
- `validate_feature_similar_nodes.py`：仅基于节点固有特征寻找相似节点，验证业务扩散可行性。
- `counterfactual_matrix_completion_artifact/`：当前最终融合实际使用的子模型和元数据；未采用的实验模型没有上传。
- `counterfactual_validation_error_details.csv`：867 条验证记录的真实值、预测值和逐条误差。
- `feature_similar_node_pairs.csv`：节点 Top-10 特征近邻关系。
- `feature_similar_same_business_details.csv`：相似节点共同业务的成本/收入差异明细。
- `feature_neighbor_masked_predictions.csv`：相似节点方法在 10% 遮挡集上的预测。
- 两份 `.md` 中文报告：正式反事实模型报告与相似节点业务扩散预验证报告。

## 数据文件

- `multibusiness_outcomes.csv`：完整 7 天节点—业务金额样本。
- `multibusiness_nodes.csv`：节点静态特征。
- `outcomes_raw.csv`：历史窗口数据，用于构造历史日均特征。

这些数据来自原仓库已有数据及其清洗结果，没有引入外部数据。

## 运行

建议使用 Python 3.11+：

```bash
pip install -r requirements.txt
python run_counterfactual_matrix_completion.py
python optimize_counterfactual_blends.py
python run_counterfactual_history_augmentation.py
python export_counterfactual_validation_errors.py
python validate_feature_similar_nodes.py
```

前三步会重新训练并生成全部候选模型，因此可能覆盖当前结果和模型产物。随机种子及 90/10 遮挡逻辑已写入脚本。

## 相似节点预验证

相似度只使用 45 个类别特征和 17 个数值容量特征，金额与业务结果不参与相似度计算。相似节点相同业务的平均金额差异显著小于随机对照，但纯近邻模型的 WAPE 尚未超过当前历史融合模型，因此目前把它定位为下一步可融合的补充信号。

