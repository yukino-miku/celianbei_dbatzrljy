# 问题一输出说明

- `manifest.json`：运行时间、样本数、模型和文件清单。
- `tables/data_validation_checks.csv`：原始数据结构与物理检查。
- `tables/anomaly_records.csv`：3 行局部修复的原值、清洗值和标记。
- `tables/battery_early_degradation_features.csv`：49 块电池的早期退化特征；测试电池 SOH200 为空。
- `tables/model_truncation_validation.csv`：四种模型在前 100/150 次截断下的逐电池未来误差。
- `tables/model_selection_summary.csv`：候选模型综合评价与最终选择。
- `tables/model_lifetime_stability.csv`：逐电池、逐模型使用前 100/150/200 次时的 EOL 稳定性。
- `tables/candidate_model_fits.csv`：候选模型参数、拟合状态和 EOL。
- `tables/battery_lifetime_estimates.csv`：40 块训练电池的二次模型寿命、bootstrap 区间和结构敏感性。
- `tables/battery_question1_results.csv`：训练电池特征与寿命合并总表。
- `tables/strategy_question1_summary.csv`：策略级稳健统计。
- `tables/strategy_mapping.csv`：S1–S9 与完整策略名映射。

所有寿命列均为早期曲线外推指标，不是真实完整寿命标签。
