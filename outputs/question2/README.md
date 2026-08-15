# 问题二输出说明

本目录由 `scripts/run_question2.py` 自动生成。输入只来自问题一已验证表与原始电池汇总元数据。

- `strategy_difference_tests.csv`：五类响应的 Welch、Kruskal、置换检验与全局效应量。
- `pairwise_strategy_tests.csv`：全局显著后才执行的成对精确置换、效应量和 Holm 校正。
- `strategy_robust_summary.csv`：策略均值、中位数、标准差、四分位数、范围及 5,000 次 bootstrap 区间。
- `parameter_ols_results.csv`、`ridge_parameter_results.csv`：解释性基准及正则化对照。
- `collinearity_diagnostics.csv`、`parameter_correlations.csv`：VIF、条件数和相关矩阵。
- `stress_p_search.csv`、`stress_model_results.csv`：共享指数选择和全数据/批次/同批次结果。
- `grouped_bootstrap_*.csv`：策略优先、策略内电池重采样的不确定性。
- `batch_sensitivity_coefficients.csv`：不同批次处理方式的系数对照。
- `cross_metric_conclusion_matrix.csv`：SOH200、斜率、EOL 三指标一致性分类。

`grouped_bootstrap_coefficients.csv` 是完整迭代级结果，体积较大；论文通常使用其 summary 表。
