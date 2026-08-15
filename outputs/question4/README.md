# Question 4 outputs

本目录由 `scripts/run_question4.py` 自动生成，复用问题一至问题三结果，不重新执行或修改清洗。

- `tables/charge_time_model_validation.csv`：充电时间候选模型 LOSO；
- `tables/degradation_proxy_validation.csv`：退化代理候选 LOSO；
- `tables/parameter_trust_domain.csv`：矩形、凸包和最近邻信任域定义；
- `tables/optimization_grid_trusted.csv`：主可信域全部 4,413 个候选；
- `tables/pareto_candidates_main.csv`：主 Pareto 前沿；
- `tables/recommended_strategies.csv`：快充、knee、寿命三类代表点及不确定性；
- `tables/recommendation_existing_strategy_comparison_all.csv`：推荐与 S4–S9 全部现有策略对照；
- `tables/*sensitivity*`、`bootstrap_*`、`leave_one_strategy_out_*`：鲁棒性结果；
- `manifest.json`：完整结果清单和限制。

所有优化点均为代理模型候选，未经实验验证。
