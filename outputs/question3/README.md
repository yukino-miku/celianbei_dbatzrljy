# 问题三输出说明

本目录由 `scripts/run_question3.py` 自动生成。

- `pseudotest_cycle_predictions.csv`：40 块训练电池、6 个模型、151–200 的逐 cycle 真实值和预测值。
- `pseudotest_battery_errors.csv`、`model_comparison.csv`、`strategy_prediction_errors.csv`：电池、模型和策略三级误差。
- `test_cycle151_200_predictions.csv`：9 块测试电池 450 行点预测、95%经验区间、模型分歧和偏离标记。
- `test_component_predictions.csv`、`ensemble_weights.csv`：分模型预测与逐电池贡献。
- `eol_pseudotest_by_battery.csv`、`eol_scheme_comparison.csv`：三类 EOL 方案的稳定性比较。
- `test_eol_estimates.csv`：测试 EOL 点估计、bootstrap 区间、结构敏感性和可靠性。
- `manifest.json`：数据边界、泄漏防护、参数和完整产物清单。

`test_eol_curves.csv` 仅用于绘制 constrained quadratic 外推轨迹，不是观测数据。
