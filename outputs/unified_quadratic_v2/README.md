# unified_quadratic_v2

本目录为不覆盖旧 `outputs/question1..4` 的统一重构正式结果。

- `question1/tables`: 每电池 `a,d1,d2,R100,R200,A`、参数边界、300 次 bootstrap 和截断稳定性。
- `question2/tables`: 策略级桥接、OLS/Ridge、LOSO、VIF、两阶段暴露、1000 次分组 bootstrap、局部匹配、机制关联和证据矩阵。
- `question3/tables`: 40 块电池严格伪测试、策略先验惩罚选择、特征消融、9 块测试电池逐循环预测和统一 quadratic EOL。
- `question4/tables`: 充电时间验证、可信域全部候选、中位/P10 Pareto、1000 次寿命 bootstrap、代表策略、敏感性和旧新模型比较。
- `manifest.json`: 正式样本数、随机种子、模型选择与唯一 EOL 定义。

所有 EOL 都是基于早期数据的条件性模型外推，不是真实完整寿命标签。所有优化候选均未经新实验验证。
