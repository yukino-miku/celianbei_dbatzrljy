# 统一二次退化结果摘要

## 问题一

- 37.5% 的训练电池至少一个退化参数命中零边界。
- 单电池 bootstrap 的 d1-d2 相关系数中位数为 -0.921，存在强参数补偿。
- 主策略响应选择 R100/A，但最终 EOL 仍由同一 d1/d2 quadratic 计算。
- 100→200 截断的 d2/A 中位相对漂移约 0.430；150→200 降至 0.095。

## 问题二

- raw parameter Ridge 的综合标准化 LOSO RMSE 为 1.142；sum-stress benchmark 为 1.229；separate exposure 为 1.302。
- raw 模型只有 Q1→R100 的 bootstrap 方向稳定且区间不跨 0。
- 暴露模型 E1→R100 为负、E2→R100 为正，方向稳定率约 99%，但 LOSO 较差；对 A 的方向不稳定。
- M2 早期 IR/T/充电时间没有在整策略留出下稳定改善 M1。
- S3/S9 表明同参数不同结构可产生明显不同退化，batch/structure 不能忽略。

## 问题三

- policy prior 惩罚经验证选择 lambda=0，策略先验没有净收益。
- adaptive ensemble 平均 RMSE 0.000361，优于 individual quadratic 0.000514 和 policy quadratic 0.000597。
- 最终短期模型为 ensemble，但 9 块测试电池 EOL 全部回归统一 quadratic。
- Battery 2 的 EOL 区间特别宽；Battery 9、16 条件可靠性相对较高。

## 问题四

- 时间模型选择 physical offset，dataset 3 LOSO RMSE 0.439 min。
- fast/compromise/longevity 代表点为 (5.20,58,4.55)、(5.10,64,4.30)、(5.00,67,4.00)。
- 对应时间 9.817/9.986/10.214 min，L 中位数 1354/1377/1399，P10 1240/1271/1299。
- 三点均在主可信域且属于 P10 非支配集合；折中与寿命点 bootstrap 参数区域宽，需按区域推荐。
- 旧 stress 在已有策略 EOL 经验重现上略优；新 quadratic 的价值主要是四问逻辑一致和不确定性可传播，不代表预测性能全面胜出。

## 结论强度

最可靠的是 151–200 次短期预测；参数独立效应属于有限方向性或不支持；长期 EOL 和 Pareto 点都是模型条件性结论，必须通过新实验验证。
