# 问题二方法

## 1. 分析单位与复用口径

- 直接读取问题一的 `battery_question1_results.csv`；不重复清洗、不重新定义 SOH，也不接触 9 块测试电池未来信息。
- 策略整体比较以 40 块非测试电池为观测单位；参数模型以策略中位数为观测单位，避免把 cycle 行虚增成独立样本。
- S2（`80PER_3_6C`）保留在九策略比较中；因其 `C1` 缺失，参数模型只使用其余 8 个策略，且绝不把 `C1` 填成 3.6。

## 2. 策略整体差异

对 SOH150、SOH200、50–200 次稳健斜率、曲率和 quadratic EOL 同时计算 Welch ANOVA、Kruskal–Wallis 与 19,999 次标签置换检验。报告置换 eta-squared、Kruskal epsilon-squared 与中位数跨度。全局置换检验达到 0.05 后才进行成对精确置换，报告 Cliff's delta、Hedges' g、组内 bootstrap 中位数差区间，并在每个响应内做 Holm 校正。

## 3. 参数模型

基准模型为标准化的 `C1 + Q1 + C2` OLS；同时给出岭回归留一策略选参。交互结构不堆高阶项，而用两个可解释暴露量：

\[
E_1=C_1Q_1/80,\qquad E_2=C_2(80-Q_1)/80.
\]

三参数模型、两阶段暴露模型和一维应力模型用 AICc 与 leave-one-strategy-out 误差比较。OLS 的常规 p 值只作描述，不用于单独确认参数效应。

## 4. SOC 加权应力

\[
S_p=\frac{Q_1}{80}C_1^p+\frac{80-Q_1}{80}C_2^p,
\quad 1\le p\le3.
\]

在 41 个网格点上分别计算 SOH200、50–200 斜率和 EOL 的留一策略标准化 RMSE，再以三者平均值选择一个共享 `p`。这避免针对某一结果单独挑参。若最优值落在边界，只解释为搜索区间内的敏感性结果。

## 5. 批次与不确定性

`dataset_id=1` 只含 S1，`dataset_id=2` 只含 S2/S3，`dataset_id=3` 含 S4–S9，且 `NEWSTRUCTURE` 与 dataset 3 重合。因此同时报告：全数据未控制、加入 dataset 2/3 虚拟变量、仅 dataset 3。由于设计不交叉，批次与策略不能被完全识别。

不确定性采用策略优先、策略内再重采样电池的 grouped bootstrap。参数相关另给精确 Spearman 置换 p 值与 bootstrap 区间。全部验证均在策略层留一，不打乱循环时间。
