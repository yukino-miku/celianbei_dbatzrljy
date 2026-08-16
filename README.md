# celianbei_dbatzrljy

2026 年“策联杯”A 题锂离子电池建模项目。现已在**不删除、不覆盖旧问题一至四结果**的前提下，完成统一二次退化主线重构：

\[
(C_1,Q_1,C_2)\rightarrow(d_1,d_2)\rightarrow
SOH(n)=a-d_1n/100-d_2(n/100)^2\rightarrow L_{SOH=0.8}.
\]

旧结果仍位于 `outputs/question1..4` 和 `figures/question1..4`；新结果单独位于 `outputs/unified_quadratic_v2`、`figures/unified_quadratic_v2`，新论文为 `paper/unified_main.tex`。问题四旧 SOC stress 路线只作 baseline，不再作为主寿命代理。

## 十二个核心问题的明确回答

1. **问题一为什么选择 quadratic？** 线性、quadratic、power、exponential 均按时间顺序做 100/150 次截断验证。受约束 quadratic 参数少、长期单调、EOL 可解析，且在短期外推误差与寿命稳定性之间最均衡，因此保留为四问唯一最终寿命函数；这不表示其长期 EOL 是实测真值。

2. **`d1`、`d2` 分别表示什么？** `d1` 是百循环尺度上的基础退化项，`d2` 是非负二次加速项。瞬时损失率为 `d1+2*d2*n/100`；`R100=d1+2*d2`，`R200=d1+4*d2`，`A=2*d2`。

3. **前 200 次能否稳定识别 `d1/d2`？** 只能有限识别。40 块训练电池中 37.5% 至少一个参数命中零边界；单电池 300 次 bootstrap 的 `d1-d2` 相关系数中位数为 -0.921，显示强补偿。后续策略层改用更易解释的 `R100/A`，但最终仍换回同一 `d1/d2` quadratic。

4. **`C1/Q1/C2` 如何影响 `d1/d2`？** dataset 3 只有 6 个独立策略点。原始三参数 Ridge 的综合标准化 LOSO RMSE 为 1.142，未优于折内均值；绝大多数系数 bootstrap 区间跨 0。只有 `Q1 -> R100` 方向较稳定。因此不能给出确定独立效应，只能把 `Q1/C2` 的部分方向列为 `directional_but_limited`。

5. **`Q1` 如何改变两个阶段的 SOC 持续范围？** 令 `q=Q1/80`，`E1=q*C1^p`、`E2=(1-q)*C2^p`。增大 `Q1` 同时增加第一阶段权重、缩短第二阶段权重；两者必须分开分析。最佳暴露候选为 `p=1`，但综合 LOSO RMSE 1.302，不可把它解释为物理定律。

6. **第二阶段高倍率主要影响基础退化还是加速退化？** `E2 -> R100` 的 bootstrap 方向稳定率约 99.4%，而 `E2 -> A` 仅约 54.1%；数据更偏向“影响 100 次附近基础速率”的趋势，但暴露模型 LOSO 较差，所以结论仍有限。

7. **IR/Tavg/chargetime 是否提供独立机制证据？** 没有稳定证据。严格按整策略留出的 M2（加入第 1–50 次 IR、温度、充电时间）没有稳定优于 M1（策略参数+初始容量），SOH200 的标准化误差反而从 1.183 增至 1.633。不能称为因果中介。

8. **问题三如何利用策略先验并用前 150 次修正？** 对每个目标电池构造排除该电池后的策略 `mu_d1/mu_d2`，再用前 150 次 SOH 和标准化收缩惩罚拟合个体 quadratic。惩罚 `lambda` 由 40 块训练电池伪测试选择，结果为 `lambda=0`，说明当前稀疏先验没有带来净收益。短期仍采用验证更好的 adaptive ensemble，之后把“真实 1–150 + 预测 151–200”重新拟合到同一 quadratic 求 EOL。

9. **问题四为什么不再单独使用 SOC-stress 寿命模型？** 单独 stress→EOL 会使问题四的寿命定义与问题一至三脱节。新主线直接预测退化曲线参数并生成完整 SOH；旧 stress 仍保留作拟合与 Pareto benchmark。

10. **问题四如何严格得到优化目标？** 对可信域内每个候选先由问题二模型预测 `R100/A`，投影并换算 `d2=A/2, d1=R100-A`，取 `a=1` 生成完整 SOH，再解析求 EOL。正式 1000 次策略优先/策略内电池 bootstrap 输出 `L_median/L_p10/L_p90`，分别构造中位寿命和 P10 鲁棒 Pareto。

11. **新旧问题四是否一致？** 只在“约 10 分钟附近存在时间–寿命折中”这一大方向一致，寿命水平与推荐区域并不完全一致。旧 stress 对已有策略 EOL 的原定义重现 MAE 约 88 次，新 `a=1` quadratic 桥接约 118 次，定义并非完全同尺度。新模型的优势是四问逻辑统一，不是经验拟合必然更好。

12. **哪些结论稳健，哪些受小样本限制？** 151–200 次短期预测最可靠，因为有 40 块电池真实未来验证；数据清洗、时间顺序验证和单调 quadratic 口径较稳健。具体充电参数独立效应、测试电池长期 EOL、Pareto 精确点均受 6 个策略点、批次混杂和长期外推限制，应视为趋势或待实验验证的模型推荐。

## 正式结果摘要

- 统一策略响应：`R100/A`；选定策略模型为标准化 raw-parameter Ridge，`alpha=10`。
- 问题三伪测试：adaptive ensemble 平均 RMSE `0.000361`，individual quadratic `0.000514`，policy quadratic `0.000597`；最终短期模型仍为 adaptive ensemble。
- 问题四代表点：

| 类型 | C1 | Q1 | C2 | 时间/min | L_median | L_p10 |
|---|---:|---:|---:|---:|---:|---:|
| fast-charge | 5.20 | 58 | 4.55 | 9.817 | 1354 | 1240 |
| ideal-point compromise | 5.10 | 64 | 4.30 | 9.986 | 1377 | 1271 |
| longevity | 5.00 | 67 | 4.00 | 10.214 | 1399 | 1299 |

这些策略全部在凸包与最近邻信任域交集内，但仍是**代理模型推荐，不是实验验证结果**。折中和寿命方案的 bootstrap 参数区域较宽，论文主张推荐区域而非精确单点。

## 复现

```powershell
cd D:\mywork\code\celianbei_dbatzrljy
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\run_unified_refactor.py
.\.venv\Scripts\python.exe -m pytest -q
```

正式运行固定种子 `20260815`，包括每电池参数 bootstrap 300 次、问题二系数 bootstrap 1000 次、问题四候选寿命 bootstrap 1000 次、测试电池 EOL bootstrap 300 次；生成 33 张新增 PNG 和对应 SVG。开发调试可用 `--quick`，但其输出不能作为正式提交结果。

## 目录

```text
outputs/question1..4/                         # 旧 baseline 表（保留）
figures/question1..4/                         # 旧 baseline 图（保留）
outputs/unified_quadratic_v2/                 # 新统一结果表与 manifest
figures/unified_quadratic_v2/                 # 新增 PNG/SVG 候选图
src/celianbei_dbatzrljy/unified_*.py           # 统一核心、流水线、绘图
scripts/run_unified_refactor.py                # 正式复现入口
paper/main.tex                                 # 旧 baseline 论文
paper/unified_main.tex                         # 新统一论文
output/pdf/celianbei_A_paper.pdf               # 旧 baseline PDF
output/pdf/celianbei_A_unified_quadratic.pdf   # 新统一 PDF
```

完整方法与结果说明见 `docs/10_unified_quadratic_methodology.md` 和 `docs/11_unified_quadratic_results.md`。
