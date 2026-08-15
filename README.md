# celianbei_dbatzrljy

2026 年“策联杯”数学建模精英联赛 A 题项目：锂离子电池快充策略、SOH 退化与循环寿命估计。

当前已完成问题一至问题三。问题三使用 40 块训练电池做严格 leave-one-battery-out 伪测试，只用 9 块测试电池的 cycle 1–150 预测 151–200，并在短期预测后按问题一 constrained quadratic 口径给出条件性 EOL 估计。

## 问题一口径

- 依据官方补充说明，循环寿命定义为模型预测 SOH 首次达到 0.8 的循环次数；附件没有真实完整寿命标签并不构成数据缺失。
- 40 块非测试电池用于问题一建模和按时间顺序验证。
- 9 块 `prediction_test=1` 电池只保留已有观测用于数据检查和可视化，不参与模型拟合、选择或寿命统计。
- 第 150/200 次循环 SOH 和分段退化速率是早期退化指标，不是真实寿命。

## 复现

```powershell
cd D:\mywork\code\celianbei_dbatzrljy
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\run_question1.py --bootstrap-samples 300
.\.venv\Scripts\python.exe scripts\run_question2.py
.\.venv\Scripts\python.exe scripts\run_question3.py
.\.venv\Scripts\python.exe -m pytest -q
```

问题三完整运行生成 18 张结果表、13 张 320 dpi PNG 和对应 SVG；随机过程使用固定种子，输出清单见 `outputs/question3/manifest.json`。

## 目录

```text
celianbei_dbatzrljy/
├─ materials/                 # 原始题面、论文规范及补充说明（只读）
├─ data/raw/                  # 题目原始 CSV（只读）
├─ data/processed/question1/  # 问题一清洗后数据
├─ docs/                      # 数据审计、方法、结果与 AI 使用记录
├─ src/                       # 模块化清洗、特征、模型、汇总和绘图代码
├─ scripts/                   # 可直接运行的流水线
├─ tests/                     # 数据完整性及问题一、问题二关键逻辑测试
├─ outputs/question1/tables/  # 问题一结果表
├─ outputs/question2/tables/  # 问题二结果表
├─ outputs/question3/tables/  # 问题三伪测试、测试预测和EOL结果
├─ figures/question1/         # 问题一 PNG 与 SVG 候选图
├─ figures/question2/         # 问题二 PNG 与 SVG 候选图
└─ figures/question3/         # 问题三 PNG 与 SVG 候选图
```

## 重要说明

问题三的 151–200 预测可用训练电池真实未来严格验证；80% SOH 寿命没有真实标签，只能根据截断稳定性、重采样区间和候选模型分歧评估。测试 EOL 不是精确真值。
