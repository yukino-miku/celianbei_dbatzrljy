# celianbei_dbatzrljy

2026 年“策联杯”数学建模精英联赛 A 题项目：锂离子电池快充策略、SOH 退化与循环寿命估计。

当前已完成问题一的数据整理、早期退化特征提取、候选退化模型时间截断验证、80% SOH 寿命估计、不确定性分析及策略级比较。问题二的充电参数效应模型尚未展开。

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
.\.venv\Scripts\python.exe -m pytest -q
```

完整运行约生成：1 份清洗后循环数据、11 张结果表、15 张 320 dpi PNG 和对应的 15 张 SVG 矢量图。随机过程使用固定种子，输出清单见 `outputs/question1/manifest.json`。

## 目录

```text
celianbei_dbatzrljy/
├─ materials/                 # 原始题面、论文规范及补充说明（只读）
├─ data/raw/                  # 题目原始 CSV（只读）
├─ data/processed/question1/  # 问题一清洗后数据
├─ docs/                      # 数据审计、方法、结果与 AI 使用记录
├─ src/                       # 模块化清洗、特征、模型、汇总和绘图代码
├─ scripts/                   # 可直接运行的流水线
├─ tests/                     # 数据完整性及问题一关键逻辑测试
├─ outputs/question1/tables/  # 问题一结果表
└─ figures/question1/         # PNG 与 SVG 候选图
```

## 重要说明

问题一的长期寿命均为远距离外推结果。残差块 bootstrap 区间仅反映选定模型条件下的不确定性，候选模型之间的差异代表额外的结构不确定性；两者不能混为一谈。策略差异目前只作描述性比较，不作充电参数的因果解释。
