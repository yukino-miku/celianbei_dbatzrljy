# 数模论文

- LaTeX 源稿：`paper/main.tex`
- 最终 PDF：`output/pdf/celianbei_A_paper.pdf`
- 编译引擎：XeLaTeX
- 版式：摘要独占第 1 页；正文、AI 工具使用声明与参考文献共 17 页；附录自第 18 页起，含支撑材料清单、复现命令和完整源程序。

在项目根目录执行：

```powershell
cd paper
xelatex -interaction=nonstopmode -output-directory=build main.tex
xelatex -interaction=nonstopmode -output-directory=build main.tex
```

论文引用的图片直接来自 `figures/question1` 至 `figures/question4`。修改正文或图片后需重新编译两遍，以更新交叉引用。
