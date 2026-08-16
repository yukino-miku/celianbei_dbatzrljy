# 数模论文

- baseline 源稿：`paper/main.tex`
- baseline PDF：`output/pdf/celianbei_A_paper.pdf`
- 统一二次退化主线源稿：`paper/unified_main.tex`
- 统一二次退化主线 PDF：`output/pdf/celianbei_A_unified_quadratic.pdf`
- 编译引擎：XeLaTeX
- 两套论文与结果并存，统一主线版本不会覆盖原 baseline。

在项目根目录执行：

```powershell
cd paper
xelatex -interaction=nonstopmode -output-directory=build main.tex
xelatex -interaction=nonstopmode -output-directory=build main.tex
xelatex -interaction=nonstopmode -output-directory=build unified_main.tex
xelatex -interaction=nonstopmode -output-directory=build unified_main.tex
```

baseline 图片来自 `figures/question1` 至 `figures/question4`，统一主线图片来自
`figures/unified_quadratic_v2`。修改正文或图片后需重新编译两遍，以更新交叉引用。
