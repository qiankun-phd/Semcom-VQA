# 中文整合稿

## 阅读与编译

- 完整稿入口：`main_zh.tex`。
- 编译结果：`build/main_zh.pdf`，当前共 22 页。
- 内容：标题、摘要、关键词、六节正文、统一参考文献及文末补充材料。
- 这是便于讨论和修改的中文阅读稿，不是期刊投稿模板；英文主稿仍使用 `main.tex`。

从 `paper/` 目录执行：

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main_zh.tex
```

## 正文维护

完整稿与原有各节独立稿共用 `sections_zh/` 下的正文文件。后续修改中文内容应编辑该目录，避免维护两套文字。

| 内容 | 共享源文件 |
| --- | --- |
| 摘要及关键词 | `sections_zh/00_abstract.tex` |
| 引言 | `sections_zh/01_introduction.tex` |
| 相关工作 | `sections_zh/02_related_work.tex` |
| 系统模型 | `sections_zh/03_system_model.tex` |
| 问题条件下的证据路由 | `sections_zh/04_evidence_routing.tex` |
| 实验与讨论 | `sections_zh/06_experiments.tex` |
| 单段结论 | `sections_zh/08_conclusion.tex` |
| 补充材料 | `sections_zh/supplementary.tex` |

原有 `abstract_zh.tex`、`introduction_zh.tex` 等文件保留为独立编译入口。跨章节引用从 `build/main_zh.aux` 导入，因此应先编译完整稿，再编译独立稿。

## 合稿约定

- 正文图、表、公式分别连续编号；补充材料图表使用 S 前缀。
- 文献统一读取 `refs.bib`。中英文正文均引用 41 条文献，编译后的顺序相同。
- 系统示意图和图表标题使用中文；实验图沿用已核验的英文图内标签与原始数值，方便中英文对照。
- 本次合稿不改变实验数据、统计口径或英文主稿，不启动任何实验。
- 已声明的实机能耗、在线链路和替代接收器能耗等证据边界仍然保留。合稿完成不等于投稿前文献核验、作者信息及期刊格式检查已经全部完成。

本次覆盖核查记录见 `../docs/plan_tgcn/chinese-assembly-review-2026-09-07.md`。
