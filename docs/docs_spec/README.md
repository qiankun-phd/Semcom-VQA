# docs_spec 索引（2026-07-07 整理）

## 现役文档（9 份）

| 文件 | 用途 |
|---|---|
| **UAV_VQA_SemCom_System_and_Results_v6.html** | **项目总报告**（唯一权威版）：Part A 语义通信设计 / B 语义通信实验 / C BUBBLES 场景详设 / D 求解算法细节 / E RL 实验结果 / F 发现·定位·复现。16 图 30+ 表，全部实测数字。 |
| gen_report_v6.py | 上述报告的生成器（改内容→重跑即得新版；图源 /tmp/report_figs/，缺失从 160 取） |
| UAV_VQA_SemCom_Principle.md | 证据-问题互补性原理文档（论文一头条发现的原始推导） |
| BUBBLES_D2.1_Analysis.md | SESAR BUBBLES D2.1 精读：场景参数锚点 + 分隔链公式 + TLS + 引用策略 |
| V19_Design_Review_2026-07.md | v19 算法双审查合并结论 + P0/P1 修复清单 + A/B 判据（RL 侧的规格书） |
| RL_Experiment_Standards_Survey.md | DRL 无线论文实验规范调研（收敛/基线/泛化的合格线与实验矩阵依据） |
| Novelty_Journal_Assessment_2026-07.md | 新颖性评估与期刊定位（TCCN 主投判断 + 补强表） |
| Paper1_Positioning_and_Citations.md | 论文一新颖性口径红线 + related work 划界段落 + BibTeX |
| SemCom_Reference_Repos_Code_Analysis.md | 五参考仓库代码考据（MA-DeepSC/PADC/ADJSCC/DeepJSCC-f/SJTU） |

## 论文工作目录（在上级 HPPO-VQA/ 下）

- `paper_semcom/` — 论文一（语义通信有效性，8 页可编译骨架）
- `paper_rl/` — 论文二（BUBBLES 资源分配，撰写中）
- `paper/` — 旧线论文（Role-Split 卸载，独立）

## archive/（13 份，已被取代但保留可溯）

报告 v2-v5 及其生成器（被 v6 取代）、182 时代 Results_Report、Comparison_Plan（已执行完）、
System_Spec 初版（被 v6 Part A 取代）。
