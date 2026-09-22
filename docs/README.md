# 文档索引

先读当前研究记录。下面的历史计划和旧报告保留作对照，不覆盖当前结论。

## 当前入口

| 文件 | 用途 |
|---|---|
| [RESEARCH_CONTROL.md](RESEARCH_CONTROL.md) | 研究目标、证据边界、现在做到哪一步 |
| [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) | 哪项实验做过、结果能支持什么、为什么不要原样重跑 |
| [NETWORK_RESULTS.md](NETWORK_RESULTS.md) | 编码器、门控、接收器怎么连接，以及同口径准确率 |
| [diffusion-vqa-novelty-review-2026-09-14.md](diffusion-vqa-novelty-review-2026-09-14.md) | 扩散式 VQA 路线的新颖性核对 |

实验室地址不写在这些公开文档里，只留在本地、不提交的 `LOCAL_HOSTS.md`。

## 文稿

TGCN 主文在 `paper/`：`main.tex`、`sections/`、`refs.bib`、`figures/`。`paper/outputs/` 是实验产物、权重和图像缓存，体积约 9 GB，不进入仓库。

`paper_related/` 是另外两条稿（TCCN 的 MA-HPPO、BUBBLES/RL），只供对照，不是这条主文。

## 历史材料

| 目录 | 状态 |
|---|---|
| [plan_tgcn/](plan_tgcn/) | 2026-09 文稿分节修改记录 |
| [docs_spec/](docs_spec/) | 2026-07 系统报告和定位备忘。`docs_spec/README.md` 仍指向当时的文件名，以本页和 `RESEARCH_CONTROL.md` 为准 |
| [plan_full/](plan_full/) | 更早的 TCCN、雷达和远程实验计划 |
| [STATUS.md](STATUS.md) | 2026-05 的 RL 训练流水状态，不是当前 VQA 编码实验的入口 |

原始预测、协议和统计仍在本地 `paper/outputs/<实验目录>/`。总账只保存索引，不代替那些原始文件。
