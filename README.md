# Semcom-VQA

面向无人机视觉问答的语义通信研究记录，以及 IEEE TGCN 文稿源文件。

当前主线不是问题引导编码。问题门控在已完成评测里没有超过同预算的普通神经压缩。系统主体仍是：预训练神经图像编码、实际码流、RGB 重建、固定 Qwen 回答。尚未成立的贡献候选，是在回答质量约束下选择压缩强度和信道保护。证据边界以 [docs/RESEARCH_CONTROL.md](docs/RESEARCH_CONTROL.md) 为准，不要用更早的稿件标题代替它。

## 从哪里读

| 路径 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 文档分层：当前记录、文稿、历史计划 |
| [docs/RESEARCH_CONTROL.md](docs/RESEARCH_CONTROL.md) | 目标、已证实的事实、还不能说的结论 |
| [docs/EXPERIMENT_LEDGER.md](docs/EXPERIMENT_LEDGER.md) | 实验账本 |
| [docs/NETWORK_RESULTS.md](docs/NETWORK_RESULTS.md) | 网络连接和同口径结果 |
| [paper/](paper/) | TGCN LaTeX：`main.tex`、`sections/`、`refs.bib`、`figures/` |
| [code/vqa_semcom_v0/](code/vqa_semcom_v0/) | 主实验代码。配置是 JSON，虽然文件名是 `configs/v0.yaml` |

编译文稿：

```bash
cd paper
latexmk -pdf -outdir=build main.tex
```

`paper/build/` 不提交。图缺失时文稿仍能编译，对应位置会显示占位。

## 不在这个仓库里

仓库是公开的，所以下面这些只留在本地工作区：

- `paper/outputs/`：预测、码流、权重、图像缓存和第三方数据，约 9 GB
- `code/hppo-uav/`：DI-engine 分支和远程启动脚本
- `docs/LOCAL_HOSTS.md`：实验室主机对照
- `reviews/`、会议幻灯片、编译好的 PDF 包

大数据集也没有放进来。本地路径写在 `docs/LOCAL_HOSTS.md`。
