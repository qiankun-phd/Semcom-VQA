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
| [code/bubbles_semsched/](code/bubbles_semsched/) | 新方向：U-space 巡检回传的语义调度。代码、图表脚本、结果文件和复现包，见下一节 |

编译文稿：

```bash
cd paper
latexmk -pdf -outdir=build main.tex
```

`paper/build/` 不提交。图缺失时文稿仍能编译，对应位置会显示占位。

## BUBBLES 语义调度（进行中）

2026 年 10 月起的新方向，和上面的 TGCN 文稿是两条线，互不依赖：按固定 4D 航迹巡检的无人机把"相对参考图的变化"回传地面，调度器逐时隙决定只发符号层还是加发图像 token、以及信道匹配，目标是在每架次召回要求下提高空域容量。

全部内容在 [code/bubbles_semsched/](code/bubbles_semsched/)：

- `repro/`：复现包。仿真与学习代码、按实验分组的命令脚本、已训练的调度策略检查点。命令脚本由当时的运行记录生成，并与结果文件里的元数据核对过。
- `figs/`：出图出表脚本和当前的图表。只重出图表不需要运行仿真。
- `res/`：图表读取的结果文件。

研究问题、目前的结果和已知局限写在 [code/bubbles_semsched/README.md](code/bubbles_semsched/README.md)，复现方法写在 [code/bubbles_semsched/repro/README.md](code/bubbles_semsched/repro/README.md)。

## 不在这个仓库里

仓库是公开的，所以下面这些只留在本地工作区：

- `paper/outputs/`：预测、码流、权重、图像缓存和第三方数据，约 9 GB
- `code/hppo-uav/`：DI-engine 分支和远程启动脚本
- `docs/LOCAL_HOSTS.md`：实验室主机对照
- `reviews/`、会议幻灯片、编译好的 PDF 包

`code/bubbles_semsched/` 里只有调度策略的检查点（约 3 MB）；检测器权重、图像编码器权重和训练日志没有放进来，链脚本注释里的主机地址已隐去。

大数据集也没有放进来。本地路径写在 `docs/LOCAL_HOSTS.md`。
