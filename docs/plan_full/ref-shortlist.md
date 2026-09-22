# Reference Shortlist — TCCN-Adjacent Reconnaissance (D1)

> 用途:作为 D5–D6 写作时的"格式 / 体量 / 故事 framing"参照,而非引用本身。每条必须在 D5–D6 经 Semantic Scholar / CrossRef 验证后才进入 `paper/refs.bib`。
> 现在只收 5–8 条最相关的近 2 年文献,并标注用途,**不**抄袭其内容。

## 目标期刊近邻文献(待 D5–D6 验证)

### 必读(直接对标本文场景)

| # | 候选标题 / 关键词 | 关联点 | 来源 / URL | 状态 |
|---|---|---|---|---|
| 1 | Wang Z, Zhang J, et al. "Toward realization of low-altitude economy networks: Core architecture, integrated technologies, and future directions," **IEEE TCCN**, 2025 | 直接对标 TCCN 期刊的低空经济 framing,可作为 §I 第一段引子 | doi: 10.1109/TCCN.2025.3601015(会议版引文 [1]) | 沿用,验证版本号 |
| 2 | Liu S, et al. "UAV-enabled semantic communication in MEC under jamming attacks," **IEEE TWC**, 2024 | 对抗 jamming + 语义 + UAV-MEC,与 §VI Jamming 实验设置对照 | https://dl.acm.org/doi/abs/10.1109/TWC.2024.3454073 | 验证作者 / 卷号 |
| 3 | Multi-UAV multi-hop networking via MARL + LLM, 2025 (preprint) | 多 UAV CTDE 训练拓扑参考 | https://www.researchgate.net/publication/391707002 | 验证是否已发表 |
| 4 | Multi-Agent RL UAV Swarm Communications Against Jamming, **IEEE TWC**, 2023 | 多 UAV + 抗 jamming 的故事先例 | https://dl.acm.org/doi/abs/10.1109/TWC.2023.3268082 | 验证作者 / DOI |
| 5 | Hwang H, et al. "DRL-driven dynamic resource allocation for task-oriented semantic communication," **IEEE TCOM**, 2023 | 任务导向语义通信 + DRL,与本文 §VI 主对比表对照 | 会议版引文 [10] doi: 10.1109/TCOMM.2023.3274145 | 沿用,验证 |

### 备读(用于桶分布)

| # | 候选 | 用于哪一桶 |
|---|---|---|
| 6 | Xie H, Qin Z, et al. "Task-oriented multi-user semantic communications," **IEEE JSAC**, 2022 | 桶 1:语义通信 |
| 7 | Yu C, et al. "The surprising effectiveness of PPO in cooperative MARL" (MAPPO), **NeurIPS**, 2022 | 桶 4:Multi-agent RL |
| 8 | Bester C J, et al. "Multi-pass Q-networks for hybrid action," 2019 | 桶 3:Hybrid-action RL |
| 9 | Pinto L, et al. "Robust adversarial reinforcement learning," **ICML**, 2017 | 桶 5:Robust RL |
| 10 | Antol S, et al. "VQA: Visual question answering," **ICCV**, 2015 | 背景 |

## TCCN 期刊体量参照(待 D5 量化)

从 2025 年 TCCN 已发表文章观察(D1 浅尝):
- Regular Paper 多在 12–14 双栏页之间;
- 引用普遍 50–80 篇;
- §VI Experiments 通常占 3–4 页,含 ≥ 6 张图;
- §V Theoretical Analysis 不强制,但有则加分;
- Limitations 章节存在但简短(0.3 页内常见)。

## 用法约束(自约束)

1. **任何条目在 D5 前不得复制到 `refs.bib`**。先验证 DOI / 作者 / 年 / 卷 / 页。
2. 阅读时只看 abstract + intro 的 contribution 列表 + experiments 章节体量,**不**抄袭其方法或公式。
3. 与本文的差异点(Multi-UAV + 真实 VQA-LUT + hybrid action + 形式化收敛讨论)必须在 §II Related Work 各桶末段显式标出。

## 信息来源(D1 reconnaissance)

- [IEEE Transactions on Cognitive Communications and Networking 期刊页](https://www.comsoc.org/publications/journals/ieee-tccn/ieee-transactions-cognitive-communications-and-networking-submit)
- [Liu et al. UAV-enabled semantic comm under jamming, TWC 2024](https://dl.acm.org/doi/abs/10.1109/TWC.2024.3454073)
- [Multi-UAV MEC trajectory planning + resource allocation 综述, arXiv 2024](https://arxiv.org/html/2409.17882v1)
- [MARL UAV swarm vs jamming, TWC 2023](https://dl.acm.org/doi/abs/10.1109/TWC.2023.3268082)
- [SuperClaude / 综述 Survey on UAV Control with MARL, MDPI Drones 2025](https://www.mdpi.com/2504-446X/9/7/484)
