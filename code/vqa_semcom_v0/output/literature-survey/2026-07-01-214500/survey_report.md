# Literature Survey: VQA/任务导向语义通信系统设计与有效性评估（面向 UAV/低空场景）

**Date:** 2026-07-01 | **Papers Found:** 130 unique (123 relevant) | **Date Range:** 主检索窗口近 1 年（2025-07 起），另含关键早期奠基文献 | **Search Queries:** 5 (arXiv/Semantic Scholar/DBLP) + 5 (web)

## Paper Summary Table

| # | Title | Authors | Year | Venue | Notes |
|---|-------|---------|------|-------|-------|
| 1 | [Task-Oriented Multi-User Semantic Communications for VQA Task (MU-DeepSC)](https://arxiv.org/abs/2108.07357) | Xie, Qin et al. | 2021/22 | IEEE WCL/JSAC | **Seminal**：首个 VQA 语义通信系统；CLEVR；acc-vs-SNR；基线=error-free / JPEG75+Huffman+LDPC1/3+16QAM |
| 2 | [Goal-Oriented Semantic Communication for Wireless VQA (GO-SG)](https://arxiv.org/abs/2411.02452) | Liu et al. | 2024 | — | **最近亲缘**：传 scene-graph 三元组+bbox（数字 token 路线）；基线=全图传输/未排序SG/GT；指标=accuracy+延迟分解+分题型 |
| 3 | [Scene Understanding Enabled Semantic Communication with Open Channel Coding](https://arxiv.org/abs/2501.14520) | — | 2025 | — | 场景理解+开放信道编码，数字语义路线延续 |
| 4 | [How to Evaluate Semantic Communications](https://arxiv.org/abs/2309.04891) | — | 2023 | — | **评估方法论**：任务指标+语义指标(ViTScore)，CBR=k/n 带宽归一化 |
| 5 | Deep JSCC for Wireless Image Transmission | Bourtsoulatze et al. | 2019 | IEEE TCCN | **Seminal**：cliff effect 叙事起点；JPEG/JPEG2000+LDPC 在低 SNR 崩溃、JSCC 平滑退化 |
| 6 | [Adaptive Semantic Token Communication for Transformer-based Edge Inference](https://arxiv.org/abs/2505.17604) | — | 2025 | — | token 数+embedding 维度自适应；Lyapunov 随机优化做资源分配；任务=目标检测 |
| 7 | [Entropy-and-Channel-Aware Adaptive-Rate SemCom with MLLM-Aided Feature Compensation](https://arxiv.org/abs/2501.15414) | — | 2025 | — | 熵+信道感知的变速率语义编码，MLLM 补偿 |
| 8 | [STCC: Unified Source-Channel Semantic Token Coding](https://arxiv.org/abs/2606.11819) | — | 2026 | — | token 级统一信源信道编码框架 |
| 9 | Task-Oriented SemCom in Large Multimodal Models-Based Vehicle Networks | — | 2025 | IEEE TMC (19c) | LMM 做接收端的任务导向语义通信，车联网 |
| 10 | Task-Oriented SemCom With Importance-Aware Rate Control | — | 2025 | IEEE Comm. Letters | 特征重要性→速率控制 |
| 11 | Bandwidth and Power Allocation for Task-Oriented SemCom | C. Liu et al. | 2022 | — | **资源分配奠基**：任务导向带宽/功率分配 vs 等分/静态基线 |
| 12 | [Semantic-Aware Resource Allocation Based on DRL for 5G-V2X HetNets (SARADC)](https://arxiv.org/abs/2406.07996) | Shao et al. | 2024 | — | PPO 优化信道/功率/占空比/语义符号长度；指标=语义谱效 HSSE、语义吞吐 |
| 13 | RL-Driven Semantic Compression Model Selection and Resource Allocation | Lin et al. | 2025 | IEEE ISCC | RL 选压缩模型 + 资源分配（与"选表示等级"最接近的 RA 工作） |
| 14 | DRL- and Information-Bottleneck-Enabled Task-Oriented SemCom | — | 2025 | IEEE JSAC | IB 理论 + DRL 联合 |
| 15 | [Importance-Aware Resource Allocation for Collaborative Task-Oriented SemCom (iCoTASC)](https://arxiv.org/abs/2606.29052) | — | 2026 | — | **方法论镜像**：离线效用函数+**预计算 LUT**+在线查表精调，免重训练的信道自适应语义资源分配 |
| 16 | [Hybrid RL for Resource Allocation in VQA-Oriented UAV Semantic Offloading](https://doi.org/10.1109/WCNC65185.2026.11555390) | **Zhang Q. et al.（本组前作）** | 2026 | IEEE WCNC | DeepSC-VQA 编码 + MEC 推理 + HPPO 联合轨迹/无线资源/语义卸载，优化能耗-时延 |
| 17 | [Efficient Onboard Vision-Language Inference in UAV-Enabled LAE Networks](https://arxiv.org/abs/2510.10028) | — | 2025 | — | UAV 机载 VLM（14B DeepSeek-R1, 5-6 tok/s)+LLM 增强优化 |
| 18 | [UAV-Assisted Cooperative Edge Inference for LAE via MoE-based Hierarchical DRL](https://arxiv.org/abs/2605.19290) | — | 2025 | GLOBECOM | 空地协同边缘推理，MoE 分层 DRL |
| 19 | RL-based RA for Multi-Task Multi-Codebook Semantic Communications | — | 2026 | Physical Communication | 多任务多码本语义资源分配 |
| 20 | [Foundation Model-Based Adaptive Semantic Image Transmission](https://arxiv.org/abs/2509.23590) | — | 2025 | — | 基础模型驱动的动态环境自适应语义图传 |

---

## Theme Clusters

### Theme 1: 端到端可学习 VQA/任务导向语义收发机（模拟特征路线）
**Summary**: 以 MU-DeepSC/DeepSC-VQA 为源头：发端 DNN 抽取语义特征直接映射为信道符号（analog JSCC），收端联合解码执行任务。评估固定为 accuracy-vs-SNR（AWGN/Rayleigh/Rician，-5→20 dB），对照 JPEG+LDPC 传统栈与 error-free 上界。
**Key Papers**: #1, #6, #8, #9。
**Contribution**: 确立了"任务精度即通信指标"的范式与 cliff-effect 对比法。

### Theme 2: 数字/符号化语义传输（token、场景图、可解释表示）
**Summary**: 传离散语义符号（scene-graph 三元组、bbox、语义 token）而非模拟特征，兼容现有数字协议栈，可解释、可缓存。GO-SG（#2）是与本项目 s1 路线最接近的工作：按问题相关性排序取 top-N 三元组，报告 accuracy+延迟分解+分题型。
**Key Papers**: #2, #3, #8, #10。
**Contribution**: 证明面向问题的符号化证据可用一小部分字节达到甚至超过全图精度。

### Theme 3: 有效性评估方法论
**Summary**: 标准配方 = 任务指标（accuracy）+ 传输代价（CBR 或 bytes/query、延迟分解）+ 三信道 SNR 扫描 + cliff-effect 图 + 分题型柱状图；基线四件套 = error-free 上界 / 传统 SSCC(JPEG+LDPC+QAM) / 已有语义方案(DeepSC 类) / oracle。
**Key Papers**: #4, #5, #1, #2。
**Contribution**: 给出可复现、可横向比较的实验协议。

### Theme 4: 语义感知资源分配（DRL/约束优化/查表）
**Summary**: 把"语义符号长度、压缩模型选择、特征维度"作为与带宽/功率并列的分配变量；方法从 PPO（#12）、分层/混合动作 RL（#16）、IB+DRL（#14）、Lyapunov（#6）到**离线效用+LUT 在线查表**（#15）。iCoTASC 与本项目 LUT 思路几乎同构，说明"离线测量质量模型 + 在线轻量调度"是 2026 年的前沿共识。
**Key Papers**: #11-16, #19。
**Contribution**: 语义层资源（表示等级/符号数/模型选择）正式进入资源分配问题空间。

### Theme 5: UAV/低空经济场景的语义通信与边缘推理
**Summary**: LAE 叙事下的 UAV 机载 VLM、空地协同推理、A2G 信道（LoS 概率模型）、能耗/时延联合优化；普遍用分层或 MoE DRL。真实航拍 VQA 数据评估仍然稀缺（多用 CLEVR/通用数据集）。
**Key Papers**: #16, #17, #18, 及 air-ground MEC offloading (VTC 2025)。
**Contribution**: 提供了本项目场景合法性与 A2G 信道建模的锚点。

---

## Research Gaps

1. **表示等级（evidence level）作为一等资源变量**：现有工作固定一种语义表示（特征/token/图），在其内部调速率；"cache→token→ROI→全图"的离散表示等级选择+跨等级切换机制没有系统研究——这是本项目最独特的自由度。
2. **真实 VLM 接收端 + 真实航拍数据的有效性验证**：文献主流仍是 CLEVR/合成数据+小任务模型；用真实 VLM（Qwen2-VL 级）在 VisDrone 类小目标密集航拍图上做 acc-vs-SNR 的公开结果几乎没有。
3. **资源分配所用质量模型的可信度**：多数 RA 论文的"语义质量-资源"函数是解析假设或小规模拟合，缺少带置信区间的实测标定（measured LUT）以及"仿真环境 ↔ 实测系统"的闭环验证。
4. **语义缓存/新鲜度（AoI）作为语义资源**：cache 命中/过期对任务精度的影响在语义通信框架内少有量化。
5. **约束式 RL 在语义 RA 中的应用**：现有工作几乎都用加权奖励，硬性 deadline/质量/空域约束下的 Lagrangian/约束 RL（如本项目 TCH-PPO）鲜有对照。

---

## Cross-Domain Findings

- **自适应推理（early-exit / model cascade）→ 表示等级选择**：计算机视觉的级联推理"够用即停"思想迁移到通信侧即"最小充分证据"服务等级，理论工具是充分统计量/信息瓶颈（#14）。
- **Lyapunov 随机优化（#6）**：为 token 级速率适配提供无模型的排队稳定性保证，可作为 RL 之外的对照方法。
- **LLM-guided optimization / MoE-HDRL（#17, #18）**：用大模型或 MoE 做高层决策、传统优化做低层，佐证本项目"高层任务调度 + 低层资源分配"的分层结构。
- **AoI（信息年龄）文献 → 语义缓存新鲜度**：freshness bin 可以严格化为 AoI 阈值模型，使 cache 等级的质量衰减有理论出处。

---

## Innovation Proposals

### Proposal 1: 实测 LUT 驱动的证据等级自适应 VQA 语义通信系统（系统主线）
**Description**: 用真实 VLM（Qwen2-VL）+ 真实退化信道在 VisDrone-VQA 上实测「问题类型 × 证据等级 × SNR」精度面，替换 v0 合成 LUT；以"问题条件路由（symbolic→token, perceptual→image）"为核心机制，证明零参数语义规则即可达到数据标定选择器的性能。
**Feasibility**: 数据=VisDrone(已有)；算力=单卡 10GB 级（已验证）；新颖性=填补 Gap 1+2，目标 TCCN/JSAC/TWC；周期=大部分已在 182 服务器完成，剩余收尾 2-4 周。
**Potential Weaknesses**: 机制消融已显示多维 LUT 中只有 question_type 承载增益——必须诚实重构贡献叙事，勿夸大 LUT 维度/LCB 的作用。
**Landing Plan**: ① 固化 5 题型 × 3 信道 × 2 VLM 结果与 F1/F2/F4/F5/F6 图；② 补 train/test 划分与统计显著性；③ 写"Evidence–Question Complementarity"原理章节。成功指标=自适应方案在全 SNR 段 ≥ 所有固定等级且字节数 -30%。回退=若审稿质疑规则太简单，用 NL 问题→symbolic/perceptual 分类器泛化。

### Proposal 2: 让 SNR/新鲜度维度"活起来"的系统扩展
**Description**: 当前 s2 采用速率自适应后对 SNR 过于平缓、freshness 无差异，导致除题型外的 LUT 维度失效；引入固定速率数字路径（cliff 场景）、时变语义缓存（AoI 衰减）与 ROI 等级的真实实现，使信道/缓存维度对等级选择产生真实影响。
**Feasibility**: 代码基础已有（naive fixed-rate 路径已建）；周期 3-6 周。
**Potential Weaknesses**: 若做完维度仍不承载增益，则应接受"题型路由"为最终结论。
**Landing Plan**: ① Rayleigh 固定速率 cliff 路径全量跑；② 缓存 AoI 模型接入并实测 s0 精度衰减；③ 重跑消融看 snr/freshness 是否变为 load-bearing。

### Proposal 3: 实测质量面上的约束混合 PPO 语义资源分配（第二阶段）
**Description**: 将实测 LUT 注入 `vqa_resource_env`，用 TCH-PPO（Lagrangian 约束混合动作 PPO）联合分配 UAV 指派/证据等级/带宽/功率/CPU-GPU，对照等分/贪心/oracle/无约束 PPO；卖点=约束满足率与语义效用的权衡，呼应 Gap 3+5。
**Feasibility**: 环境与基线代码已全部就绪（本文件夹），缺的只是实测 LUT 与训练算力；周期 4-8 周。
**Potential Weaknesses**: RL 尚处 smoke 阶段（1 episode、success=0），训练稳定性未知；场景为脚本化 demo，需随机化与多种子统计。
**Landing Plan**: ① 实测 LUT 接入 + 场景随机化；② PPO/TCH-PPO 各 ≥5 seeds 训练至收敛并报 CI；③ 与 5 个启发式基线 + joint greedy oracle 全面对比。

---

## References

See `references.bib` in this directory. Raw search dumps: `../2026-07-01-search/*.json`（130 篇去重原始记录）。

---

**Generated**: 2026-07-01 ~21:45 UTC+8 | **Tool**: literature-survey skill v1.0
