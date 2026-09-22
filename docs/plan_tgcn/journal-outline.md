# Journal Outline — TCCN Extension

> 用途:章节级骨架 + 字数 / 图表 / 引用预算 + 与会议版的逐节映射。D5–D11 的写作就照这本"图纸"填。

总目标:**≤ 14 双栏页**,≈ 11 000–13 000 词正文,≥ 70 篇引用,8+ 图,4+ 表。

---

## 0. 全文页面预算

| 节 | 页预算 | 词预算(双栏估) | 主图 | 主表 |
|---|---|---|---|---|
| §I Introduction | 1.25 | 1100 | — | — |
| §II Related Work | 1.5 | 1400 | — | — |
| §III System Model | 2.0 | 1800 | Fig.1 系统示意(沿用 + 多 UAV 重绘) | — |
| §IV Methodology | 2.25 | 2000 | Fig.2 网络架构;Algo.1 MA-HPPO 伪代码 | — |
| §V Theoretical Analysis | 1.0 | 800 | — | — |
| §VI Experiments | 4.0 | 3500 | Fig.3–9(7 张) | Table I–IV(4 张) |
| §VII Discussion | 0.5 | 500 | — | — |
| §VIII Limitations | 0.25 | 250 | — | — |
| §IX Conclusion | 0.25 | 250 | — | — |
| References | 1.0 | — | — | — |
| **合计** | **~14** | ~11 600 | **9** | **4** |

---

## 1. 章节细化

### §I. Introduction

| 段 | 内容 | 来源 |
|---|---|---|
| 1 | 低空经济 + UAV 兴起 + VQA 在 UAV 场景的价值 | 会议版 ¶1–2,扩展 |
| 2 | 语义通信 / DeepSC-VQA 的范式优势 | 会议版 ¶3,简化 |
| 3 | 现有 UAV-MEC 资源调度文献的两大局限:① 未联合优化轨迹 + 资源 + 语义符号;② 仅考虑静态 / 单 UAV / 无扰动场景 | **新写** |
| 4 | 本文方案概述 + 4 条 contribution(对应 `journal-target.md` §2.2) | **新写**(替换会议版 3 条) |
| 5 | Paper organization | **新写** |

> 核心:贡献从 3 条扩到 **4 条**,第三条为新增的"理论分析",第四条把"仿真"升级为"comprehensive comparison + robustness"。

### §II. Related Work(独立成章 — 新增)

5 桶组织,**避免 paper-by-paper 罗列,按方法论流派写**:

1. **Semantic & task-oriented communication**(DeepSC, DeepSC-VQA, semantic entropy / rate)≈ 14 篇
2. **UAV-assisted MEC & resource allocation**(trajectory + offloading + power)≈ 16 篇
3. **Hybrid-action reinforcement learning**(PDQN, PADDPG, HPPO, MPDQN)≈ 12 篇
4. **Multi-agent RL for wireless / UAV networks**(MAPPO, HAPPO, MADDPG, CTDE)≈ 14 篇
5. **Robust / adversarial RL in communication**(jamming-aware, dynamic env)≈ 10 篇 + 综述类 4 篇

**写作模式**:每桶以一段开头("One line of work …"),最后一段定位本文相对每桶的差异。

### §III. System Model

| 子节 | 内容 | 旧 / 新 |
|---|---|---|
| A. Network Model | BS + MEC + ≤4 UAV + ≤9 UE,通信图 | **扩**(从 1 UAV / 3 UE 扩到 multi) |
| B. Channel Model | LoS/NLoS + 时变小尺度衰落(每 K 步重采样) | **扩**(加时变) |
| C. Semantic Transmission Modeling | DeepSC-VQA + 语义熵 + 语义速率 | 沿用,公式编号重排 |
| D. Task Arrival & UE Mobility | Poisson 任务到达 + 慢速 random walk | **新写** |
| E. Delay & Energy Consumption Model | 飞行 + 悬停 + 传输 + MEC 计算 | 沿用并扩 multi-UAV |

### §IV. Methodology

| 子节 | 内容 | 旧 / 新 |
|---|---|---|
| A. Dec-POMDP Formulation | 全局 / 局部 obs、joint hybrid action、reward | **升级**(MDP → Dec-POMDP) |
| B. Joint System Consumption Metric | `O(t)=α·T̃+(1−α)·Ẽ`,沿用 | 沿用 |
| C. Action Space | 每 UAV 一组 hybrid(连续:轨迹 + 功率;离散:信道 + symbol),UE 仍只有 symbol + channel | **扩** multi |
| D. Piecewise Reward | 4 分支 + curriculum on α | **扩**,加 curriculum |
| E. MA-HPPO with CTDE | 共享 critic 用全局 obs;各 UAV 独立 actor;参数共享 / 不共享两版 | **新写** |
| F. Training Procedure | Algo.1 伪代码,clipped surrogate + GAE + entropy bonus | **新写**(Algo 1 重做) |

### §V. Theoretical Analysis(新增 — 半页 to 1 页)

- **Proposition 1**(Unbiased policy gradient under factorized hybrid policy)
  - 证明骨架:把策略写成 `π(a|s) = π_d(a_d|s) · π_c(a_c|s, a_d)` 或独立分解,显式给出 score function;引用 Sutton & Barto / Schulman 2015。
- **Proposition 2**(Monotonic improvement under clipped surrogate)
  - 证明骨架:复用 PPO 单调改进引理(TRPO bound),在 hybrid 因子化下 KL 拆为 KL_d + KL_c,代入 clip 上界。
- **Complexity**:
  - 单步 actor 前向:O(L · d²)+ K_d·O(|A_d|);
  - 训练每 iter:O(N · T · K · L · d²),N agents,T horizon,K epochs,L layers。
- **Sample-complexity remark**:沿用 Schulman 2017,加 hybrid action 的常数因子讨论。

> **写作策略**:不是要给 self-contained 严证,而是要让审稿人相信 hybrid PPO 的收敛性来自 PPO 收敛性(可论证)+ 因子化策略的合法性(可论证)。

### §VI. Experiments(主战场)

#### A. Setup
- **Simulation parameters**(Table I,扩自会议版 Table I,加多 UAV / 任务到达 / 时变信道)
- **VQA_table 来源声明**:DeepSC-VQA 离线训练,SNR 网格 [-10:5:20] dB,K_uav ∈ {394, 788, 1576, 2364, 3152},K_usr ∈ {2,4,6,8,10},版本 hash + 注脚
- **Hardware / training cost**:GPU 型号、训练 wall-clock、参数量
- **Baselines**: MA-HPPO(ours)、MAPPO-hybrid、HAPPO、Hybrid-SAC、PADDPG、PDQN、Greedy 上界

#### B. Numerical Results — 7 个子节,每个对应 1 张图或 1 张表

| Sub | 标题 | 图 / 表 | 主张 |
|---|---|---|---|
| 1 | Convergence | **Fig.3** 7-曲线 with std 阴影 | MA-HPPO 收敛最稳、最高 |
| 2 | Main Comparison | **Table II** 7×5 主对比表(cost / delay / energy / similarity / success rate) | MA-HPPO 在所有指标占优 |
| 3 | Multi-UAV Scaling | **Fig.4** n_uav ∈ {1,2,4} × n_usr ∈ {3,6,9} 网格热图 / 折线 | 收益随规模增长 |
| 4 | Trajectory Visualization | **Fig.5** multi-UAV 2D + 3D 协同 | 学到避让 / 分工 |
| 5 | Semantic Similarity CDF | **Fig.6** 动态信道下的 CDF | 高相似度尾部更厚 |
| 6 | Ablations | **Table III** 4 项消融 | piecewise reward / Dueling / 共享 encoder 各贡献多少 |
| 7 | Sensitivity | **Fig.7** α / ξ_th / G_th / B 扫描 | 在合理区间内稳健 |
| 8 | Robustness | **Fig.8**(a) 信道扰动 ±dB,(b) UE 数 ± | 退化曲线最平 |
| 9 | Adversarial Jamming | **Fig.9** jamming 强度 sweeping | MA-HPPO 不被打崩 |
| 10 | Large-scale UE | **Table IV** n_usr ≥ 10 sensitivity | 可拓展性 |
| 11 | Wall-clock & Complexity | 与 Table I 合并 / 或单独 mini-table | 训练成本可接受 |

#### 写作要求(每个子节)
1. 一句"This experiment tests whether [claim]"
2. 实验设置 1–2 行
3. 观察 + 解读("Fig.X(a) shows … which demonstrates …")
4. 把负结果 / 反预期也写出来,不藏

### §VII. Discussion

- 为什么 MA-HPPO 在 hybrid + multi-UAV 下能赢:① 因子化策略 + clip 兼顾两类动作稳定性;② CTDE 让 critic 看到全局状态,缓解 nonstationarity;③ 分段 reward 把"语义任务可行性"硬约束转为可微梯度信号。
- 与 MAPPO-hybrid 的差距来源:reward shaping + Dueling 离散头。
- 跨域泛化:训练在 (n_usr=6) 上、测试在 (n_usr=9) 的迁移性能。

### §VIII. Limitations(新增,必须写)

1. 单 BS / 单 cell 假设
2. UAV 高度固定 + 简化空气动力学
3. VQA_table 离散化误差(SNR / K 网格分辨率)
4. 离线训练的 LUT 在新数据集上的泛化未验证
5. 训练样本 / 算力代价高于经典优化方案
6. jamming 模型为白噪声扰动,不含智能干扰对手

### §IX. Conclusion

- 1 段总结四条贡献
- 1 段 future work:异构 UAV、跨域 / 跨任务 generalization、在线学习 / meta-RL、与生成式语义通信结合

---

## 2. 旧→新映射(逐元素)

| 会议版元素 | 期刊版命运 |
|---|---|
| Fig.1 系统示意 | **重绘** 加 multi-UAV |
| Fig.2 收敛曲线 | **重跑** 7 baseline 替换 |
| Fig.3 主对比柱图 | **替换为 Table II + Fig.4** |
| Fig.4 UAV 轨迹 | **重做** multi-UAV 版,保留单 UAV 在附录 |
| Fig.5 相似度 CDF | **重跑** 含动态条件 |
| Table I 系统参数 | **扩**(加 multi-UAV / Poisson / 时变项) |
| Algo.1 伪代码 | **重写** for MA-HPPO |
| 引用 [1]–[16] | 完整保留并 verify;再补 ≈ 54 篇至 ≥ 70 |
| Section I 三条贡献 | **替换为 4 条** |
| Future work 段 | **削掉**多 UAV / 动态条件部分(已在期刊版正文实现),换为新展望 |

---

## 3. 引用预算与桶分布(目标 ≥ 70)

| 桶 | 目标条数 | 必含的代表文献(候选,D5–D6 验证) |
|---|---|---|
| Semantic & task-oriented comm | 14 | Xie 2022 ISAC, Yan 2024 TWC QoE-DeepSC, Qin 2024 MNET, [Conference paper §I refs 4–7] |
| UAV-MEC & resource allocation | 16 | Wang 2025 TCCN low-altitude, Hu 2025 TCOM rate-splitting, Hwang 2023 TCOM DRL semantic, Liu 2024 jamming UAV |
| Hybrid-action RL | 12 | Bester 2019 PDQN, Fan 2019 P-DQN, Xiong 2018 PADDPG, HPPO orig refs |
| Multi-agent RL & MAPPO | 14 | MAPPO 2021, HAPPO 2022, MADDPG, CTDE survey |
| Robust / adversarial RL | 10 | Lillicrap, Pinto 2017 robust RL, jamming-aware RL refs |
| 综述 / background(VQA, low-altitude econ, 6G semantic) | 4–6 | Antol 2015 VQA, 6G semantic survey 2024 |

> **铁律**:每条引用 D7–D8 写作时即时验证,不在 D13 才补;`[CITATION NEEDED]` 占位符在 D13 必须清零。

---

## 4. 图 / 表与代码-数据归属表

| 资产 | 由哪段代码生成 | 数据源 | 何时定稿 |
|---|---|---|---|
| Fig.1 system | TikZ / draw.io | — | D1 |
| Fig.2 architecture | TikZ | — | D7 |
| Fig.3 convergence | `plot/train_metrics.py` | 训练 log(D5–D7) | D7 |
| Fig.4 multi-UAV scaling | `plot/scaling.py`(新写) | sweep run(D5–D7) | D9 |
| Fig.5 trajectory 2D/3D | `plot/traij.py`(扩 multi) | eval rollout | D9 |
| Fig.6 similarity CDF | `plot/plot_paper/simcdf.py` | eval log | D9 |
| Fig.7 sensitivity | `plot/sensitivity.py`(新写) | sweep run(D8–D10) | D10 |
| Fig.8 robustness | `plot/robust.py`(新写) | sweep run(D8–D10) | D10 |
| Fig.9 jamming | `plot/jamming.py`(新写) | sweep run(D8–D10) | D10 |
| Algo.1 MA-HPPO | LaTeX algorithmic | — | D7 |
| Table I parameters | LaTeX manual | cfg | D7 |
| Table II main comparison | `plot/maketable.py`(新写) | 训练 log | D7 |
| Table III ablations | `plot/ablation_table.py`(新写) | 消融 run | D10 |
| Table IV large-scale UE | `plot/scale_table.py`(新写) | 大规模 run | D10 |

---

## 5. 写作风格 checklist(D5–D11 反复对照)

- [ ] 5 句式 abstract(achieved / why-hard / how / evidence / headline number)
- [ ] Intro 在 1.25 页内交代 4 条贡献
- [ ] 每个 §VI 子节先写 "We test whether …"
- [ ] 所有图 caption 自包含,不依赖正文
- [ ] 误差棒方法(std)显式标注
- [ ] 单位 / 方向箭头(↑/↓)出现在所有 metric 列
- [ ] 不出现 "may / can / could" 类 hedging,除非真有不确定性
- [ ] 不出现填充词("actually / very / really / basically")
- [ ] 一致术语:semantic similarity / similarity / relevance 三选一,固定一个
- [ ] Limitations 至少 6 项
- [ ] References 全部 verified,且**包含 conference paper [1]** 作为来源声明

---

## 6. 风险触点(在大纲层面)

| 风险 | 大纲应对 |
|---|---|
| §V 理论被认为"不深" | 备好"discussion-style + sketch + 引用 PPO 单调性"作 fallback,不写空头大定理 |
| §VI 实验图过多挤页 | Sensitivity / large-scale UE 可压成单图多子图;adversarial jamming 单图 |
| Related Work 桶之间重复 | 每桶结束前一句"differing from …",显式区分 |
| Multi-UAV 实验若无显著 gain | 改用 "robust under perturbation" 作 headline,不强行卖 multi-UAV |
