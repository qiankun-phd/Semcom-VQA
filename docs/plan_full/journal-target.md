# Journal Target Lock — TCCN Extension

> 用途:锁定目标期刊与故事主线,作为 14 日历日扩稿计划的"宪法"。所有写作 / 实验 / 引用 决策都必须能映射回本文件。

---

## 0. 一行结论

把会议稿 *"Hybrid Reinforcement Learning for Resource Allocation in VQA-Oriented UAV Semantic Offloading"* 扩为 IEEE TCCN regular paper:**Multi-UAV、多用户、动态信道与对抗扰动条件下,基于 MA-HPPO + 分段奖励 + 收敛性分析的 VQA 语义卸载与资源调度框架**,实验 / 理论 / 鲁棒性三方面相对会议版增量 ≥ 30%。

---

## 1. 期刊与版式

| 项 | 锁定值 |
|---|---|
| 期刊 | **IEEE Transactions on Cognitive Communications and Networking (TCCN)** |
| 投稿类型 | Regular Paper |
| 页限 | **≤ 14 双栏页**(含图表与参考文献) |
| 模板 | IEEEtran journal class(`\documentclass[journal,onecolumn,twoside]{IEEEtran}` → 提交版 twocolumn) |
| 期刊侧重(用于 framing) | learning-/AI-driven cognitive networking,resource adaptation under uncertainty,semantic & task-oriented communication |
| 评审形式 | 单盲(保留作者信息) |
| 期刊政策(会议→期刊) | 必须有 substantial new contribution(≥ 30%),并须在 cover letter 列出 delta |
| 重投容忍 | 一般 1–2 轮 major / minor revision |

---

## 2. 故事主线 — Route A: Multi-UAV + Robust HPPO

### 2.1 一句话贡献(One-sentence contribution)

> "We extend hybrid PPO to a multi-UAV semantic offloading network for VQA tasks, jointly optimizing trajectory, channel, power, and semantic symbol selection under dynamic traffic, time-varying channels, and adversarial jamming, with provable monotonic improvement and consistent gains over single- and multi-agent RL baselines."

### 2.2 四条 contribution(待 D5 抛光,先锁结构)

1. **System-level**: A multi-UAV, multi-user MEC-assisted DeepSC-VQA network with **dynamic task arrival, mobile UEs, and time-varying air-to-ground channels**, formulated as a **Dec-POMDP** with a unified delay-energy joint cost.
2. **Algorithmic**: **MA-HPPO**(centralized-training decentralized-execution Hybrid PPO)over a vast hybrid action space(continuous: trajectory + power; discrete: channel + semantic symbol per UAV/UE), with a **piecewise reward** that explicitly couples communication-resource decisions to semantic-task fidelity.
3. **Theoretical**: A **convergence discussion** for hybrid-action PPO — propositions on policy-gradient unbiasedness under the factorized hybrid policy, monotonic improvement under the clipped surrogate objective, and a per-iteration computational complexity analysis.
4. **Empirical**: Comprehensive comparison against diagnostic baselines
   and the LUT-grade Greedy reference across Greedy-vs-QRS
   cost/fidelity, component/MoE ablation, channel-loss robustness, and
   fixed-M large-UE pressure. Current evidence does **not** support a
   monolithic MA-HPPO dominance claim; the target journal evidence is
   LearnRisk-QRS remote multi-seed cost/fidelity and floor-safety
   trade-off evidence.

### 2.3 期刊故事的一句"so-what"

> 在低空经济与 6G 语义通信的交汇处,UAV-assisted VQA 卸载是少数同时受 **资源 / 任务 / 信道 / 安全** 四重不确定性约束的场景;本文给出了在该场景下 hybrid-action MARL 的**首个**收敛性可论证、跨多种扰动条件下仍稳定的解。

---

## 3. 与会议版的差异化(Delta ≥ 30%,Cover-Letter 用)

| 维度 | 会议版(6 页) | 期刊版(≤14 页) | Delta 类型 |
|---|---|---|---|
| 系统模型 | 1 UAV + 3 UE + 静态 + 静信道 | 多 UAV(≤4)+ 多 UE(≤9)+ Poisson 任务 + UE 移动 + 信道时变 | **新模型** |
| 算法 | 单 HPPO | **MA-HPPO (CTDE) + 分段奖励 curriculum** | **新算法** |
| 理论 | 无 | **§V Convergence Discussion + 复杂度命题** | **新章节** |
| 基线 | PADDPG, PDQN(2) | + MAPPO-hybrid, HAPPO, Hybrid-SAC, Greedy 上界(共 7) | **新基线** |
| 实验图表 | 4 fig + 1 table | 8+ fig + 4+ table(主对比 / 多 UAV 扩展 / 消融 / 敏感度 / 鲁棒性 / 对抗 jamming / 大规模 UE / 复杂度) | **大幅扩充** |
| 鲁棒性 | 未涉及 | 信道扰动 ±dB / UE 数 ± / α curriculum / **对抗 jamming** | **新内容** |
| 消融 | 未涉及 | 4 项(piecewise reward / Dueling vs Discrete / shared vs separate encoder / fixed vs learnable σ) | **新内容** |
| 引用规模 | 16 篇 | ≥ 70 篇,5 桶组织 | **大幅扩充** |
| Limitations | 无 | 独立小节 | **新章节** |
| 可复现性 | 路径硬编码 | LUT cfg 化 + 公开 commit hash + 可复现声明 | 工程改进 |

> **Delta 量化目标(D12 审计填表)**:页数 +8、图 +4、表 +3、章节 +2(§V, Limitations)、新算法 1、新基线 4、新实验类别 5。

---

## 4. Cover Letter 骨架(D14 填实)

```
Editor-in-Chief, IEEE TCCN

We submit the manuscript "<title>" for consideration as a regular paper.

This work substantially extends our conference paper [1] (presented at <venue>),
with the following major new contributions:

  (i)   System model extended from single-UAV / static to multi-UAV with
        dynamic task arrival, UE mobility, and time-varying channels;
  (ii)  A new MA-HPPO algorithm (CTDE) replacing single-agent HPPO;
  (iii) A new theoretical analysis section (Section V) including
        convergence and complexity propositions;
  (iv)  Four additional baselines (MAPPO-hybrid, HAPPO, Hybrid-SAC, Greedy);
  (v)   Five new experimental dimensions (multi-UAV scaling, ablations,
        sensitivity, robustness to channel/UE perturbations, adversarial
        jamming, large-scale UE generalization).

Total new content vs the conference version exceeds 50% by page count and 70%
by figure/table count. The conference paper is cited as [1] and clearly
acknowledged.

We confirm this manuscript has not been submitted elsewhere.

Authors: ...
Corresponding author: ...
```

---

## 5. 作者与署名顺序(沿用会议版)

1. Qiankun Zhang(MPU,one-author)
2. Zelin Ji(SIAS, UESTC)
3. Yue Liu(MPU,通讯作者)
4. Ze Song(MPU)
5. Zhijin Qin(Tsinghua)

> 期刊版若有显著新工作贡献人,D11 前在 cover letter 与 acknowledgments 显式声明。

---

## 6. 数据 / 代码可复现性策略

| 资产 | 处理 |
|---|---|
| `VQA_table.mat` | **真实 DeepSC-VQA 离线训练数据**,在 §VI Setup 与 Reproducibility 注脚说明:训练数据集、SNR 网格、symbol 网格、版本与 hash;若开源受限,提供合成 surrogate 公式作为 fallback |
| 代码 | 在 `journal-ext` 分支整理,提交时给 GitHub commit hash,同步 README 跑通流程 |
| 模型 ckpt | 至少 release MA-HPPO 主结果对应的 `ckpt_best.pth.tar`(哈希记录于 §VI) |

---

## 7. 时间锚点(对应 14 日历日)

| 锚点 | 日 | 必交 |
|---|---|---|
| Outline 锁定 | D1 | `journal-target.md`(本文件)+ `journal-outline.md` |
| 多 UAV env + MA-HPPO 跑通 | D3 | 训练曲线初见上升 |
| 7 算法基线全接入 | D4 | 同图可对比 |
| 主表 + 收敛图首版 | D7 | 数字落位 |
| 全文一稿 | D11 | LaTeX 完整 |
| Self-review + Citation 全核 | D13 | 0 placeholder |
| 提交 | **D14** | PDF + cover letter |

---

## 8. 不可妥协项(任何阶段都不能砍)

- **引用零幻觉**:所有 cite 必须 Semantic Scholar / CrossRef 双重验证;不能验证就不写。
- **delta ≥ 30%**:D12 审计若不达标,优先补 §V 理论与新实验类别,而不是稀释会议版内容。
- **数据真实性声明**:VQA_table 来源不能含糊。
- **Limitations 章诚实**:单 BS 假设、固定 UAV 高度、LUT 离散化误差、训练 / 推理算力代价,必须写。
