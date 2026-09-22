# 论文一（语义通信有效性）定位材料：口径 + 划界段落 + BibTeX

> 2026-07-05。基于深度调研的六个近邻工作，arXiv 号已逐一 curl 核实。
> 注意：`../paper/` 现有稿是另一条线（Role-Split 多 UAV 卸载，DeepSC-VQA 符号系统）；
> 论文一（M0–M5 对比 + 互补性）**尚无 LaTeX 草稿**，本文档是其 related work 的种子。

## 一、新颖性口径（写作红线）

**禁止**："the first VLM-receiver end-to-end VQA evaluation over wireless channels"
（arXiv 2505.02413 已用 LLaVA + Fisher-Snedecor F 衰落做过端到端 VQA 精度）。

**安全口径**：
> To our knowledge, this is the largest-scale end-to-end VQA-accuracy evaluation of a real
> VLM receiver over coded wireless links (548 aerial images / 5 question types / 3 channels,
> vs. 41 road-scene images in [LMM-VN]), and the first to route between *evidence levels*
> (cached answer / detector tokens / image) by question type rather than selecting regions
> or features within a single modality.

**同样禁止**："first to study VQA as the receiver task"（MU-DeepSC 2021 已确立）；
增量表述锚定在：任务无关发射机 + 冻结商用 VLM + 证据等级路由 + 无差错信道下仍胜整图（发现②）。

## 二、Related Work 划界段落（英文，可直接改写入稿）

### Goal-oriented VQA semantic communication
> MU-DeepSC~\cite{xie2021mudeepsc} pioneered VQA as the receiver task with a jointly trained
> task-specific transceiver; its gains concentrate in low-SNR robustness against separate
> source--channel coding. GO-SG~\cite{liu2024gosg} ranks bounding boxes and scene-graph
> triplets by question keywords, and observes -- consistent with our finding -- that evidence
> formats and question categories are complementary; however it treats full-image transmission
> as the accuracy *upper bound* (trading 4\% accuracy for 65\% latency), offers only manual
> guidelines instead of an automatic selector, and answers with a neuro-symbolic reasoner
> rather than a real VLM. In contrast, we show that question-conditioned evidence routing
> *surpasses* full-image transmission even over an error-free channel, and that a
> zero-parameter semantic rule exactly matches a data-calibrated selector.

### Task-adaptive modality selection
> Park and Yoon~\cite{park2025transmit} select visual semantic modalities per classic CV task
> (objects for classification, layouts for localization, scene graphs for retrieval) with a
> fixed task-to-semantics mapping and a diffusion-based receiver; VQA is not among the tasks
> and no per-question adaptation is learned. LMM-based vehicle networks~\cite{jiang2025lmmvn}
> evaluate a real LLaVA receiver over fading channels but restrict question awareness to
> spatial region selection *within* the image modality on 41 road-scene images. Our work
> instead routes among evidence *levels* per question type, on an order-of-magnitude larger
> aerial benchmark with a unified complex-channel-use accounting across analog and digital
> baselines.

### Token-budget / progressive semantic transmission
> Progressive edge--cloud VLM communication~\cite{devos2026progressive} encodes nested-prefix
> latent tokens with provably monotone reconstruction risk, and task-adaptive scene-graph
> filtering~\cite{park2025transmit} reports that a pruned graph can slightly outperform the
> full graph on retrieval. Our nested top-$t$ detector-token truncation differs in three ways:
> the budget is confidence-ordered *symbolic* evidence rather than learned latents; budget
> sensitivity separates question families by an order of magnitude (comparison saturates at
> $t{=}3$ vs. $t{=}48$ for counting); and moderate truncation *improves* task accuracy
> (+7.7 points on threshold questions) by acting as a false-positive filter -- a non-monotone
> effect absent from reconstruction-oriented progressive schemes.

### （论文二用）Semantic-aware DRL resource allocation
> Context-aware UAV digital semantic communication~\cite{sun2026context} already places a
> continuous semantic compression factor inside a DRL action space (TQC over trajectory,
> compression, relaying). Our controller differs in treating the *discrete evidence level*
> as a first-class routing action coupled with a risk-aware seven-multiplier Lagrangian dual,
> a semantic-feasibility safety projection, and Lyapunov virtual queues driven by a
> measurement-calibrated accuracy lower confidence bound.

## 三、BibTeX（venue 待二次核实项已标注）

```bibtex
@article{xie2021mudeepsc,
  author  = {Xie, Huiqiang and Qin, Zhijin and Li, Geoffrey Ye},
  title   = {Task-Oriented Multi-User Semantic Communications},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2022}, volume = {40}, number = {9}, pages = {2584--2597},
  note    = {arXiv:2108.07357}
}
@misc{liu2024gosg,
  author = {Liu, Sige and Li, Nan and Deng, Yansha and Quek, Tony Q. S.},
  title  = {Goal-Oriented Semantic Communication for Wireless Visual Question Answering},
  year   = {2024}, eprint = {2411.02452}, archivePrefix = {arXiv}
}
@article{park2025transmit,
  author  = {Park, Jeonghun and Yoon, Sung Whan},
  title   = {Transmit What You Need: Task-Adaptive Semantic Communications for Visual Information},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2025},
  note    = {arXiv:2412.13646; DOI 10.1109/JSAC.2025.3623159 待核实}
}
@misc{jiang2025lmmvn,
  title  = {Task-Oriented Semantic Communication in Large Multimodal Models-based Vehicle Networks},
  year   = {2025}, eprint = {2505.02413}, archivePrefix = {arXiv},
  note   = {作者名单待补（arXiv 页面核实）}
}
@misc{devos2026progressive,
  title  = {Progressive Semantic Communication for Efficient Edge-Cloud Vision-Language Models},
  year   = {2026}, eprint = {2604.26508}, archivePrefix = {arXiv},
  note   = {作者名单待补}
}
@misc{sun2026context,
  title  = {Context-Aware Information Transfer via Digital Semantic Communication in UAV-Based Networks},
  year   = {2026}, eprint = {2601.01430}, archivePrefix = {arXiv},
  note   = {作者名单待补}
}
@article{won2025madeepsc,
  author  = {Won, Dongwook and Do, Quang Tuan and Win, Thwe Thwe and Lee, Donghyun and Oh, Junsuk and Cho, Sungrae},
  title   = {Multidomain Adaptive Semantic Communications},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2025}, volume = {43}, number = {7}, pages = {2506--2517},
  doi     = {10.1109/JSAC.2025.3559127}
}
@article{zhang2023padc,
  author  = {Zhang, Wenyu and others},
  title   = {Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2023},
  note    = {卷期页码待补}
}
```

## 四、现有 TCCN 稿（Role-Split 卸载线）的最小补引

其 related work 应补 \cite{sun2026context}（2601.01430）：连续语义压缩因子进 DRL 动作空间的先例，
与 \qmethod 的"DeepSC-VQA 符号基数控制"构成必须划界的近邻——差异点：角色分离慢/快控制、
问题条件语义地板、MoE 快层与 learned-risk veto。
