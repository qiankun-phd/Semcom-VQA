# 实验总账

更新：2026-09-22 UTC。入口：[研究总控](RESEARCH_CONTROL.md) · [网络与结果](NETWORK_RESULTS.md)。

EXP 编号是本次补录的稳定索引，不代表新增实验、论文实验序号或原始目录改名。首批覆盖最近九组工作；未登记的早期工作需先查原目录，不能默认为未做。

“执行状态”与“证据状态”分开：完成训练不代表达标，完成开发评测不代表独立测试通过。下表的“保留/不扩量”是研究建议，不是删除、搬迁文件或终止进程的操作。

## 1. 快速查重表

|ID / 工作|训练或评测单位|执行状态|证据状态与用途|
|---|---|---|---|
|EXP-001 多问题拟合与 320 控制|32 训练图/115 问；24 诊断图/85 问|已完成|正确问题门控未提升；保留作诊断，不能再称该控制未做|
|EXP-002 同图双目标切换|8 训练图；10 诊断对|已完成|合成定位训练拟合好，诊断迁移弱；无 VQA 结论|
|EXP-003 类别匹配扩展定位|64 训练对；32 诊断对|已完成|目标切换泛化弱；不能归因于真实问题措辞|
|EXP-004 真实 VQA + COCO 框辅助|1024 训练图；128 验证图；59 复用诊断图|已完成|定位 BCE 小幅正向；该轮未测 VQA 增益|
|EXP-005 600 图真实码流评测|600 图/题，六类均衡|已完成|普通神经压缩优于本轮 JPEG；问题门控未超过普通编码|
|EXP-006 对齐后直接任务量化图|32 训练图/115 问；24 开发图/85 问|已完成|实现与梯度检查通过；2k/4k 正确问题未胜均匀量化|
|EXP-007 预算条件教师码图网络|扩展至 256 图；旧 300 图保留集评测|已完成|正确问题 2k/4k 均低于均匀基线；复用 EXP-005 子集，不是另一批全新测试|
|EXP-008 区域动作选择器|100 开发图；80/20 划分；三种子|已完成并复核|无实际准确率收益；接收器训练重叠、空间映射问题限制结论|
|EXP-009 Qwen 压缩输入适配第一阶段|600 训练图；120 开发图；新 300 图封存|训练及开发评测完成；新测试未做|4k 主要对照未达成增益；不是问题引导联合训练|
|EXP-010 TDIUC 共同信道基线适配|2400 训练图/题；120 开发图；新 300 图仍封存|基线训练、开发诊断和 144 次信道 smoke 完成|RSVQA 48–52/120，T24/48/96 为 50/52/50；可靠性待复核，完整 SNR 未做|

准确率、字节数及网络细节统一列于[网络结果索引](NETWORK_RESULTS.md)，本页保留假设、执行和证据出处。

## 2. 各项记录

### EXP-001：问题门控能否利用同图不同问题？

- 方法：已有 question/image-only 门控分别续训 345 次更新，3 轮；基础编解码器和 Qwen 冻结。包含原图、长边上限 320 的等比例缩放、普通编码、正确/错配/空问题控制；缩放控制不是强制正方形输入。
- 数据：32 图/115 个真实问题训练，24 图/85 问诊断。统计独立单位不能把 85 问直接当作 85 张独立图。
- 结果：诊断正确问题由 65/85 降到 63/85；普通编码保持 66/85；原图 70/85、320 缩放 67/85。对照中部分预测复用，不能将全部汇总行数称为新增推理次数。
- 决定：当时的 960 图、三种子扩展未启动。不能把此结果解释为只是“没有做过小样本拟合”。
- 证据：[报告](../paper/outputs/question_guided_fit_diagnostic_20260919/REPORT.md) · [决定](../paper/outputs/question_guided_fit_diagnostic_20260919/DECISION.md) · [协议](../paper/outputs/question_guided_fit_diagnostic_20260919/protocol.json)。

### EXP-002：可控定位提示能否切换关注对象？

- 方法：8 张训练图，每张两个不重叠 COCO 对象；使用合成 Locate 提示。paired BCE 与 BCE+contrast 两组各 400 次更新，单种子。
- 结果：训练 raw 分数双方向正确为 8/8；诊断 raw 为 1/10。最终 gate 的有效诊断对中分别为 1/8 和 0/8。
- 限制：没有加载 Qwen，未测 VQA、码率或能耗；训练可拟合与诊断可泛化不同。
- 证据：[分析](../paper/outputs/question_gate_target_switch_pilot_20260920/analysis-output/analysis-report.md) · [协议](../paper/outputs/question_gate_target_switch_pilot_20260920/protocol.json)。

### EXP-003：增加类别匹配样本能否修复定位迁移？

- 方法：从既有 2400/120 图池构造 64 个训练对、32 个诊断对；raw-only/final-gate 两组各 200 次更新。2400 是来源池，不是实际训练对数。
- 结果：诊断 raw 严格双正向为 0/32；final gate 为 2/32。
- 限制：仍是合成定位问题；不直接检验真实 VQA，也不足以证明“真实问题与定位提示不一致”是失败根因。
- 证据：[分析](../paper/outputs/question_gate_matched_adapt_pilot_20260920/analysis-output/analysis-report.md) · [协议](../paper/outputs/question_gate_matched_adapt_pilot_20260920/protocol.json)。

### EXP-004：真实 VQA 与 COCO 框弱监督能否学到相关区域？

- 方法：修复对应前版中心裁剪/坐标与损失缩放问题后，用真实问题可匹配的正 COCO 框监督；1024/128 图，另有 59 张复用诊断图。
- 训练：只训练 local_match 的 33,024 个参数。real-question 与 image-only 各 3 轮、3072 次更新；仅平衡框 BCE，没有回答 CE。
- 结果：验证 BCE 正确问题 0.557937，错配 0.579433，空问题 0.584586，单独空问题训练控制 0.576433。最终门控内外差约 0.066820 对 0.066377，差异很小。
- 限制：定位损失改善不能证明 VQA 改善；这一轮没有新的 VQA/字节/能耗终点。修复只对已验证版本成立，不能外推到全部后续缓存。
- 证据：[分析](../paper/outputs/question_gate_real_vqa_aligned_pilot_20260921/analysis-output/analysis-report.md) · [协议](../paper/outputs/question_gate_real_vqa_aligned_pilot_20260921/protocol.json)。旧 `question_gate_real_vqa_large_pilot_20260920` 结果须连同其实现问题阅读。

### EXP-005：定位训练是否能转化为端到端收益？

- 方法：冻结 EXP-004 检查点；600 张历史排除后的 COCO val 图，各一个 TDIUC 问题，六类各 100。五种编码条件 × 两预算 + 原图，共 6600 条预测。
- 预算：主要 8000 B、次要 4000 B，计上下行总量；问题组包括 16 B 问题封装与 UTF-8 文本，其他组没有该下行。不要与后续纯图像预算混写。
- 结果：8k 普通神经 466/600，图像专用门控 461，正确问题 460，错配 458，JPEG 438；原图 473。问题对图像专用门控修复 4 题、损失 5 题。
- 限制：JPEG 为固定的、与答案无关的多分辨率网格策略，不代表最优 JPEG 上限。干净码流解码全部成功不等于无线交付率；NVML 不支持，能耗为空。
- 证据：[协议](../paper/outputs/question_gate_e2e_holdout600_20260921/protocol.json) · [统计](../paper/outputs/question_gate_e2e_holdout600_20260921/summary.json) · [完成记录](../paper/outputs/question_gate_e2e_holdout600_20260921/complete.json)。本次将远端已完成统计同步到本地，未重跑。

### EXP-006：对齐后的逐区域量化能否由回答损失训练？

- 方法：32 图/115 问训练，24 图/85 问开发；全视野坐标对齐，5×5、16 级问题量化图，真实码流前向与近似梯度训练。345 次更新。
- 训练：问题码图网络；冻结 TinyCLIP、基础编解码器及 Qwen。目标为回答 CE、码率项与重建 MSE。
- 结果：2k 均匀 64/85，正确问题 63，错配 61；4k 均匀 66，正确问题 64，错配 65。五项几何测试及三次 CE 梯度检查通过，不等于准确率目标通过。
- 证据：[结果](../paper/outputs/question_quant_alignment_20260921/REPORT.md) · [复核](../paper/outputs/question_quant_alignment_20260921/RESULTS_REVIEW.md) · [协议](../paper/outputs/question_quant_alignment_20260921/protocol.json)。

### EXP-007：预算条件网络能否学到答案辅助搜索的有利码图？

- 方法：教师码图回归与正确/错配排序，先 128 图，再扩到 256 图；扩展阶段 55 个敏感图问预算对，42/13 训练验证。40 轮、单种子。没有在此轮直接联合更新 Qwen。
- 评测：旧 300 张图、六类各 50；2k/4k × 均匀/正确/错配，共 1800 条预测。这 300 图全部属于 EXP-005 的 600 图清单，不能累计为另一批全新测试图。上游排除审计包含本项目 standard Qwen LoRA 的 train.json 等已知训练/开发清单；这不保证基础模型预训练未见过这些图。
- 结果：2k 为 232/225/227，4k 为 238/232/232，顺序均为均匀/正确/错配。换问题通常改变重建图，但没有带来正确率净增益。
- 限制：教师候选在筛选的 22 对中恢复 16 对，是使用答案的搜索，不是部署算法的成绩，也不是整个数据集的理论上限。旧缓存几何需按该版本追溯，不能套用 EXP-006 的修复结论。
- 证据：[五步报告](../paper/outputs/question_quant_five_step_20260921/five-step-execution-report-zh.md) · [扩展协议](../paper/outputs/question_quant_budgeted_network_expanded_20260921/protocol.json) · [300 图分析](../paper/outputs/question_quant_budgeted_independent300_20260921/analysis-output/analysis-report.md)。

### EXP-008：直接选局部增强动作能否实现问题收益？

- 方法：七种动作；100 张候选评测图分 80/20，两个预算；平坦与分层选择器，各三种子。上限 100 轮，按开发集选最佳检查点，不代表使用最后一轮。
- 结果：分层集成的均匀、正确问题、错配问题都为 32/40；正确问题选了四次局部动作，但没有改变正确性。40 对仅来自 20 张验证图。
- 可靠性限制：100/100 图在 standard Qwen LoRA 训练集中；99/100 非方图的中心裁剪特征被直接当成原图坐标；局部正例训练仅来自 9 图、验证仅 2 图。
- 决定：原报告“五步完成”只是执行完成，不是泛化证明。优先采用后续可靠性复核的解释；不支持直接扩量重跑同一配置。
- 证据：[可靠性复核（解释优先）](../paper/outputs/question_region_selector_hierarchical_20260921/reliability-audit-20260921.md) · [原报告](../paper/outputs/question_region_selector_hierarchical_20260921/five-step-report-zh.md) · [协议](../paper/outputs/question_region_selector_hierarchical_20260921/protocol.json)。

### EXP-009：接收端是否能适应压缩重建图像？

- 方法：已有 TDIUC standard LoRA 为起点；从既有 LoRA 训练池取 600 图训练，另设 120 图开发、新 300 图封存。六类分别每类 100/20/50；开发/测试的历史排除审计覆盖 45 份清单、6134 个图像 ID。2026-09-21 本地清单复核：新 300 图与旧 300 图交集为 0，新 120 图与旧 300 图交集也为 0。
- 训练：clean_control 与 compressed_mixed 各 300 次优化器更新、梯度累积 8；各 2400 次样本呈现。前者原图四遍，后者每图原图/2k/4k/8k 各一次；样本顺序控制一致。
- 结构：只更新 Qwen 语言层 q/k/v/o 的 rank-8 LoRA，5,898,240 个可训练参数；视觉主干与基础神经编解码器冻结；问题引导发送器尚未训练。
- 结果：三接收器 × 四输入条件 × 120 图 = 1440 开发预测。主要 4k 比较 mixed 93/120，clean_control 95/120，修复 3、损失 5。2k 的正向变化是次要开发结果。
- 状态：训练、开发评测、依赖校验和适配器哈希核对完成；新 300 测试未评估。六次 smoke 更新已丢弃；构建对齐缓存不等于已训练问题门控。
- 证据：[报告](../paper/outputs/qwen_compression_joint_stage1_20260921/REPORT.md) · [协议](../paper/outputs/qwen_compression_joint_stage1_20260921/protocol.json) · [统计](../paper/outputs/qwen_compression_joint_stage1_20260921/summary.json) · [状态](../paper/outputs/qwen_compression_joint_stage1_20260921/status.json)。

### EXP-010：普通神经压缩与已有 VQA 系统的共同信道比较

- 授权：用户同意统一数据、不同 SNR 的系统比较；2026-09-22 UTC 实际启动，服务器 `lab-s2`，独立目录 `outputs/tdiuc_common_channel_baselines_20260922`。
- 相对旧实验新增：旧 RSVQA/T-DeepSC 在 VisDrone 上适配，不能直接拿来与现有 TDIUC Qwen 比。此次重新适配两基线，后续在共同物理信道比较，不再同时扩张问题引导或路由。
- 网络：冻结 ordinary CompressAI 与 standard Qwen；训练 RSVQA 融合分类头，以及 T-DeepSC 视觉/文本/信道/解码模块。冻结 RSVQA ResNet152/UniSkip、T-DeepSC COCO Faster R-CNN 区域提取。分类答案 194 类，只来自训练集。
- 数据审计：复用 standard Qwen 的 2400 图/题（六类各 400），EXP-009 的新 120 开发图；训练/开发/测试图像 ID 互斥，训练与开发文件 SHA 无交集。新 300 图仅查身份，不读答案。开发 OOV 为 2/120，不删题。
- 开发训练：每个模型单种子 7，AdamW 1e-4、batch32、最多 40 轮/3000 步，至少 20 轮后 patience10；开发 CE 选检查点。RSVQA 循环原图和 2/4/8k JPEG，T24/48/96 在线共同 Rician 信道训练。
- 通信：共同 K=6 dB、Es/N0、CSI、图像/seed 对应衰落和噪声；数字路径真实 LDPC/QPSK/CRC，T 路径保留连续符号。拟定 -5~20 dB、间隔 2.5 dB、20 个信道种子；不同源编码不通过填充伪造同资源。
- 验收：先无噪声、视觉置零和同类图置乱检查是否真正使用图像；再六图真实信道 smoke。诊断未通过则复核，不能把弱基线当作新方法优势。完整 SNR 扫描和 sealed test 不在自动启动阶段内。
- 已完成：代码语法检查；NumPy/Torch 信道一致性、噪声前缀、LDPC 包长、CRC、与 pyldpc 解码参考对照；RSVQA/T24/T48/T96 的 batch32 有限损失/梯度检查。Smoke 更新全部丢弃，没有用于正式模型。
- 执行完成：supervisor 为 DEVELOPMENT_COMPLETE，实际耗时 1355.48 秒；RSVQA 40 轮，T24/T48 各 23 轮，T96 22 轮。开发 CE 选定轮次为 40/13/13/12。
- 结果：RSVQA 2k/4k/8k 为 52/52/48（分母 120）；T24/48/96 为 50/52/50。同集标准 Qwen 的普通神经重建复用结果为 87/93/92，原图 98/120。单种子、开发检查点选择、资源和预训练不等，不能当作独立/等资源优势。
- 诊断：RSVQA 8k 正确图像 48/120、同类置乱 49/120，REVIEW_REQUIRED；T 的正常输入仅比最强控制高 3–4 题。6 图/1 seed/两表示/三预算/四 SNR 的 144 次真实链路 smoke 通过，不是 VQA-SNR 曲线。完整 SNR 扫描、新 JPEG+Qwen 推理、独立测试和能耗仍未做。
- 汇报：已生成同集统计与两张图，并发布 [Notion 导师阶段汇报](https://app.notion.com/p/3e37a69b8f53814e83dfc8998459efbc)；[本地汇报](../paper/outputs/tdiuc_common_channel_baselines_20260922/2026-09-22--tdiuc-common-channel--r01--supervisor-brief.md)。本次汇总未启动任何新训练或实验。
- 证据：[协议](../paper/outputs/tdiuc_common_channel_baselines_20260922/PROTOCOL.md) · [运行代码](../paper/outputs/tdiuc_common_channel_baselines_20260922/run_experiment.py) · [监督器](../paper/outputs/tdiuc_common_channel_baselines_20260922/launch.py)。实时状态在服务器 `supervisor.json` / `status.json`，本地副本是拉取时的快照。

## 3. 新实验登记模板

```text
ID / 日期 / 名称：
待验证假设（仅一句）：
相关旧实验 ID / 本轮新增变量 / 为何不是重复：
网络版本、检查点、训练模块、冻结模块：
训练/开发/测试清单及与所有已训练模块的数据交集：
主要基线、主要预算、真实字节计量边界：
主要指标、预先约定的继续/停止条件：
训练轮数、优化器更新数、种子、算力上限：
原始输出目录 / 协议、代码、权重与清单哈希：
执行状态：计划 / 运行 / 完成 / 失败 / 未启动
证据状态：开发信号 / 未达目标 / 独立验证通过 / 暂不可解释
结果：分子/分母、准确率、实际字节；未测指标填“未测”
限制、是否改变原计划、下一步决策及日期：
```

不要把新 300 图保留测试用于选择超参数、预算或检查点。需要解封时先冻结方案与主要对比，并单独登记。
