# 语义通信参考实现代码考据与借鉴分析（五仓库）

> 2026-07-04。对 MA-DeepSC (JSAC 2025) 及其 README 引用的 4 个参考实现逐行分析的汇总。
> 所有仓库均已本地克隆精读，逐条论断附代码出处。我们的系统：UAV 场景 VQA 语义通信，
> 按问题类型自适应选证据等级（s1 检测器 token / s2 图像 JPEG+LDPC；M2 uncoded analog
> baseline），接收端 Qwen2-VL 答题，指标 VQA 精度-vs-SNR（AWGN/Rayleigh/Rician, −5→20 dB）。

## 0. 五仓库一览与谱系

| 仓库 | 论文 | 形态 | 能否直接跑 | 预训练权重 |
|---|---|---|---|---|
| MA-DeepSC | JSAC 2025 | 6 个 Colab notebook | ✗（绑作者 Drive、`nn.GDN` 必崩、定量评估代码被删） | 无 |
| PADC | TWC 2023 | 16 个 .py，~1700 行 | 接近（需重训+修 2-3 个小 bug） | 无 |
| ADJSCC | TCSVT 2022 | 8 个核心 .py | 接近（TF2.1 老环境） | 无 |
| DeepJSCC-f | JSAIT 2020 | 单文件 jscc.py 898 行 | ✗（TF1.15 老栈） | 无 |
| SJTU Task-Unaware | JSAC 2022 | 84 文件 | ✗（missing import、死代码） | 仅 MNIST 一组 |

**谱系（代码考据结论）**：
- MA-DeepSC 的全部信道工具（`Power_norm`/`AWGN_channel`/`Fading_channel`/`mask_gen`/`Channel_VLC`/`AF_block`）**逐函数抄自 PADC `models.py`**，连"P=1 时符号功率=2"注释和 `nn.GDN` 笔误都一并继承（PADC models.py:20,32 该笔误在未使用的 block 里所以不崩，MA-DeepSC 抄去用了才崩）。
- AF 模块的原始出处是 **ADJSCC**（util_module.py:41-48，SE-block + concat SNR 标量）。
- MA-DeepSC 的 MDAN 是 **SJTU 仓库 CycleGAN 域适应**（solver_svhn_mnist.py）的 StarGAN 化推广。
- MA-DeepSC README 引 DeepJSCC-f，但**反馈机制在其代码中不存在**；反馈的代码真身只在 DeepJSCC-f 仓库（jscc.py:425-527，约 40 行核心）。
- **ADJSCC-V（mask 变速率）的完整训练代码在 PADC 仓库**（MA-DeepSC 缺失的部分），引用应指 TWC 2023。

## 1. 物理层实现对比（公平性核对要点）

| 项 | PADC/MA-DeepSC | ADJSCC | DeepJSCC-f | SJTU |
|---|---|---|---|---|
| 功率归一化 | 整帧 L2，每实维功率 P=1（复符号=2）| 逐码字，每复符号功率=1 | 逐样本，每实符号功率=1 | **无归一化**（噪声对齐信号功率）|
| 衰落 h 归一化 | **E\|h\|²=2 未归一化 → SNR 偏乐观 3dB** | E\|h\|²=1 ✓ 正确 | h∼CN(0,1) ✓，块衰落（每图一个 h）| 无衰落 |
| CSI/均衡 | 完美 CSI 零迫 `y=(hx+n)/h` | slow_fading 不均衡（解码器隐式学）；`_eq` 版完美 ZF | 不均衡，多轮共享同一 h | — |
| 训练 SNR | 逐样本 randint(0,28) ✓ | 逐样本 U(0,20) 连续 ✓ + 每 epoch 重采 | **固定单点**（每 SNR 一模型）| 固定 10dB/3dB |
| 信道在环 | ✓ | ✓ | ✓ | ✓（但 SJTU 一半脚本的"AWGN"实为 `np.random.random` **均匀非零均值噪声**，功率差 ~4.8dB）|

**对我们**：M2 的功率归一化公式按 PADC 代码行为写（每实维功率 P；其注释"P=1/√2 得单位功率"是笔误，实际应 P=1/2）；引用其 fading 数字需注明 +3dB 乐观偏差；我们含估计误差的 3 信道设置严格强于所有五者，写 related work 可点出。

## 2. 各仓库对我们的核心借鉴（按落地优先级）

### P1 s1 可变 token 数 —— PADC 嵌套 mask 协议（原始出处+完整代码）
- **前缀/嵌套掩码而非随机丢弃**（models.py:378-384）：训练/标定永远"保留前 t 个"，迫使表示按重要性有序排列 → 单模型任意速率 + 优雅退化（successive refinement）。
- **逐样本二维随机化 (SNR, t)**（DeepJSCC_V_train_CIFAR10.py:56-57）。
- **功率按实际传输符号数归一化**（Power_norm_VLC, models.py:317-322）：每 token 恒定符号功率，能量随 t 线性——token 数-能量-精度公平计费的正确写法。
- **接收侧零填充定形状**（models.py:172,406）：单 VQA 头兼容所有 t。
- s1 冻结提取器的等价物：按显著性排序 token 后 keep-top-t，t 作为 LUT/预测器采集的扫描维度。
- 勿继承：`AWGN_channel_VLC` batch 共享噪声（models.py:329）。

### P2 LUT → 参数化预测器 —— PADC OracleNet 配方
角色同构：冻结传输系统 + 环境条件 → 可达质量代理，再做约束选择。
- **标签 = 逐样本在线 rollout**（Oracle_train_CIFAR10.py:44-68）：随机抽 (SNR,CR)、冻结系统真跑一次、逐图记质量。我们建 LUT 的原始逐样本记录就是现成训练集，格心平均是信息损失。
- **单次信道实现 + 回归 = 学条件均值**；我们 0/1 正确性标签 → **MSE 换 BCE**，输出为 accuracy 概率 + 校准（温度缩放）；LUT 退化为校准验证基准。
- **决策变量作条件输入，昂贵特征只算一次**（OracleNet.py:38,44）：question 特征一次前向，扫遍 (service, t, snr) 候选。预测器优于 LUT 处：SNR 连续插值 + per-question 个性化（LUT 只到 question_type 粒度）。
- **两阶段冻结训练**：不要求系统可导——对含 Qwen2-VL 的不可导管线是唯一可行模板。
- 风险提示：PSNR 面平滑好学；我们的 accuracy 有 cliff，悬崖邻域需加密采样 + 必须校准。

### P3 AF token 门控 + 失配矩阵实验 —— ADJSCC
- **AF 精确结构**（util_module.py:41-48）：GAP → concat SNR 标量 → FC(C/16) relu → FC(C) sigmoid → 通道乘。移植到 token (B,N,d)：沿 token 维 mean-pool 得 (B,d)，门控 d 维对全 token 共享；SNR 建议归一化到 [−1,1] 再拼（原版拼原始 dB 值，CNN+BN 能吸收，VLM 特征尺度敏感）；每层编码头后插一个、信道投影层后不插。
- **可选 token 级扩展**（原版没有）：加一路逐 token 标量门控 → 门控值排序即 token 重要性排序 → 与 P1 的 mask 变速率天然衔接。
- **训练配套必须成套**：逐样本连续随机 SNR + 信道在环 + 每 epoch 重采；**保留 train_mix 消融**（同随机化但无 AF，bdjscc_cifar10.py:54）——证明"门控有效、不只是随机化的功劳"，审稿人必问。
- **失配矩阵实验**（bdjscc_cifar10.py:90-117）可升级为我们独有贡献：把"训练 SNR × 测试 SNR"推广为"**策略假定 SNR × 真实 SNR**"二维热力图——LUT 用 SNR_assumed 选等级、实际信道 SNR_true，对角线=完美 CSI 上界，离对角衰减=对 SNR 估计误差的鲁棒性。ADJSCC 评估永远喂真实 SNR、从不测条件化输入出错的情形——这是我们可补的缺口。数字 s2 的 cliff 与模拟 s1 的平滑退化在失配图上形成本质对比，恰好支撑 cliff-effect 叙事。每格点 ≥10 次信道实现取平均。

### P4 增量证据传输（置信度门控补传）—— DeepJSCC-f 四模式
- **发端镜像收端做门控**（jscc.py:487-499）：它用共享权重 decoder 在发端模拟收端重建（不进 loss 纯推断）。对应我们：UAV 端轻量代理/校准置信度预测器模拟"Qwen 拿当前证据能否答对"，不足才触发 s2 补传。
- **增量以收端已有状态为条件**（jscc.py:462）：补传不是重发整图——以第一轮 token 摘要 + SNR 为条件选 ROI/QP/分辨率。
- **收端证据累积而非替换**（OutputsCombiner, jscc.py:407-422）：补传后 prompt 同时带 token 证据和图像（消融点："补传后丢弃 token" vs "累积"）。
- **带噪反馈消融接口**（jscc.py:473-477）：反馈请求消息也走无线链路，完美/带噪/丢包反馈对 acc-SNR 的影响可做消融，有文献先例。
- **达标即停的可变长评估**（TargetPSNRsHistogram, jscc.py:109-131）：报告"达到目标 VQA 精度所需平均信道用量分布"——自适应增量传输的核心卖点呈现范式。
- 不可移植：符号级输出反馈（连续 y 反馈，前提全模拟传输；我们 s2 数字链路软信息已丢弃，反馈只能语义级）；端到端可微（我们含 JPEG/LDPC/VLM 三个不可微环节）；像素残差（我们的"缺失信息"在语义域）。
- 块衰落提醒：论文 System Model 必须显式声明 token 轮与补传轮是否同一相干块。

### P5 域漂移适配（UAV 天气/光照）—— SJTU 思想，换掉 GAN 实现
- **可迁移核心："改输入、冻链路"**——漂移在链路最前端消化，下游编码器和接收端大模型零改动（solver_svhn_mnist.py:80-84,187-189）。
- **他们自己的推荐配置关闭像素级 cycle 重建、只靠类条件判别器保任务语义**（readme.md:22）——"语义级充分性优先于像素保真"的现成证据链，且其 proposed 配置需要新域标签（solver:308）。
- 非 GAN 替代（UAV 机上可行）：确定性光度校正（直方图匹配/Retinex/白平衡）≈ 无对抗的 G21，零幻觉风险；检测器级适配（BN 统计量自适应/TENT 式只更新归一化层）；校准层适配（在线重标定 s1/s2 选择阈值——他们框架里不存在，可作差异化贡献）。
- **评估协议整体照搬**（readme.md:18-67）：三方案对比 = 不适配 / 前端适配+冻结链路 / 整链重训。
- 像素级 CycleGAN 本身不适用：开放域航拍会增删小目标，直接破坏计数充分统计量。

## 3. task-unaware 理论呼应（互补性论述的引用姿势）
- SJTU 文确立"发射机不知任务 → 只能优化任务无关代理目标"设定，与我们 UAV 端（YOLO+编码不知具体问题）同构，可作**上位框架**引用。
- 但其代码无信息瓶颈/充分统计量形式化（纯 MSE 重建代理 + 接收端主导微调；互信息度量 `mu()` 定义了从未调用，save_figure.py:130-156）。
- 我们的定位："把单一 observable-information 代理细化为**任务族条件下的分层充分统计量**"——计数任务 I(count; tokens)≈H(count)（检测计数是充分统计量）、存在性任务图像分支保 I(presence; image)。DPI 链条要我们自己写；其混合损失的存在恰是"纯重建代理对任务次优"的反面证据，为我们按问题类型选证据等级提供动机。

## 4. 引用/复现注意事项汇总
1. MA-DeepSC：公开代码不支撑定量复现（评估代码被删、仅 3 个 SNR 点）；信道代码转抄自 PADC。
2. PADC fading +3dB 乐观偏差；功率注释 1/√2 笔误（按代码行为 P=1/2 才是单位复符号功率）。
3. ADJSCC 是 AF 与失配实验的原始出处，信道归一化正确，引 AF 引它不引 MA-DeepSC。
4. DeepJSCC-f `x_tst.repeat(10)` 返回值未赋值——"每图 10 次信道实现"实际无效（jscc.py:733），其数字方差比论文声称大。
5. SJTU 仓库工程质量差（missing import、评估泄漏、max-窗口精度统计），数值不宜对标复现，只借协议与结构思想。
6. 五仓库全部无可用预训练权重 → 都不能作为可跑 baseline；我们的 M2 uncoded analog 仍是"DeepSC 式模拟传输"的受控代表。
