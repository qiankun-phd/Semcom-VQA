# UAV-VQA 语义通信系统 — 对比实验方案 (Comparison Experiment Plan)

> 目的:把"我们的语义通信系统 vs 传统通信 / 现有 VQA 语义通信"放进同一坐标系,**用领域认可的实验范式证明创新性**。本方案的基线、指标、信道、图表全部对齐 MU-DeepSC、Liu GO-SG、DeepSC、DeepJSCC 等代表作的做法。
>
> 版本:v1 (2026-06-28) ｜ 配套系统方案见 `UAV_VQA_SemCom_System_Spec.md`

---

## 0. 文档范围

本方案只定义**对比实验**(怎么比、和谁比、用什么指标、画什么图)。系统本身的设计(LUT、信道、检测器、Qwen、RL)见系统方案文档。

实验分两大组,职责严格分离:

| 实验组 | 比什么 | 对手 | 目的 |
|---|---|---|---|
| **组 1 — 对外 head-to-head** | 语义通信**传输方案**本身 | 传统通信 + 现有 VQA 语义通信 | 证明"目标导向 + 自适应服务选择"优于现有传输范式 |
| **组 2 — 对内资源分配消融** | RL 双时标资源分配 | 我方静态/阈值/贪心策略 | 证明 RL 分配带来的额外增益(此维度其他系统没有,不对外比) |

---

## 1. 创新主张与待证命题

我们要用实验**证明**的命题(每条对应至少一张图/表):

- **C1(优雅降级 / cliff-effect)**:传统整图传输在阈值 SNR 以下断崖式崩溃;我们的自适应服务选择实现准确率平滑降级。← 招牌图
- **C2(低 SNR 优势)**:在低 SNR / 衰落信道,我方准确率显著高于传统 SSCC 与固定单一服务。
- **C3(目标导向效率)**:相同准确率下,我方传输代价(字节/CBR/时延)远低于整图传输。
- **C4(自适应增益)**:自适应选择优于任何固定服务(s1-only / s2-only),且贴近 Oracle 上界。
- **C5(资源分配增益,内部)**:RL 双时标分配在多流/受限预算下优于静态/阈值/贪心。

---

## 2. 公平对比协议(Fairness Protocol)

> 这是审稿人最先攻击的点。三条铁律保证准确率差异 **100% 归因于通信方案**。

1. **同后端**:所有方法的答案推理统一用 Qwen2-VL-2B(`v1_7_direct_calibrated` 不调用 VLM 的 s1 路除外,但其语义来自同一检测器)。基线绝不换更强/更弱的 VQA 模型。
2. **同信道**:同一套衰落实现(AWGN/Rayleigh/Rician,E[|h|²]=1)、同一 SNR 点、同一随机种子(seed=23)。
3. **同预算**:同一字节/时延预算 τ(默认 `tx_time_budget_s=0.3`,`bandwidth_hz=1e6`)。在"准确率-字节"图里则统一扫预算。
4. **同数据**:VisDrone-DET val,548 图 → presence+counting 任务集(`outputs/tasks/v1_7_tasks.csv`),所有方法跑同一任务列表。
5. **同评分**:presence 用 yes/no 精确匹配;counting 用 `count_tolerance_ratio=0.1` 容差匹配。归一化规则统一(`normalized_prediction`)。

---

## 3. 对比方法定义(标准四方 + 两个上界)

| ID | 方法 | 类别 | 传什么 / 怎么编码 | 在我们框架的位置 | 文献依据 |
|---|---|---|---|---|---|
| **M0** | Error-free 上界 | 上界 | 无噪原图直接喂 Qwen | — | MU-DeepSC / GO-SG 都画 |
| **M1** | 传统 SSCC | 传统通信 | 整图 → JPEG(率自适应到字节预算)→ **LDPC + BPSK/QAM** → 衰落信道 → 重建 → Qwen | = 固定 **s2** | MU-DeepSC(JPEG75+LDPC1/3+16QAM) |
| **M2** | DeepSC 模拟特征 | 现有语义通信 | 紧凑特征 → 功率归一 → **模拟复高斯噪声**(无数字编码)→ 解码喂后端 | 新增 analog 路 | DeepSC / MU-DeepSC |
| **M3** | GO-SG 数字 token | 现有语义通信 | 检测证据(类别+bbox+置信)token → LDPC+BPSK | = 固定 **s1** | Liu GO-SG(2411.02452) |
| **M4** | **我们的自适应选择** | **本文方法** | 按语义效用 LCB 在 s0/s1/s2/s3 间**逐任务选最优** | 核心贡献 | — |
| **M5** | Oracle 逐样本 | 上界 | 天选每任务真实最优服务 | 性能天花板 | 自定义 |

### M2 实现要点(唯一需要新写、且要谨慎对齐的基线)

DeepSC 的精髓是**模拟(analog)传输 + 无数字信道编码**,以此换取"无悬崖、优雅降级"。在我们以 Qwen(吃像素)为后端的设定下,有三种忠实度递减的实现选项,**需在 A 阶段拍板**:

- **M2a(最忠实,成本高)**:训一个轻量 JSCC 自编码器(image→feature→加噪→重建像素)→ Qwen。本质接近 DeepJSCC,要训练。
- **M2b(折中,推荐)**:把 JPEG 压缩后的 DCT/像素系数当**模拟符号**直接过信道(功率归一 + 复高斯噪声,无 LDPC)→ 重建 → Qwen。这就是经典"uncoded analog image transmission"基线,**无需训练**,且能真实展示"模拟无悬崖 vs 数字有悬崖"的对照。
- **M2c(最轻,忠实度最低)**:在检测特征向量(bbox 坐标/置信)上加模拟噪声 → 直接喂语义解码器。仅作 token 路的模拟对照。

> **推荐 M2b**:无训练成本、忠实体现 DeepSC 的"模拟优雅降级"卖点、与 M1(数字整图)形成最干净的"数字 vs 模拟"对照。M2a 留作 reviewer 追问时的升级选项。

---

## 4. 评价指标(Metrics)

对齐 GO-SG / MU-DeepSC / How-to-Evaluate 三件套:

### 4.1 任务准确率(主指标)
- 总体 accuracy;
- **按问题类型**拆分(presence / counting)— Liu Fig.5、MU-DeepSC 都有;
- 95% 置信区间(Wilson),多种子。

### 4.2 传输代价(目标导向核心)
- **payload bytes / query**(已有 `payload_bytes`);
- **CBR(channel bandwidth ratio)= 信道符号数 / 源维度**,符号数 = payload_bits / bits_per_symbol — 对齐 DeepJSCC/How-to-Evaluate 的归一约定;
- **端到端时延**:`t = payload_bits / (B·log₂(1+SNR·|h|²)) + t_detector + t_vlm`(对齐 GO-SG 的 latency 分解)。

### 4.3 鲁棒性
- 准确率随 SNR 下降的**斜率**(越平越好 → 量化 cliff-effect)。

---

## 5. 信道模型与 SNR 扫描

| 项 | 取值 | 说明 |
|---|---|---|
| 信道类型 | **AWGN + Rayleigh + Rician(K=6dB)** | `digital_link.sample_power_gain` 已支持三种;配置层补全 |
| SNR 扫描 | **−5, 0, 5, 10, 15, 20 dB** | 覆盖 GO-SG(−2→8)与 MU-DeepSC(0→20)区间 |
| 带宽 B | 1 MHz | `bandwidth_hz=1e6` |
| 预算 τ | 0.3 s | `tx_time_budget_s` |
| 归一 | E[|h|²]=1 | 公平功率约束 |
| 种子 | 23 + 多种子重复 | 置信区间 |

每种信道画一条 Acc-SNR 曲线族(对齐 MU-DeepSC 的 AWGN/Rayleigh/Rician 三幅图)。

---

## 6. 图表清单(Figures & Tables)

| 编号 | 类型 | X 轴 | Y 轴 | 曲线/分组 | 证明 | 对标 |
|---|---|---|---|---|---|---|
| **F1** | Acc-SNR(×3 信道) | SNR (dB) | accuracy | M1/M2/M3/M4 + M0/M5 | C2,C4 | MU-DeepSC Fig.3 |
| **F2 ★招牌** | cliff-effect | SNR (dB) | accuracy | M1(整图,断崖)vs M4(自适应,平滑) | C1 | DeepJSCC |
| **F3** | Acc-vs-代价 | bytes/CBR per query | accuracy | M1/M2/M3/M4 | C3 | GO-SG Fig.4b |
| **F4** | 按问题类型 | {presence,counting} | accuracy | 各方法柱状 | C2 | GO-SG Fig.5 |
| **F5** | Pareto | latency | accuracy(气泡=字节) | 各方法散点 | C3 | GO-SG Fig.6 |
| **T1** | 主结果表 | 方法 × SNR | accuracy±CI | 全方法 | C2,C4 | 通用 |
| **T2** | 代价表 | 方法 | bytes/CBR/latency | 全方法 | C3 | 通用 |
| **F6**(内部) | 资源分配 | 负载/流数 | 准确率 or 队列长度 | RL vs static/threshold/greedy | C5 | — |

---

## 7. 实验组 2 — 资源分配消融(内部)

> 仅在我方系统内部比,因 DeepSC/GO-SG 无"多流/队列/预算分配"维度,强行对外比不公平。

- **对手**:① 静态固定分配 ② SNR 阈值规则 ③ 贪心(当前最大效用)④ 随机 ⑤ 我方 RL 双时标(PPO+Lyapunov)。
- **场景**:多 UAV 流竞争受限带宽 + 队列动态。
- **指标**:系统总准确率、队列稳定性(Lyapunov drift / 平均队长)、时延违约率。
- **现状**:RL 闭环仍 blocked,属后置里程碑;先用规则版 M4 完成组 1,再补 RL 做组 2。

---

## 8. 现状对齐 vs 待补(Gap Analysis)

| 项 | 状态 | 文件/动作 |
|---|---|---|
| M1 传统 SSCC(整图 JPEG+LDPC) | ✅ 已有 | = s2,`digital_link.transmit_image_rate_adaptive` |
| M3 数字 token(GO-SG 式) | ✅ 已有 | = s1,`build_detector_lightweight_evidence` + `transmit_image_to_path` |
| M4 自适应选择(规则版) | ⚠️ 半成品 | LUT 已在,需写 LCB 贪心选择器(不依赖 RL) |
| 三信道 awgn/rayleigh/rician | ✅ 底层就绪 | `sample_power_gain` 已支持;配置层补扫描 |
| **M2 DeepSC 模拟特征** | ❌ 待写 | 新增 analog 路(推荐 M2b) |
| M0 error-free / M5 Oracle | ❌ 待写 | eval 加两条上界路径 |
| latency + CBR 归一 | ❌ 待写 | 后处理脚本加列 |
| 全量可信 LUT | ⏳ 阻塞 | 先修 `run_v1_detector_eval` 的 EXIT_1,再跑 548 |
| 对比图 F1–F5 | ❌ 待写 | `make_comparison_figures.py` |
| 组 2 RL | ⏳ 后置 | RL 闭环 unblock 后 |

---

## 9. 执行顺序与里程碑

1. **M0**(本文档,B 阶段)✅
2. **A1**:修 EXIT_1 → 跑 548 全量 → 带 Wilson 区间的可信 LUT(数据底座)。
3. **A2**:实现 M2(DeepSC 模拟,M2b)+ M0/M5 上界 + 三信道配置 + latency/CBR 后处理。
4. **A3**:写 M4 规则版 LCB 贪心选择器。
5. **A4**:一次性跑组 1 标准四方 + 上界扫描(共用任务/信道,省算力)。
6. **A5**:出图 F1–F5 + 表 T1–T2。
7. **B(后置)**:RL unblock → 组 2 资源分配消融 → F6。

> 关键省算力点:M1/M3/M4 共享同一批 Qwen 推理结果(只是选不同服务),一次全量扫描可同时产出 LUT + 组 1 多数曲线。

---

## 10. 预期叙事(论文 Story)

> "现有 VQA 语义通信(GO-SG/MU-DeepSC)在**单一固定**传输策略下取得低 SNR 增益;但在真实低空 A2G 信道中 SNR 剧烈波动,固定策略要么浪费带宽(总传图)要么精度不足(总传 token)。我们提出**语义效用驱动的自适应服务选择**:用带 Wilson 下界的语义质量 LUT 在 cache/token/image/ROI 间逐任务择优,在传统整图发生悬崖式崩溃的低 SNR 区实现优雅降级,并由两时标 RL 在多流约束下优化资源分配——在相同后端/信道/预算下,准确率全程贴近 Oracle 上界,传输代价远低于整图传输。"

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| M2 模拟特征与 Qwen 后端耦合难做忠实 | 用 M2b(模拟整图)规避训练,保留 M2a 作升级 |
| 自适应 M4 若不显著优于固定 s2 | 检查预算/信道是否落在"自适应有用"区间;低 SNR + 紧预算才是主战场 |
| RL 仍 blocked | 组 1 用规则版 M4 即可成文;RL 作组 2 增量,不阻塞主线 |
| 单数据集/单 VLM 泛化质疑 | 留补充实验位:换一个问题类型或 VLM 做鲁棒性点 |

---

## 12. 文献依据(Literature Basis)

- **MU-DeepSC / DeepSC-VQA**, Xie & Qin, arXiv:2108.07357 — CLEVR;基线 error-free / JPEG75+Huffman+LDPC1/3+16QAM / 单模态;AWGN+Rayleigh+Rician,0–20dB。
- **Liu GO-SG**, arXiv:2411.02452 — 场景图+bbox top-N;基线 DO-SG/Original-SG/Image-Transmission/Ground-Truth;Rayleigh,−2→8dB;Acc-SNR、Acc-N_top、按类型、latency 分解;+59% acc / −65% latency。
- **DeepSC**(text), Xie 2021 — Huffman/Brotli+Turbo/RS+8PSK;BLEU+句相似度;传统 >12dB 才反超、Rayleigh 下崩。
- **DeepJSCC 系列** — cliff-effect;基线 BPG+容量码 / JPEG+5G-LDPC;CBR 归一。
- **How to Evaluate SemCom**, arXiv:2309.04891 — task 指标 + 语义指标,别只用 PSNR;CBR=k/n 归一。
