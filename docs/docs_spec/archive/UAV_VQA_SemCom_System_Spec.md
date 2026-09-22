# UAV-VQA 语义通信系统 — 详细实现方案

> 版本：v2.0（rate-adaptive 信道）· 服务器 lab-s1 · 分支 `codex/lut-semantic-utility-upgrade` · 生成日期 2026-06-28

---

## 1. 系统总览

本系统是一个**目标导向（goal-oriented）的 UAV 视觉问答语义通信系统**：无人机拍摄低空图像，针对每个视觉问题（VQA task）在多个"语义服务等级"之间选择以何种形式把信息传到边缘节点，边缘用视觉语言模型（Qwen2-VL）作答；一个强化学习控制器在通信/计算/缓存资源约束下，决策"传什么、传多少、给多少资源"，以最大化任务成功率。

**核心思想**：通信目标不是 bit/像素级保真，而是 **VQA 答案正确率**。低 SNR/简单问题用轻量语义证据（detector token）即可；高 SNR/复杂问题才付出代价传图像。

**三个关键组件**：
1. **语义质量模型（LUT）**：把"在某信道/视角/任务条件下，某服务等级能达到的答案正确率"建成可查询函数，供 RL 在线决策（不必每步真跑 VLM）。
2. **通信信道模型**：把 SNR 通过链路自适应映射到可达速率→字节预算→图像质量，决定语义证据的退化程度。
3. **RL 资源分配**：two-timescale PPO + Lyapunov 队列，决策服务等级、带宽/功率/算力分配与 UAV 机动。

---

## 2. 端到端数据流

```
┌─────────────── UAV 端 ───────────────┐        ┌──────── 衰落信道 ────────┐        ┌─────────── 边缘/BS 端 ───────────┐
│ 低空图像 (VisDrone)                    │        │ Rician/Rayleigh 衰落      │        │ 反序列化 → 语义证据/图像           │
│   → YOLOv8n 检测 (类/框/置信度)         │  ───►  │ 链路自适应:               │  ───►  │   → Qwen2-VL 读(问题+证据) 作答     │
│   → 语义提取/排序 (服务等级 s0-s3)      │        │   R=B·log2(1+SNR|h|²)     │        │   → 答案(yes/no 或整数)            │
│   → 源编码 + 信道编码 + 调制            │        │   字节预算→质量; 深衰落outage │        │   → 与 ground-truth 比对 → 正确率   │
└──────────────────────────────────────┘        └──────────────────────────┘        └───────────────────────────────────┘
            ▲                                                                                        │
            └────────────── RL 控制器: 服务等级 + 带宽/功率/算力 + UAV 机动 (按 LUT 效用 + Lyapunov 队列决策) ◄──┘
```

---

## 3. 任务与场景

- **问题类型（question_type）**：`presence`（有无某类目标，yes/no）、`counting`（数某类目标，整数）。（架构预留 attribute/relation/risk）
- **任务字段**：`image_id, question_type, target_class, object_count, risk_level(normal/critical), epsilon_k(质量阈值), tau_k(deadline)`。
- **任务文件**：`outputs/tasks/v1_7_tasks.csv`。
- **场景预设**：`normal/nominal_patrol`、`disaster_hotspot`、`low_snr_soft`、`low_snr_blockage`、`utm_conflict`、`edge`（用于 RL 环境压力测试）。

---

## 4. 语义服务等级 (s0–s3)

| 等级 | 名称 | 传输内容 | Payload | 用途 |
|---|---|---|---|---|
| **s0** | cache | 不传新证据，用历史缓存答案 | 0 | 新鲜缓存命中时最省 |
| **s1** | semantic token | YOLO 检测压成结构化证据（类计数+top-12 框+聚合统计） | ~150–256B | 鲁棒、低带宽，counting 强 |
| **s2** | image | 退化后整图传给 Qwen | 数十~数百 KB | 高保真，presence/复杂问题强 |
| **s3** | ROI | detector 引导的裁剪图 | 中等 | 折中（同 image 退化机制） |

**s1 语义 token 的真实内容**（`build_detector_lightweight_evidence`，`src/vqa_semcom/detector/visdrone_yolo.py:246`）：
```
detector_target_count / detector_total_count
detector_counts_by_class        例: car:14, pedestrian:8
target_mean_confidence / target_top_confidence
target_box_area_sum / all_box_area_sum / density_hint
top_target_boxes: 按置信度排序前12个 (类别, x, y, w, h, conf)
```
该文本作为"已传输的语义证据"喂给 Qwen，并约束其"只用这些 token，不得臆测"。

---

## 5. 语义质量模型（LUT）

### 5.1 定义
把 VQA 正确率建成可查询函数（不在线跑 VLM）：
```
A_k = LUT[ question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level ]
```
取值用 **accuracy 的置信下界 LCB**（保守 QoS 决策），由 **Wilson 二项区间**计算（`src/vqa_semcom/semantic/utility.py:174 wilson_interval`）。

### 5.2 关键实现
- **离线构造**：枚举 task × service_level × snr × view × freshness × risk，对每格用真实 Qwen/detector 跑出经验正确率（成功=1/0），聚合为 `accuracy_mean / accuracy_ci_low / accuracy_lcb / uncertainty / payload_kb`。文件：`scripts/run_v1_detector_eval.py`、`src/vqa_semcom/semantic/utility.py`。
- **SNR 单调校准**：`calibrate_snr_monotonicity`（utility.py:300）强制 accuracy 随 SNR 单调不降，避免采样噪声造成的非物理抖动。
- **在线查询接口**：`SemanticUtilityModel.U_sem(...)`（utility.py:400）；缺格用 `_nearest_snr_cell` 最近邻回退。RL 侧用 `get_service_candidates(obs)`（utility.py:509）拿到每个候选服务的 `accuracy_lcb / semantic_feasible / deadline_feasible / joint_feasible`。
- **演进**：V0 规则表 → V0.5 校准 → V1 mock-VLM → 真实 Qwen 测量 → VisDrone detector 证据 → V1.6/1.7/1.8 校准 → **V1.9 SNR 校准**（snr_bin 取代旧 channel_bin）。
- **当前数据**：`outputs/lut/v1_9_semantic_utility_with_ci.csv`，648 格，总样本 160542。

### 5.3 参数化效用模型（新增探索）
`src/vqa_semcom/semantic/parametric_utility.py`：**贝叶斯逻辑回归**（IRLS 求 MAP + Laplace 协方差 + quasi-binomial 过散度 + 折外结构残差方差）拟合 LUT 各格，输出 mean/LCB/uncertainty，支持**离格 SNR 插值**与压缩。
- **诚实结论（负结果）**：在这个 648 格、低噪声的完整网格上，离散查找表点精度仍优于参数化模型（表是记忆）；参数化模型价值在校准的不确定性 + 离格泛化 + 压缩，不在点精度。LCB 已校准到 coverage≈0.95。

---

## 6. 通信信道模型（v2.0：链路自适应）

### 6.1 物理依据
真实"标准通信"传图像用**链路自适应（AMC）+ ARQ**：把工作点维持在阈值以上 → 图像近乎无损到达；**SNR 决定可达速率**，速率在带宽/时延预算下决定**能传多少字节**（压缩质量）。误码导致的可见损坏只是深衰落 outage 的例外，而非常态。

### 6.2 数学
```
可达速率:    R(SNR) = B · E[ log2(1 + SNR · |h|²) ]        (遍历谱效, bits/s/Hz × B)
字节预算:    budget_bytes = R · τ / 8                       (τ = 每图传输时隙)
图像编码:    在 budget 内搜索最高 JPEG 质量/分辨率 → 真实 cv2 编解码 → 喂 Qwen
深衰落中断:  P_outage = P( log2(1+SNR·|h|²) < r_min )       (Shannon 信息中断, 理想容量逼近码)
             r_min = (min_payload·8 / τ) / B
```
- **衰落**：`sample_power_gain(kind, k_factor_db)`，Rayleigh(K=0) / Rician(K=6dB，A2G LoS) / AWGN；E[|h|²]=1 归一。
- **token 路 (s1)**：payload 极小（~256B）→ r_min≈0.0068 b/s/Hz → outage≈0（-5dB 仅 0.16%）→ **token 全 SNR 鲁棒**（精度由 detector 决定，与 SNR 无关）。这正是语义通信的核心优势。

### 6.3 实现
- 文件：`src/vqa_semcom/degradation/digital_link.py`
  - `ergodic_spectral_efficiency / rate_budget_bytes / outage_probability`
  - `_fit_jpeg_to_budget`（分辨率+质量二分搜索命中预算）
  - `transmit_image_rate_adaptive`（图像路）
  - `fer_for / transmit_image_to_path`（按 `channel_mode` 路由）
- 路由：`vlm.channel_model=ldpc_fading` 触发数字链路；`fading_link.channel_mode=rate_adaptive` 选链路自适应（备选 `ldpc_erasure`：真实 pyldpc 蒙特卡洛标定 FER + 分块擦除）。
- 配置：`configs/v2_0_ldpc_channel.yaml`（B=1MHz, τ=0.3s, Rician K6, min_payload 1500B）。
- 接入：`degrade_image`（channel.py）、`degrade_detections_for_channel`（visdrone_yolo.py）按 channel_model 路由；旧手设档位路径保留不动。

### 6.4 参数化的字节预算梯度（τ=0.3, B=1MHz, Rician K6）
| SNR | 谱效 b/s/Hz | 预算 KB | p_outage |
|---|---|---|---|
| -5dB | 0.38 | 14.0 | 0.014 |
| 0dB | 0.94 | 34.4 | 0.003 |
| 5dB | 1.91 | 69.9 | 0.001 |
| 10dB | 3.23 | 118 | 0.0003 |
| 15dB | 4.75 | 174 | 0.0001 |
| 20dB | 6.35 | 233 | 0.0001 |
（满质量图 ~892KB；outage 全程接近 0，无"0dB 丢 74%"伪迹。）

---

## 7. 目标检测器

- **模型**：YOLOv8n（ultralytics 8.4.78），VisDrone-DET 训练 50 epoch，imgsz 640。
- **数据**：train 6471 / val 548 图，11 类（pedestrian/people/bicycle/car/van/truck/tricycle/awning-tricycle/bus/motor/others）。
- **性能**：总 mAP@50 = 0.27；**QA 主力类 car mAP@50 = 0.72 / recall 0.76**（presence/counting 主要目标足够强；总 mAP 被稀有难类拖低）。
- **权重**：`outputs/detector/visdrone_yolov8n/weights/best.pt`。
- **信道退化**：`degrade_detections_for_channel` — rate-adaptive 模式下按 outage 概率丢帧（丢弃/乱码），因 token 鲁棒 → 几乎全通过。

---

## 8. 视觉语言模型（接收端）

- **模型**：Qwen2-VL-2B-Instruct（HF 本地缓存，4.2G，RTX 3080 10G）。
- **输入**：`build_vlm_prompt(task, evidence_text)`；s1 喂结构化 token 文本，s2/s3 喂（退化后）图像。
- **作答约束**：presence→yes/no；counting→单个整数；counting 默认取 `detector_target_count` 除非 token 明确否定。
- **角色定位（对标 2025 趋势）**：用真实 VLM 作 answer reasoner（比 Liu 2411.02452 的 logic-based reasoner 更 AI-native）。

---

## 9. 强化学习资源分配

- **框架**：DI-engine（评估过 tianshou），集中式训练+执行。
- **主算法**：TCH-PPO → **Two-timescale mobility-aware semantic PPO**（语义路径选择 + UAV 机动，双时间尺度）。
- **机制**：Lyapunov-guided admission / resource projection；deadline-aware evidence guard；infeasibility-aware reject control（`semantic_path=reject`）。
- **动作**：service_level（cache/token/image/ROI/reject）+ bandwidth + power + cpu_share + gpu_share + UAV assignment/机动。
- **奖励**：semantic utility（A_k≥ε_k 的任务成功）− 时延/能耗惩罚 − 队列/风险欠账。
- **接口**：环境用 `get_service_candidates` 从 LUT 取每个候选服务的可行性与效用。
- **已知卡点**：`low_snr_blockage` 场景 deadline violation 高（~0.854），在做环境侧诊断。

---

## 10. 实验设置

- **服务器**：`lab-s1`（network-ra），conda `uav_semcom`（torch 2.2.2+cu121, numpy 1.26.4, opencv 4.10, ultralytics 8.4.78, pyldpc 5.2）。运行需 `PYTHONPATH=src:.`。
- **数据集**：VisDrone-DET（train 6471 / val 548，源自 Ultralytics GitHub assets）。
- **VLM/检测器**：Qwen2-VL-2B + YOLOv8n（best.pt）。
- **入口**：`python -m scripts.run_v1_detector_eval --config configs/v2_0_ldpc_channel.yaml --evaluator qwen --service-levels 1,2 [--limit-images N]`。

---

## 11. Pilot 结果（60 图 Qwen，v2.0, τ=0.3）

**Accuracy vs SNR：**

| 服务 / 问题 | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB | 跨度 |
|---|---|---|---|---|---|---|---|
| presence s1 (token) | 0.754 | 0.754 | 0.754 | 0.754 | 0.754 | 0.754 | **0.000** |
| presence s2 (image) | 0.763 | 0.818 | 0.831 | 0.852 | 0.860 | 0.860 | **0.097** |
| counting s1 (token) | 0.534 | 0.534 | 0.534 | 0.534 | 0.534 | 0.534 | 0.000 |
| counting s2 (image) | 0.305 | 0.339 | 0.347 | 0.347 | 0.339 | 0.331 | 0.042 |
| ALL s2 (image) | 0.610 | 0.658 | 0.669 | 0.684 | 0.686 | 0.684 | 0.076 |

**结论**：
1. 修复成功——s2 图像 SNR 跨度从旧的 0.025 → 0.097（presence）。
2. s1 token 完全 SNR-平（鲁棒），精度由 detector 决定。
3. **交叉叙事**：低 SNR token 赢（鲁棒）；高 SNR image 赢（presence 0.86>0.75）；counting 全程 token 赢（detector 直接计数）。→ 资源分配应学的策略。

---

## 12. 代码与文件映射

| 模块 | 文件 / 函数 |
|---|---|
| 语义 LUT | `src/vqa_semcom/semantic/utility.py`（U_sem, wilson_interval, calibrate_snr_monotonicity, get_service_candidates） |
| 参数化效用 | `src/vqa_semcom/semantic/parametric_utility.py` + `scripts/fit_parametric_semantic_utility.py` |
| 信道(链路自适应) | `src/vqa_semcom/degradation/digital_link.py`；配置 `configs/v2_0_ldpc_channel.yaml` |
| 退化路由 | `src/vqa_semcom/degradation/channel.py`（degrade_image）、`detector/visdrone_yolo.py`（degrade_detections_for_channel） |
| 检测器 | `scripts/train_visdrone_detector.py`、`prepare_visdrone_yolo.py`；权重 best.pt |
| VLM 评测 | `scripts/run_v1_detector_eval.py`、`run_v1_vlm_eval.py`、`vlm/` |
| RL | `src/vqa_semcom/rl/`（v19_ppo.py 等） |
| 测试 | `tests/`（149 通过：含 test_semantic_utility / test_parametric_semantic_utility / test_digital_link） |

---

## 13. 与研究方向对标（定位）

- **对齐主流**：task-oriented（优化答案正确率）、传 detector 语义证据而非像素、VLM 作接收端、SNR/信道自适应、cache/token/image 多级自适应——契合 Liu 2411.02452（wireless VQA）、UAV cognitive semcom（2502.03761）、2025 VLM-receiver 趋势。
- **诚实定位**：s2 图像路当前是"速率自适应 JPEG"（经典分离信源信道编码），**不是**领域定义的"学习式语义图像编码（DeepJSCC/VQ）"。建议把 s2 定位成"传统高保真服务/资源分配的另一极"，把创新押在 **RL 资源分配 + 语义效用 LUT + token 语义服务** 上（契合 RL 主线）；若要图像路也"语义化"，需另接 DeepJSCC/VQ 学习式 codec。
