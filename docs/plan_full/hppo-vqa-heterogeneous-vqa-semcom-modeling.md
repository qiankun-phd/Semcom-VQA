# HPPO-VQA 场景与建模细化：UAV 使能的异构计算-通信 VQA 语义网络

日期：2026-06-01

目标：把 HPPO-VQA 从“UAV 做 VQA/语义通信”细化成可直接建模、可直接仿真、可写进 TCCN/JSAC 类系统论文的场景。本文档采用“无雷达主线”：UAV 自主巡逻、视觉感知、触发 VQA 任务，并在 UAV/UE/边缘节点之间联合优化语义通信、异构计算资源和 UAV 控制。雷达可以作为可选触发源，但不再是主线依赖。

## 1. 一句话场景定义

HPPO-VQA 面向一个多 UAV 低空巡逻网络：UAV 携带相机和有限 onboard 计算资源，在区域内执行视觉巡逻；当发现目标或事件时，系统不是直接上传原始图像，而是在“巡逻、语义快照、VQA 确认、高保真证据”之间选择合适语义模式，并联合调度 UAV 轨迹、发射功率、带宽/RB、CPU/GPU/RAM/VRAM、模型缓存和 VQA 问题优先级。

推荐英文表述：

> HPPO-VQA studies a UAV-enabled visual semantic communication network, where UAVs jointly decide where to sense, which visual semantic task to execute, which semantic representation to transmit, and where to run VQA inference under coupled communication, computation, memory, and mobility constraints.

这个场景的关键不是“UAV 拍图后传输”，而是：

- UAV 是主感知节点：决定去哪看、何时看、看多细。
- VQA 是任务化语义推理：问题类型决定需要的视觉证据和语义质量。
- 语义通信不是只省 bit：它把传输压力转移到 CPU/GPU/RAM/模型缓存。
- HPPO 的混合动作空间天然出现：离散语义模式/卸载/模型选择 + 连续轨迹/功率/带宽/CPU-GPU 频率。

## 2. 系统实体与时间尺度

### 2.1 实体集合

- UAV 集合：`m in M = {1,...,M}`
- UE / ground user / ground station / edge receiver 集合：`n in N = {1,...,N}`
- 视觉任务 / 目标事件集合：`k in K(t)`
- VQA 问题类型集合：`q in Q`
  - `q=0`: target_presence，是否存在目标
  - `q=1`: target_identity，目标是什么
  - `q=2`: risk_assessment，是否有预警风险
  - `q=3`: tracking_need，是否需要持续跟踪
  - `q=4`: evidence_confirm，是否需要高保真视觉证据
- 语义模式集合：`s in S`
  - `s=0`: patrol-only，不上传视觉语义
  - `s=1`: semantic snapshot，上传类别/风险/置信度
  - `s=2`: compact feature，上传轻量视觉特征
  - `s=3`: light VQA，运行轻量 VQA 并传答案
  - `s=4`: high-fidelity VQA/evidence，运行高精度 VQA 或上传高保真证据
- 推理位置集合：`l in L = {UAV-local, UE-local, edge-server, cloud}`

### 2.2 时间尺度

HPPO-VQA 建议采用双时间尺度建模：

- 慢时间尺度 `tau`：UAV 轨迹、悬停点、相机调度、模型缓存、角色分配。
- 快时间尺度 `t`：每个 VQA/语义任务的模式选择、功率、RB、卸载、CPU/GPU 分配。

慢层更新周期可以是 `1-5 s`；快层更新周期可以是一个传输时隙或任务时隙，例如 `50-200 ms`。这样自然对应：

- UAV slow layer：控制全局覆盖、能量、缓存和长期任务队列。
- UE/query fast layer：控制每个问题的语义模式、传输资源和推理位置。

## 3. UAV 视觉巡逻任务生成

### 3.1 UAV 视觉观测

UAV `m` 在时刻 `t` 的位置：

```text
p_m(t) = [x_m(t), y_m(t), H_m(t)]
```

目标/事件 `k` 的位置：

```text
r_k(t) = [x_k(t), y_k(t), 0]
```

UAV 与目标距离：

```text
D_{m,k}(t) = ||p_m(t) - r_k(t)||_2
```

相机可见性：

```text
V_{m,k}(t) = clip(
  a_0 - a_1 D_{m,k}(t)
      - a_2 |theta_{m,k}(t)|
      - a_3 O_k(t)
      + a_4 L_k(t),
  0, 1
)
```

其中：

- `theta_{m,k}(t)`：相机视角偏差；
- `O_k(t)`：遮挡程度；
- `L_k(t)`：光照/天气质量；
- `V_{m,k}(t)`：视觉可见性分数。

如果没有真实图像，可以用仿真器给 `D, theta, O, L` 采样；如果有 UAV 图像数据集，可以用检测置信度或图像质量作为 `V` 的估计。

### 3.2 任务触发

每个潜在目标 `k` 的视觉任务触发概率：

```text
Pr(A_k(t)=1) = sigmoid(b_0 + b_1 V_{m,k}(t) + b_2 R_k(t) + b_3 I_k(t))
```

其中：

- `R_k(t)`：任务风险，如靠近敏感区、目标速度异常、类别危险度；
- `I_k(t)`：信息新颖性，如当前目标是否与缓存/知识库中已有目标重复；
- `A_k(t)`：是否生成 VQA/语义任务。

这部分是我们自己的任务生成模型，可以从 UAV 目标检测、VQA 任务驱动通信文献中借鉴“任务到达 + 语义重要性”的思想。

## 4. 三档语义模式：去雷达后的核心场景

无雷达主线下，三档策略改为 UAV 视觉驱动。

### Tier 1: Patrol-only

适用条件：

- 可见性低但风险低，或者没有明显目标；
- 已有缓存知识足够；
- 信道/计算资源紧张。

动作：

- UAV 继续巡逻；
- 不上传图像，不运行 VQA；
- 只保留轻量状态或轨迹记录。

成本：低通信、低计算、低语义精度。

### Tier 2: Semantic Snapshot

适用条件：

- UAV 检测到目标，但风险中等；
- 不需要复杂 VQA；
- 只需要类别、位置、风险标签、置信度。

动作：

- 运行轻量检测或轻量视觉 encoder；
- 上传 object token / class label / risk score / confidence；
- 可以由 UAV 本地 CPU/NPU 或边缘轻量模型完成。

### Tier 3: VQA Confirmation / High-Fidelity Evidence

适用条件：

- 高风险目标；
- 目标类别不确定；
- 需要回答具体问题；
- 需要视觉证据或人工可解释输出。

动作：

- 运行轻量或高精度 VQA；
- 选择本地推理或卸载到 edge；
- 上传 VQA answer、visual feature、或高保真图像片段。

这三档可以直接映射成离散语义模式 `s_k(t)`。

## 5. 语义任务 Workload Profile

每个任务 `k` 在语义模式 `s` 下具有 workload profile：

```text
W_k(s) = [
  d_k^in,
  z_k(s),
  c_k^cpu(s),
  c_k^gpu(s),
  m_k^ram(s),
  m_k^vram(s),
  e_k^mem(s),
  A_k(s)
]
```

含义：

- `d_k^in`：原始输入大小，例如图像帧大小；
- `z_k(s)`：输出语义符号量或传输 bit 数；
- `c_k^cpu(s)`：CPU cycles；
- `c_k^gpu(s)`：GPU FLOPs / GPU cycles；
- `m_k^ram(s)`：RAM 占用；
- `m_k^vram(s)`：VRAM 占用；
- `e_k^mem(s)`：缓存/内存访问能耗；
- `A_k(s)`：该语义模式的任务准确率或语义效用。

推荐建一个表：

| Semantic mode | Output | CPU cost | GPU cost | RAM | VRAM | Quality |
|---|---|---:|---:|---:|---:|---:|
| patrol-only | none/status | low | 0 | low | 0 | low |
| semantic snapshot | class/risk token | medium | low | medium | low | medium |
| compact feature | feature vector | medium | medium | medium | medium | medium-high |
| light VQA | answer+confidence | medium | medium | high | medium | high |
| high-fidelity VQA | answer+feature/image | high | high | high | high | highest |

这一建模直接来自“语义通信不是只减少 bit，而是把代价转移到 CPU/GPU/RAM/VRAM”的核心洞察。

## 6. 通信模型

### 6.1 空地信道与速率

UAV `m` 到 UE/edge `n` 的信道增益：

```text
h_{m,n}(t) = beta_0 d_{m,n}(t)^(-alpha) g_{m,n}(t)
```

其中 `d_{m,n}(t)` 为链路距离，`alpha` 为路径损耗指数，`g` 为小尺度衰落或 LoS/NLoS 修正。

速率采用经典 Shannon 形式：

```text
R_{m,n}(t) =
  B_{m,n}(t) log_2(
    1 + p_{m,n}(t) h_{m,n}(t) /
        (N_0 B_{m,n}(t) + I_{m,n}(t))
  )
```

借鉴来源：这是无线资源分配、MEC offloading、UAV 通信里最常见的链路速率模型，可参考 You et al. TWC 2017 MEC offloading，以及 Zeng, Zhang, Lim 的 UAV wireless communication 建模。

### 6.2 语义传输时延与能耗

如果任务 `k` 由 UAV `m` 传给节点 `n`，模式为 `s`：

```text
T_k^tx(s,t) = z_k(s) / R_{m,n}(t)
E_k^tx(s,t) = p_{m,n}(t) T_k^tx(s,t)
```

其中 `z_k(s)` 是语义模式输出大小。与传统 bit 传输不同，`z_k(s)` 由语义模式决定：VQA answer 很小，feature 中等，高保真图像最大。

## 7. 异构计算模型

### 7.1 CPU 计算模型

对于任务 `k` 在节点 `l` 上使用 CPU 执行：

```text
T_{k,l}^{cpu}(s,t) = c_k^{cpu}(s) / f_l^{cpu}(t)
E_{k,l}^{cpu}(s,t) = kappa_l^{cpu} c_k^{cpu}(s) [f_l^{cpu}(t)]^2
```

解释：动态电压频率调节中 CPU 功率常写为 `P = kappa f^3`，因此执行 `c` cycles 的能耗是 `kappa c f^2`。

借鉴来源：MEC/DVS 文献常用该形式，如 Wang et al. TCOM 2016 partial offloading with DVS，Mao et al. JSAC 2016 dynamic computation offloading，You et al. TWC 2017 MEC offloading。

### 7.2 GPU/NPU 计算模型

对于任务 `k` 在 GPU/NPU 上执行：

```text
T_{k,l}^{gpu}(s,t) = c_k^{gpu}(s) / f_l^{gpu}(t)
E_{k,l}^{gpu}(s,t) =
  kappa_l^{gpu} c_k^{gpu}(s) [f_l^{gpu}(t)]^2
  + E_l^{launch}(s)
```

其中 `E_l^{launch}(s)` 表示 GPU kernel launch、模型加载或 batch 调度开销。若不想复杂化，可以先合并进 `kappa_l^{gpu}`。

借鉴来源：CPU-GPU workload partitioning 可借鉴 Zeng et al., TWC 2021, “Energy-Efficient Resource Management for Federated Edge Learning With CPU-GPU Heterogeneous Computing”。该文明确把带宽、CPU-GPU workload partition、speed scaling 和通信计算时间联合优化。

### 7.3 CPU-GPU Workload Split

令 `x_{k,l}^{cpu}(t), x_{k,l}^{gpu}(t) in [0,1]` 表示任务在 CPU/GPU 上的切分比例：

```text
x_{k,l}^{cpu}(t) + x_{k,l}^{gpu}(t) = 1
```

并行执行时：

```text
T_{k,l}^{comp}(s,t) =
  max(
    x_{k,l}^{cpu} c_k^{cpu}(s) / f_l^{cpu},
    x_{k,l}^{gpu} c_k^{gpu}(s) / f_l^{gpu}
  )
```

串行流水线执行时：

```text
T_{k,l}^{comp}(s,t) =
  x_{k,l}^{cpu} c_k^{cpu}(s) / f_l^{cpu}
  + x_{k,l}^{gpu} c_k^{gpu}(s) / f_l^{gpu}
```

建议第一版仿真用并行 `max` 模型，后续可扩展到 pipeline queue。

### 7.4 RAM/VRAM 约束

节点 `l` 的 RAM/VRAM 资源约束：

```text
sum_k a_{k,l}(t) m_k^{ram}(s_k)
  + M_l^{cache,ram}(t)
  <= M_l^{ram,max}

sum_k a_{k,l}(t) m_k^{vram}(s_k)
  + M_l^{model,vram}(t)
  <= M_l^{vram,max}
```

RAM/VRAM 的意义：

- RAM 影响图像帧 buffer、feature queue、KB cache；
- VRAM 影响 VQA 模型是否能驻留、batch size、feature tensor；
- 缓存命中可以降低模型加载延迟和通信量。

这是本文的核心场景创新之一。现有 UAV-SemCom 多把计算抽象为 CPU cycles，而本文把 memory/cache 显式放入约束。

## 8. 模型缓存与知识库建模

### 8.1 模型缓存变量

令 `y_{l,r}(t) in {0,1}` 表示节点 `l` 是否缓存模型 `r`，例如：

- light detector；
- light VQA；
- large VQA；
- scene memory encoder；
- KB retrieval module。

模型缓存占用：

```text
sum_r y_{l,r}(t) M_r^{model} <= M_l^{vram/cache}
```

如果任务 `k` 的模式 `s` 需要模型 `r(s)`，则模型加载延迟：

```text
T_k^{load}(s,t) = (1 - y_{l,r(s)}(t)) L_{r(s)}^{load}
```

模型加载能耗：

```text
E_k^{load}(s,t) = (1 - y_{l,r(s)}(t)) E_{r(s)}^{load}
```

### 8.2 知识库/Feature Cache

令 `C_l(t)` 为节点 `l` 的知识库/feature cache 状态。缓存命中概率：

```text
P_k^{hit}(t) =
  phi(sim(e_k, C_l(t)), freshness_l(t), target_repeat_k(t))
```

如果缓存命中，可以减少传输语义量和计算量：

```text
z'_k(s,t) = z_k(s) [1 - eta_z P_k^{hit}(t)]
c'^{gpu}_k(s,t) = c_k^{gpu}(s) [1 - eta_c P_k^{hit}(t)]
```

语义质量受知识新鲜度影响：

```text
Q_k(s,t) = Q_k^0(s,t) - eta_stale Delta_k^{KB}(t)
```

这一块可以借鉴 cache-assisted MEC 的思想，但本文的不同点是缓存对象不是普通文件，而是模型权重、视觉 feature、目标记忆、语义知识。

## 9. VQA 语义质量模型

### 9.1 基础 VQA 正确率

对于任务 `k`、问题 `q`、语义模式 `s`：

```text
A_k(q,s,t) =
  A_q^{base}(s)
  + eta_v V_{m,k}(t)
  + eta_c C_{k,l}^{hit}(t)
  - eta_d D_{m,k}(t)
  - eta_ch chi_{m,n}(t)
```

其中：

- `A_q^{base}(s)`：不同语义模式/模型对问题 `q` 的基础准确率；
- `V_{m,k}(t)`：视觉可见性；
- `C_{k,l}^{hit}(t)`：缓存或知识命中；
- `chi_{m,n}(t)`：链路退化或语义失真；
- `D_{m,k}(t)`：目标距离或视觉难度。

可以 clip 到 `[0,1]`。

### 9.2 基于 DeepSC-VQA LUT 的替代建模

如果已有 DeepSC-VQA lookup table，可以写成：

```text
A_k(q,s,t) = LUT_q(SNR_{m,n}(t), K_k^sem(t), mode=s)
```

其中 `K_k^sem` 是语义符号数。HPPO-VQA 当前工程已有 VQA LUT/QRS 思路，可以直接把 LUT 作为任务质量函数。

### 9.3 语义效用

定义任务 `k` 的语义效用：

```text
U_k(t) =
  w_q A_k(q,s,t)
  + w_r Risk_k(t) A_k(q,s,t)
  - w_z z_k(s)
  - w_d T_k(t)
  - w_e E_k(t)
```

或者把风险和准确率相乘：

```text
U_k(t) = w_q Risk_k(t) A_k(q,s,t) - Cost_k(t)
```

这体现：高风险任务即使成本高，也值得用高精度 VQA；低风险任务只需要 snapshot。

## 10. UAV 能耗与轨迹模型

### 10.1 简化移动能耗

第一版仿真可以用：

```text
E_m^{fly}(t) = P_m^{hover} Delta t + P_m^{move}(v_m(t)) Delta t
```

如果只需要可控实验，也可以用线性近似：

```text
E_m^{fly}(t) =
  e_m^{dist} ||p_m(t+1)-p_m(t)||_2
  + e_m^{hover} Delta t
```

### 10.2 Rotary-Wing UAV 经典推进功率

更完整的 rotary-wing UAV 模型可采用：

```text
P(v) =
  P_0(1 + 3v^2/U_tip^2)
  + P_i(
      sqrt(1 + v^4/(4v_0^4)) - v^2/(2v_0^2)
    )^(1/2)
  + 0.5 d_0 rho s A v^3
```

借鉴来源：Zeng, Xu, Zhang 等关于 rotary-wing UAV 通信能耗最小化的工作中常用该 closed-form propulsion power model。第一版不必强行使用完整式，除非实验要主打飞行能耗真实性。

## 11. 总时延与总能耗

任务 `k` 的端到端时延：

```text
T_k(t) =
  T_k^{capture}(t)
  + T_k^{queue}(t)
  + T_k^{load}(t)
  + T_k^{comp}(t)
  + T_k^{tx}(t)
  + T_k^{decode}(t)
```

任务 `k` 的总能耗：

```text
E_k(t) =
  E_k^{fly-share}(t)
  + E_k^{tx}(t)
  + E_k^{cpu}(t)
  + E_k^{gpu}(t)
  + E_k^{mem}(t)
  + E_k^{load}(t)
```

UAV `m` 的电池约束：

```text
sum_t [
  E_m^{fly}(t)
  + E_m^{tx}(t)
  + E_m^{cpu}(t)
  + E_m^{gpu}(t)
  + E_m^{mem}(t)
] <= E_m^{bat,max}
```

这正是本文区别于普通语义通信的地方：飞行、传输、推理、缓存都在同一资源账本里。

## 12. 优化问题

### 12.1 决策变量

慢层变量：

- UAV 下一位置 `p_m(t+1)`；
- UAV 悬停/巡逻动作；
- UAV 角色 `rho_m(t) in {collector, encoder, relay, VQA-inference, cache-node}`；
- 模型缓存 `y_{m,r}(t)`；
- 长期语义预算 `K_m^{budget}(t)`。

快层变量：

- 任务语义模式 `s_k(t)`；
- 推理位置 `l_k(t)`；
- CPU/GPU split `x_{k,l}^{cpu}, x_{k,l}^{gpu}`；
- 发射功率 `p_{m,n}(t)`；
- 带宽/RB 分配 `B_{m,n}(t), b_k(t)`；
- 语义符号数 `K_k^{sem}(t)`；
- VQA 模型选择 `r_k(t)`。

### 12.2 目标函数

推荐主问题写成约束优化：

```text
maximize_pi  sum_t sum_k U_k(t)
```

subject to:

```text
A_k(q,s,t) >= A_q^{min}, for high-priority tasks
T_k(t) <= T_q^{max}
sum_n B_{m,n}(t) <= B_m^{max}
sum_k a_{k,l}(t) c_k^{cpu}(s_k) <= C_l^{cpu,max}
sum_k a_{k,l}(t) c_k^{gpu}(s_k) <= C_l^{gpu,max}
sum_k a_{k,l}(t) m_k^{ram}(s_k) <= M_l^{ram,max}
sum_k a_{k,l}(t) m_k^{vram}(s_k) <= M_l^{vram,max}
sum_t E_m(t) <= E_m^{bat,max}
p_m(t) in flight region
s_k(t), l_k(t), r_k(t), y_{l,r}(t) discrete
p_{m,n}(t), B_{m,n}(t), f_l^{cpu}(t), f_l^{gpu}(t) continuous
```

这是混合整数、非凸、动态、多智能体问题，适合 HPPO/层次化 MARL。

### 12.3 拉格朗日/Typed Semantic Price 形式

把语义质量约束写成 floor violation：

```text
g_k(t) = max(0, A_q^{min} - A_k(q,s,t))
```

定义 UAV-问题类型价格：

```text
lambda_{m,q}(t+1) =
  clip(
    lambda_{m,q}(t)
    + eta_lambda mean_{k served by m,q} g_k(t),
    0, lambda_max
  )
```

快层 utility：

```text
score_k(s,l,p,B) =
  w_q A_k(q,s,t)
  - w_T T_k(t)
  - w_E E_k(t)
  - w_Z z_k(s)
  - lambda_{m,q}(t) g_k(t)
```

这个 typed semantic price 是 HPPO-VQA 已有 QRS/typed-price 架构的核心，可以继续保留。区别是现在 `lambda` 不只反映通信语义 floor，还同时受到计算/缓存导致的 VQA 质量下降影响。

## 13. HPPO 架构落点

### 13.1 慢层 UAV Policy

慢层 actor 输出：

```text
a_m^U(t) = [
  Delta x_m,
  Delta y_m,
  Delta H_m,
  rho_m,
  cache_update_m,
  compute_budget_m
]
```

其中：

- 连续：位置变化、计算预算比例；
- 离散：角色、缓存更新动作。

适合：constrained MAPPO / HAPPO / HPPO。

### 13.2 快层 UE/Query Policy

快层 actor 输出：

```text
a_k^Q(t) = [
  s_k,
  l_k,
  r_k,
  K_k^sem,
  p_{m,n},
  B_{m,n},
  x_cpu,
  x_gpu
]
```

其中：

- 离散：语义模式、推理位置、模型选择；
- 连续：功率、带宽、CPU/GPU 分配、语义符号数。

适合：QRS baseline、contextual bandit、PDQN-style parameterized action、neural selector。

### 13.3 为什么不是一个平面 RL

如果用一个 agent 同时决定轨迹、缓存、VQA 模型、RB、功率、CPU/GPU，动作空间太大，收敛慢且解释性弱。

分层后：

- UAV 层负责慢变化、全局耦合资源；
- UE/query 层负责快变化、任务局部选择；
- typed semantic price 负责跨层沟通；
- risk head 负责 VQA floor 安全。

这就是 HPPO-VQA 的算法架构创新。

## 14. 可直接实现的仿真状态/动作/奖励

### 14.1 状态 State

每个 UAV `m`：

```text
[位置, 速度, 电量, CPU_available, GPU_available,
 RAM_available, VRAM_available, cache_vector,
 associated_UE_count, channel_load]
```

每个任务 `k`：

```text
[位置, 目标风险, 问题类型, deadline, visibility,
 semantic_priority, required_accuracy, raw_size, cache_hit_prob]
```

每条链路 `m-n`：

```text
[SNR, bandwidth_available, interference, pathloss, queue_delay]
```

全局：

```text
[任务队列长度, 平均语义 floor violation,
 price matrix lambda, edge CPU/GPU/RAM load]
```

### 14.2 动作 Action

UAV slow action：

- move direction / distance；
- hover or patrol；
- role selection；
- cache model selection；
- compute budget reservation。

Fast query action：

- semantic mode `s`；
- inference location `l`；
- VQA model `r`；
- semantic symbols `K`；
- transmit power `p`；
- RB/bandwidth `b`；
- CPU/GPU split `x`。

### 14.3 奖励 Reward

```text
r_t =
  sum_k [w_A A_k - w_T T_k - w_E E_k - w_Z z_k - w_G g_k]
  - w_battery sum_m battery_violation_m
  - w_mem sum_l memory_violation_l
  - w_delay sum_k deadline_violation_k
```

其中 `g_k` 是 VQA semantic floor violation。

高风险任务可以加权：

```text
w_A(k) = w_0 + w_r Risk_k + w_q QuestionImportance_q
```

## 15. Baseline 与消融实验

建议至少做以下 baseline：

1. Bit-level transmission：传图/feature，不做语义模式选择。
2. SemCom without heterogeneous compute：只优化语义符号和无线资源，计算抽象为常数。
3. CPU-only：没有 GPU/NPU 加速。
4. GPU-aware no memory：考虑 GPU，但不考虑 RAM/VRAM/cache。
5. No cache：不缓存模型/feature/KB。
6. No VQA priority：所有问题同权。
7. No UAV/UE role split：平面策略或统一贪心。
8. QRS deterministic：当前工程已有的安全下界。
9. HPPO-VQA full：完整异构计算-通信-轨迹联合优化。

关键指标：

- VQA accuracy / semantic utility；
- high-priority VQA floor satisfaction；
- average delay / 95th percentile delay；
- total energy：fly + tx + CPU + GPU + mem；
- communication cost / semantic symbol cost；
- CPU/GPU utilization；
- RAM/VRAM peak usage；
- cache hit rate；
- task drop rate；
- UAV battery lifetime；
- fairness across UE/tasks。

## 16. 可借鉴文献与公式来源

### 16.1 MEC Computation / Offloading

可借鉴：

- CPU frequency / DVS：`T = c/f`, `E = kappa c f^2`。
- Offloading transmission：`R = B log2(1+SINR)`, `T_tx = bits/R`, `E_tx = p T_tx`。

代表文献：

- Wang et al., “Mobile-Edge Computing: Partial Computation Offloading Using Dynamic Voltage Scaling,” IEEE TCOM 2016.
- Mao et al., “Dynamic Computation Offloading for Mobile-Edge Computing With Energy Harvesting Devices,” IEEE JSAC 2016.
- You et al., “Energy-Efficient Resource Allocation for Mobile-Edge Computation Offloading,” IEEE TWC 2017, DOI: `10.1109/TWC.2016.2633522`.

### 16.2 CPU-GPU Heterogeneous Computing

可借鉴：

- CPU/GPU workload partition；
- CPU/GPU speed scaling；
- bandwidth + computation resource joint optimization。

代表文献：

- Zeng et al., “Energy-Efficient Resource Management for Federated Edge Learning With CPU-GPU Heterogeneous Computing,” IEEE TWC 2021, DOI: `10.1109/TWC.2021.3088910`.

本文扩展：从 federated learning workload 扩展到 UAV visual semantic/VQA workload，并加入 RAM/VRAM/model cache。

### 16.3 UAV Wireless / Trajectory / Energy

可借鉴：

- air-ground channel；
- UAV trajectory + power + communication resource；
- rotary-wing propulsion power model。

代表文献：

- Zeng, Zhang, Lim, “Wireless Communications With Unmanned Aerial Vehicles: Opportunities and Challenges,” IEEE Communications Magazine 2016.
- Zeng, Xu, Zhang, “Energy Minimization for Wireless Communication With Rotary-Wing UAV,” IEEE TWC 2019.

### 16.4 Semantic Communication

可借鉴：

- task-oriented semantic utility；
- semantic similarity / task accuracy as QoS；
- semantic compression ratio / semantic symbols；
- semantic resource allocation by task importance。

代表文献：

- Xie/Qin 等 Task-oriented semantic communications, IEEE JSAC 2022.
- Wang et al., “Performance Optimization for Semantic Communications: An Attention-Based RL Approach,” IEEE JSAC 2022.
- Hwang et al., “DRL-Driven Dynamic Resource Allocation for Task-Oriented Semantic Communication,” IEEE TCOM 2023.
- Yang et al., “Energy Efficient Semantic Communication Over Wireless Networks With Rate Splitting,” IEEE JSAC 2023.
- FAST: Fidelity-Adjustable Semantic Transmission over Heterogeneous Wireless Networks.

本文扩展：把 task accuracy/VQA accuracy 与 CPU/GPU/RAM/VRAM 和 UAV 轨迹联动。

## 17. 推荐论文主线

建议把 HPPO-VQA 的场景主线改为：

> CPU-GPU-RAM-Aware Hierarchical Hybrid Policy Optimization for UAV-Enabled VQA Semantic Communication Networks

中文：

> 面向 UAV 使能 VQA 语义通信网络的 CPU/GPU/RAM 感知层次化混合策略优化。

核心贡献可以写成：

1. 提出 UAV-enabled VQA semantic communication network，UAV 同时承担视觉采集、语义推理、空中通信和边缘协同角色。
2. 建立 semantic workload profile，将 VQA/视觉语义模式映射到 bit/symbol、CPU、GPU、RAM、VRAM、缓存和语义质量。
3. 构建通信-计算-缓存-轨迹联合优化问题，显式考虑飞行能耗、通信能耗、CPU/GPU 推理能耗和 RAM/VRAM 约束。
4. 设计 HPPO-VQA 分层混合策略：慢层 UAV mobility/compute/cache control，快层 UE/query semantic mode/offloading/resource selection。
5. 通过 no heterogeneous compute、CPU-only、no cache、no VQA priority、no role split 等消融证明每个模块必要性。

## 18. 与雷达/LSS-HSR-L 的关系

如果继续使用 LSS-HSR-L：

- 不能把它作为主 VQA 图像数据集；
- 可以作为低空目标类别和风险先验；
- 可以作为 optional sensing-trigger module；
- 主线仍应是 UAV visual semantic communication。

推荐写法：

> LSS-HSR-L can be used as an auxiliary low-altitude event prior, but the main HPPO-VQA formulation does not rely on paired radar-image data. The core problem is UAV-enabled visual semantic communication with heterogeneous computation and memory-aware resource orchestration.

这样可以避免“雷达已经识别了为什么还要 UAV”的逻辑问题。

## 19. 下一步落地建议

第一阶段：仿真建模

- 实现 semantic workload profile 表；
- 给每种 VQA mode 设置 `z, CPU, GPU, RAM, VRAM, quality`；
- 加入 CPU/GPU/RAM/VRAM 状态；
- 加入缓存命中和模型加载延迟；
- 扩展 reward 和约束。

第二阶段：算法

- 先做 QRS baseline；
- 再做 fast contextual selector；
- 最后做 HPPO full hierarchical mixed policy。

第三阶段：实验

- 主实验：HPPO-VQA vs baselines；
- 消融：CPU-only, GPU-aware, no cache, no VQA priority, no role split；
- 压力测试：UE 数量、任务到达率、GPU/RAM 瓶颈、信道退化、电池限制。

## 20. 最重要的边界声明

本文场景不声称：

- 真实雷达-图像配对确认；
- 某个 VQA 模型本身的新 SOTA；
- 只靠通信资源优化解决 VQA。

本文应声称：

- VQA 语义通信是计算、通信、缓存和 UAV 控制共同决定的系统问题；
- HPPO-VQA 的创新在于异构资源感知的分层混合策略架构；
- CPU/GPU/RAM/VRAM-aware semantic workload modeling 是区别于现有 UAV-SemCom 的关键场景创新。

