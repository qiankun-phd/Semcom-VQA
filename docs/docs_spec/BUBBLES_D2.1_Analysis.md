# BUBBLES D2.1《Concept Formulation》精读报告（UAV 场景建模锚点库）

> 2026-07-05，Opus 4.8 子代理逐页精读全部 156 页产出。原文：SESAR JU BUBBLES D2.1 Ed 04.00.00
> (2022-09-27, PU, Grant 893206)，主题 = U-space 间隔管理服务 (SMS) ConOps。
> 用途：v19 仿真环境参数的权威出处 + "语义通信质量→飞行安全约束"理论桥。

## 一、最有价值的四处（结论先行）

1. **Appendix B (p99) 性能包线表**：10 类交通分类 + 巡航 5–25 m/s、爬升 3–5 m/s、尺寸 0.5–5 m ——直接替换我们自设的 UAV 运动参数。
2. **Appendix G (p123-129) 场景数值算例**：5×5 km、3 飞行层、20–120 m 高度；日均并发 8.17 架、峰值 20、日操作 784 次、平均时长 15 min（Table G-13）——任务到达率/空域大小的出处。
3. **§3.3.4 Block 4 (p60-61) 分隔最小值链式公式**：d_TC = d_NMAC + TSE + Vc·(T1+T2) + SSE + Vc·T_res + Vc·(T3+T4)，其中 **T4 = tracking 延迟 + separation 通信延迟 + pilot 执行延迟三个正态分布之和**（Table G-2 算例：通信均值 1.8s/σ1.0，总均值 7.82s/σ4.24）——**通信时延↑ → T4↑ → 分隔最小值↑ → 容量↓ → 冲突约束更易违反**的因果链全部有明文（p39 性能→分隔原则）。
4. **§4.5 CSPM→DSA 闭环 (p69, p71)**：SMS 三子服务之一是通信/监视性能监视 (CSPM)，性能退化触发动态分隔调整 (DSA)——把语义通信质量接进飞行安全约束的权威服务框架。

## 二、冲突/间隔模型映射（v19 环境改造清单）

| v19 现状 | BUBBLES 对应 | 动作 |
|---|---|---|
| Area4D 战略冲突检测 | Strategic conflict resolution，volume-based 4D deconfliction (p105, p26) | 引用即可，术语对齐 |
| conflict 约束（自设阈值） | Tactical Conflict = CPA 距离<分隔最小值 且 time-to-CPA<TC_th (p38, p61) | **改为 CPA 双判据** |
| 5 种机动模式 | §3.3.2 分隔方法：vertical/horizontal turn/translation/ground speed/mixed (p52-55) | 几乎完美对应，引用+重命名 |
| 拉格朗日 conflict 阈值 | TLS 总 1e-6 FAT/FH (p110)；**TLS_MAC = 2.5e-7 FAT/FH** (p40, p112) | **约束阈值锚定 TLS_MAC** |
| 安全距离量级 | sNMAC 50/15 ft (p104)；TC≈370m 水平/31m 垂直 (Table G-4/G-5, p124-125) | 校核环境尺度自洽 |
| 碰撞风险 | ∝N(N-1) 气体动理学模型 (p112, p116) | 容量-安全权衡引用 |

**严重度阶梯**（p103 Fig 46）：MAC → NMAC → IC → SL → TC，每级一个 barrier；Providence 小型 UAS 取 0.01 (p104)。

## 三、审稿人可能质疑的过度简化点（按风险排序）

1. **未区分 separation provision (RWC) 与 collision avoidance (CA)**——CA 必须独立于 U-space、是最后手段（p31, p33 假设#6）；avoid_conflict 单一动作需澄清对应哪一层。
2. **缺 conformance monitoring / separator 归属**——BUBBLES 有 10 种自动化模式明确责任分配（p48-51 Table 3-1）；UTM 状态机缺"偏航检测"。
3. **通信与安全两张皮**——建了 A2G 信道但没建 CSPM→DSA 闭环（通信退化→动态重算分隔）；**建成闭环正好是论文创新点**。
4. TLS 未锚定、间隔尺度未校核（上表已给修法）。

## 四、引用策略

- D2.1 可引（SESAR JU 官方 PU 交付物），但**非同行评审**——关键论断配双引：
  - **Weinert et al., AIAA JAIS 2022, DOI 10.2514/1.D0260**（sNMAC/碰撞风险 peer-reviewed 原始出处）
  - **CIR (EU) 2021/664/665/666** 法规原文（场景合法性）
  - CORUS ConOps、JARUS SORA、ICAO Doc 9854/9426
  - 待查：UPV 作者（Balbastre Tejedor / Vera Vélez 等）的 BUBBLES 期刊版
- **RL 方法背书**：BUBBLES 自己用 RL+GAN 学分隔最小值、有 Multi-UAS OpenAI Gym 环境（p140-141, p144）——引用为"UTM 领域已采用 RL"先例。

## 五、通信要求的边界（诚实标注）

BUBBLES **不给** RCP/时延/可靠性硬数字（fit-for-purpose 原则，p62 引 ED-261 GEN-SUR）；
τ_k/ε_k 的机制性解释 = "保证 T4 时间窗以给定置信度成立"（p61-62，2σ/1σ/0σ 置信档），
但数值要引其他标准（EUROCAE ED-282/ED-129B、3GPP C2 KPI）或自设并声明。
