# v19 算法设计双审查合并结论（Opus + Fable 独立审查，2026-07-05）

> 两代理独立只读审查 `/tmp/uav_main`（main @ ad31e9c）+ BUBBLES worktree + 160 旧训练产物。
> 完整报告见两代理输出；本文是交叉验证后的合并行动清单。**结论零矛盾，关键发现互证。**

## 一、判定总表（双方一致 = 高置信）

| 疑点 | 判定 | 证据强度 |
|---|---|---|
| sigmoid squash 缺 Jacobian | **伪**（策略定义在 pre-squash 高斯，新旧口径一致，ratio 中严格相消）| 双方一致，行级核对 |
| 慢头 credit assignment | **真 P0**：log_prob 每步计入（×K=3）+ Fable 加码：mobility mask 采样端 masked/更新端 unmasked → ratio≠1 虚假触发 clip + 熵也全步计入 | 双方确认 |
| λ 棘轮 | **真 P1**：limit=0 且 cost≥0 → λ 单调不减；旧 trace 实测 λ_quality 0→5.41 单调、违约率 0.9 全程不降。**λ 是障碍项不是影子价格** | Opus 有实测轨迹铁证 |
| 冲突多重罚 | **真 P1**：BUBBLES 冲突同时置 airspace+utm 两 flag → shaped −3.5，叠加 λ + Lyapunov **水位罚**（队列只增不减、按水位每步收罚而非增量）+ 隐性第 4 通道（success 翻负） | Fable 量化更细 |
| BUBBLES 罚压垮正项 | **真 P1**：后期冲突步净罚 −30~−50 vs 正项上限 +23 → cache 塌缩是理性解 | 双方量算一致 |
| BC 贴示范锁死 | **真 P1**：Fable 发现 `bc_aux_weight=0.28` **永不衰减**（只有 0.65 先验衰减）；Opus 实证 low_snr 下 proposed≈教师（0.955 vs 0.950） | 互补证据 |
| 欠训练+高方差 | **真 P2**：120ep 回报仍在涨、末期 std>|mean|、容量非瓶颈；Fable 加码：rollout 采集缺 no_grad | 双方确认 |

## 二、独家发现（单方，已复核成立）

**Fable 独家：**
- **N1 [P0] 两时标 mobility 代价双计**（v19_ppo.py:919-921 vs 1783-1795）：`semantic_utility` 奖励内已含 mobility_cost，两时标路径又加一遍 `_mobility_reward_adjustment` ——**现有"two-timescale vs monolithic"消融差异部分来自奖励定义不同而非结构贡献，该消融表不可信，须修后重跑**。
- **N2 [P1] per-episode advantage 归一化中和 λ 罚**：episode 内近常数的 λ 罚被减均值消掉 → 对偶机制一阶效应被架空（同时解释了"λ 顶格但没立刻塌缩"）。
- **N4末 [P1-env] CPA 运动学用 nearest 而非 assigned UAV**（multi_uav_env.py:1918）→ **冲突对策略基本外生**（唯一规避手段是选 cache）——BUBBLES 可控性的环境侧根因；若论文声称"机动规避冲突"必须修。

**Opus 独家：**
- 组件贡献实证：语义投影是最大贡献（去掉 −0.21 semSucc 且向 cache 塌缩），Lyapunov/双时标各 −0.09——架构方向正确的证据。
- 正式消融集缺 no_warm_start/no_LCB 对照。

## 三、合并行动清单

**P0 正确性必修（不修则两时标一切结论不可信）**
1. 慢头 log_prob：决策步计一次（非决策步置 0）+ 存 mobility mask 供更新端重放 + 熵乘 decision mask（Fable 给了 diff 草案）
2. N1 mobility 双计：删两时标路径的 `_mobility_reward_adjustment` 叠加（或让 semantic reward 跳过 mobility_cost，二选一）
3. 单元测试门：采集后立即重算 log_prob，断言与 old_log_probs 逐步相等（epoch-0 ratio≡1）

**P1 BUBBLES A/B 前必修/必调**
4. λ 棘轮：conflict_cost_limit=0.08（TLS 叙事锚定的暴露率预算）、λ_conflict 单独 λ_max=8、lambda_lr=0.1；可选 PID 衰减 λ←(1−0.01)λ+lr·slack
5. 罚通道去重：conflict_cost_weight 2.0→0.5、utm_conflict_cost_weight 1.5→0（同源只罚一次）；Lyapunov q_risk/q_utm 改增量式
6. bc_aux_weight 0.28→0.05 且 360ep 衰减到 0
7. env：CPA 运动学改用 assigned UAV（恢复冲突可控性）
8. N2：advantage 归一化改 batch 级（与攒批联动）

**P2 效率（A/B 后可做）**：采集包 no_grad（5 分钟，建议顺手）、GAE(0.95)、8-episode 攒批+minibatch、log_std clamp 上限 2.0→0.0、离散/连续熵分权

**P3 论文表述**：修 P1 前不能自称 Lagrangian primal-dual——弱化为 "adaptive penalty coefficients updated by projected sub-gradient on constraint slack"；修后可保留原表述。

## 四、修订版 BUBBLES A/B 设计（Fable 方案，采纳）

- **Arms**：A1 legacy 档 / A2 bubbles 档（同控制器同超参）/ B1 bubbles+无慢头 / B2 bubbles+关 λ_conflict 通道 / C 非学习基线（semantic_greedy、cache-only）
- **规模**：500 ep × 3 seeds；评估 2400 任务/arm（冲突率 95%CI 半宽 ≈0.02）
- **冒烟门**（三条全过才上 500ep）：λ_conflict 不贴 λ_max、non_cache_ratio∈[0.6,0.98]、epoch-0 ratio≡1 单测通过
- **成功判据**：A2 冲突率 ≤0.10 且 semSucc 相对 A1 降幅 ≤0.03、cache ratio ≤0.30；A2 冲突率显著低于 B1（Δ≥0.05）
- **失败判据与对策**：cache ratio>0.5 → 罚仍过重（limit 提 0.12 或 λ_max 降 5）；A2 冲突率 ≈ C → 冲突纯外生（必须先落地 assigned-UAV CPA 修复）
