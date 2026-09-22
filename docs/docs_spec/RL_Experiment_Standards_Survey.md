# DRL 无线资源分配期刊论文实验方法调研（论文二实验矩阵依据）

> 2026-07-05，Fable 代理调研 7 篇深读 + 3 篇辅助（TWC/JSAC/IoTJ 2021-2026）。
> 完整逐篇速记见代理报告；本文保留结论部分。

## 一、最低合格线 vs 加分项

**收敛图**：合格线 = reward vs episode（提案与扁平 PPO 同图，1000-3000 ep 量程）+ 约束方法必须给约束成本曲线+约束界虚线。加分 = λ 乘子轨迹图（Khairy JSAC'21 签名图，Lagrangian 方法审稿人必查）、±std/95%CI 阴影带、多规模收敛对比、报告"约 X ep 收敛"、policy entropy 辅助线。注：两篇 JSAC 混合动作论文不画收敛图也能发——但约束方法它是核心证据。

**基线套餐**：合格线 3-6 个 = 1-2 DRL 对照 + 1-2 启发式 + random +（最好）上界。**PPO 系论文不被要求实现 SAC/TD3**（Lya-HiPPO TWC'25 只用 conventional-PPO+启发式+上界；Ji TWC'24 用 DQN/DDQN+穷举上界）。混合动作公平比三法：扁平化（拍平给单 PPO）/ 离散化（给 DQN）/ 组合式（离散头固定换连续算法）。约束方法必备：无约束版 + 固定罚系数版（Khairy 三件套）。

**泛化轴**：合格线 ≥3 个测试期扫描轴（每轴 4-8 点、全基线同图）。频率排序：节点数量 > 任务负载 > SNR/功率 > 移动速度 > 权重系数。加分 = 显式零样本 "train on X test on Y" 专节（文献中稀缺，Khairy 独有：200 设备训练→100/600/1000 测试）、双工况评估、轨迹可视化。**没人对每个泛化点重训基线——零样本评估学好的策略即通行做法**。

**统计**：领域现实 = 大多不报 seeds/不做显著性检验。合格线 = 测试期 ≥32-100 rollouts 取平均并写明。加分 = 3-5 seeds 阴影带 + 表格 mean±std。t-test 非惯例不必做。

## 二、我们的缺口清单（按成本）

| 缺口 | 成本 | 判断 |
|---|---|---|
| 收敛三联图（reward/约束成本+界线/λ 轨迹，3 seeds 阴影） | **零**（trace 已存盘） | 必做，第一优先 |
| random 基线 | 小 | 必做 |
| 固定罚系数对照（λ 冻结） | 小 | 必做（补全三件套） |
| 扁平 PPO 基线 | 小（改动作头） | 强烈建议，审稿人必问 |
| oracle 上界（完美信息穷举） | 小 | 强烈建议 |
| 泛化三轴扫描（UAV 数/到达率/SNR） | 小（env 可配+评估脚本） | 必做 |
| 零样本专节 | 小 | 建议（差异化加分） |
| 32+ rollouts + mean±std | 小 | 必做 |
| 500→1000 ep（показ平台化） | 机时 | 建议 |
| DQN 基线 | 大（数百行） | 可选 |
| SAC/TD3 | 大 | **不值得**（先例充分；我们基线已 7 个超全部样本） |

## 三、实验矩阵终稿

训练协议：3-5 seeds × ≥1000 ep；测试每点 32+ rollouts；默认 = BUBBLES 标称工况。

| 编号 | 内容 | 跑什么 |
|---|---|---|
| Fig.1 三联 | (a) reward vs ep (b) 约束成本+界线 (c) λ 轨迹；proposed/no_lagrangian/fixed-penalty/service_only 同图，±std | trace 直接画 + fixed-penalty 补训 |
| Fig.2 | 多规模收敛（UAV 数 3 档，仅 proposed） | 补训 2 点 |
| Fig.3-5 | 性能 vs UAV 数 / 到达率 / SNR（全方法 8 条线） | 零样本评估扫描 |
| Fig.6 | 约束满足对比：违约率柱状（双工况）+ 效用-违约 Pareto | 汇总 3-5 数据 |
| Fig.7 | 零样本泛化专图（未见 profile/×1.5 负载/低 SNR，vs 重训参考点） | 评估 + 少量重训 |
| Table I | 主对比：双工况 × 8 方法，mean±std，含 oracle/random | 汇总 |
| Table II | 消融（已有 v3 数据，加 mean±std） | 已有 |
| Table III | 样本效率（达 95% 最终 reward 的 ep 数 + 墙钟） | trace 计算 |

优先级：Fig.1 → Table I 补臂 → Fig.3-5 → Fig.6 → 扁平 PPO → Fig.2/7 → (可选)DQN。

## 四、关键文献锚点

- Khairy et al., JSAC'21 (arXiv:2002.00073)——约束 DRL 规范范本（λ 轨迹图/95%CI/零样本泛化节）
- Long et al., Lya-HiPPO, TWC'25 (2409.13580)——最同构（分层 PPO+Lyapunov+UAV 语义），扁平 PPO 基线出处
- Ji et al., TWC'24 (2301.08376)——含 VQA 任务的语义 MAPPO，穷举上界+100 测试集协议
- Huang et al., JSAC'24 (2401.01140)、Li et al., JSAC'26 (2606.00668)——混合动作 SAC 系，多轴扫描范式
- Wang et al. (2312.01081)——组合式混合基线教科书；Shao et al., IoTJ (2406.07213)——value 系+传统优化套餐
