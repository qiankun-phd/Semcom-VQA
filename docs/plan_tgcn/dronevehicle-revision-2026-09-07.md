# DroneVehicle 新协议验证（2026-09-07）

## 预先冻结的范围

用户授权后续跨数据集验证，运行主机 lab-s2。只读缓存 outputs/vlm/dv_rician_predictions.csv，Rician 六个 SNR，Qwen2-VL-2B 接收器；不重新调用检测器/VLM，不下载数据，不覆盖旧结果。采用 daily-coding 实施测试与预检，results-analysis 要求明确统计单位和比较边界；本阶段负责产生完整验证数据，不宣称显著性或直接修改论文。

区分两个问题，不能混为一类泛化：

1. **目标数据集独立重拟合**：按 crc32 图像划分 275/93/122 张训练/验证/测试图像，9828/3414/4260 个问题-SNR 决策。使用与主实验相同的 18 维安全特征、双 MLP (32,16)、固定超参、300 epoch 上限、按图像验证 BCE 早停、种子 0–9。仅目标训练集计数校准和无校准各做 10 次双模型拟合；线性、类型规则、训练 LUT、固定两分支均完整保留。Oracle 只作准确率上界。这是目标重拟合，不称 zero-shot。
2. **源模型冻结迁移**：直接应用 formal_v1 Rician/train_only 的 VisDrone 模型、源训练校准参数和源验证选定价格（线性价格来自 supplement_v1）。不在 DroneVehicle 训练、校准或选择价格。10 个 MLP 种子及线性全部保存结果；单列源校准口径下固定基线。源验证的相对 1 pp 准确率条件不保证目标数据上的准确率损失。

## 数据与成本门限

严格核对配对、重复、跨分支 question type/target class/ground truth/object count；禁止答案 polarity、风险、视角、GT 和接收器输出作为特征。原始检测计数必须在同图同类别的 SNR 与问题间不变。原配置 configs/dv_rician_{main,cmp,extra}.json 需要相同的信道与原图路径；原图为 DroneVehicle adapter/images（含白边，不裁剪）。按原种子重建 CPU codec，每个图像-SNR 在全部记录路径（不只任选一个）验证保存 JPEG 哈希。若不匹配，停止新成本配旧结果，不静默继续。

成功传输使用压缩线上 bytes；outage 原记录缺少发送 bytes，显式计完整 0.3 s 时隙。所有 learned 查询计 detector 0.4275 J，VLM 使用主实验同接收器的既有约 32.31 J 增量能耗代理，**不是 DroneVehicle 新测量**。检测 packet 成本沿用既有紧凑模型，主实验的未测系统开销排除项不变。

目标重拟合的线性及 MLP 均使用固定 26 价格点与同一验证选择规则：准确率至少为各自不加价验证结果减 0.01 的候选中选最低验证能耗，平局取较小价格。不是共同绝对准确率约束，不保证测试损失，不以目标测试挑选价格。完整保存所有 sweep、逐样本标签/概率/选择、模型、曲线、manifest、SHA 和 split。

## 验证与统计约定

正式启动前：6 项针对性测试、继承主实验/补充回归测试、真实完整划分双分支 2 epoch 冒烟，以及首图六 SNR JPEG 一致性。正式进行全部 2940 个图像-SNR 的全部路径验证。CPU 单线程 BLAS、OpenCV 单线程、nice 10、CUDA 禁用。正式目录 dronevehicle_v1 新建且拒绝已存在；smoke 单列，不报告为正式性能。

正式完成核验 20 次双模型拟合、2 条线性 sweep、10 个冻结源 MLP 和 1 个线性迁移。报告 pooled 六 SNR 准确率/能耗/图像比例的 10 种子均值与样本标准差，并保留每 SNR/问题类型结果。种子不是独立数据集，图像内多问题/SNR 不当独立样本；显著性若后续开展必须用图像聚类比较，不能只依据种子标准差宣称优势。

## 预检发现并修正的字段来源问题

首轮 smoke 主动失败于原始检测计数不变门限：例如 00676/car 的 presence/counting 原始计数是 41，而其他问题的 `raw_detector_count` 在 -5 dB 变为 40，与传输后计数相同。不能只按字段名称认定 sender-observable。

源检测缓存 main 有 5807 个框；extra 的逐框签名（排除运行时间）与 main 完全一致，cmp 是 main 的精确子集。由这些原始框重新计数，与全部 presence/counting 原始计数一致；但 comparison 2 条、co_presence 8 条、threshold 8 条日志 raw 值需纠正，共 18/17502 去重决策。仅更正安全输入的 count 两维，不改原日志，不改计数标签/分支回答。逐条更正记录和原缓存 SHA 留在 data_audit.json；无目标检测的零计数由完整 main 任务计数一致性支持。

修正后 dronevehicle_smoke_v2 COMPLETE，真实划分两个校准条件各 2 epoch，首图六 SNR、三个任务文件共 18 个接收 JPEG 路径全部逐字节匹配。最初失败的 smoke_v1 保留作审计，不删除。正式前追加测试覆盖传输后字段修复且原日志值保持不变。

精确错误来源是远端 scripts/run_v1_detector_eval.py 的 comparison/co_presence/threshold 分支：缓存 tuple 将从 transmitted_records 计算的 count_a/ca 放在随后被解包为 raw_detector_count 的位置；presence/counting 分支则正确填入 raw_target_count。comparison 的 transmitted 字段还放了第二类别计数，但本次只有 counting 标签使用校准，因此不据此改动其他问题标签。

## 正式启动（仅目标重拟合）

20 项回归测试通过。后台 tmux `tgcn-dronevehicle-20260907`，PID 4163188；服务器开始时间 2026-09-08 00:20:42（服务器本地时间），日志 outputs/revision_20260907_independent/dronevehicle_v1.log。正式参数 `--target-only`，输出新目录 dronevehicle_v1；全部 2940 图像-SNR 全路径 codec 验证后进行 20 次双 MLP 和 2 条线性扫描。

**源冻结迁移已暂停**：manifest/summary/final status 标为 `PENDING_SOURCE_FEATURE_AUDIT`。未经主数据特征复核不能把历史源模型称为完全无泄漏迁移。目标重拟合 COMPLETE 不表示跨数据集整项包括迁移均已完成。原始检测框、VLM缓存及旧formal/supplement均只读保留。

## 目标重拟合完成与核验

正式 target_refit 队列 COMPLETE，总用时 109.356 s。20 次双 MLP、2 条线性扫描全部完成；2940 图像-SNR、7320 个记录的接收 JPEG 路径全部逐字节匹配。重新从保存的概率与标签计算全部 22 条、各 26 点的验证/测试 sweep，与保存指标及逐样本 picks 完全一致；所有 validation-selected index 与预设规则一致，20 份双模型与曲线齐全，数组有限且无缺失。

81 个正式输出同步本地 `paper/outputs/revision_20260907_independent/dronevehicle_v1`，逐文件 SHA-256 与远端完全一致。脚本启动 SHA-256 为 d7897918372cb3338ffd88d660df1f15a8af7c510a33f55425705577a1e19630；成功 smoke_v2 也已单独同步。原失败 smoke_v1 保留远端供审计。

### 描述性结果

测试单位为 122 张图像、4260 个问题-SNR 决策。以下汇总六 SNR，MLP 为全部 10 个训练种子的均值 ± 样本标准差；标准差不是置信区间，不代表跨数据集不确定性。节能相对同协议全图像 32.4419259234 J/query，计系统合计代理成本而非仅UAV电池。

| 方法/工作点 | 准确率 | J/query | 图像分支比例 |
| --- | --- | --- | --- |
| 全图像 | 64.0845% | 32.4419 | 100% |
| 全检测 | 72.4648% | 0.4344 | 0% |
| 类型规则 | 72.9343% | 11.2087 | 33.6620% |
| 训练 LUT | 72.4648% | 0.4344 | 0% |
| 线性默认 | 72.6995% | 2.8100 | 7.3239% |
| 线性验证选价 | 72.4648% | 0.4344 | 0% |
| MLP 默认 | 75.9085 ± 0.2056% | 12.5987 ± 1.5398 | 37.5070 ± 4.7476% |
| MLP 验证选价 | 74.6009 ± 0.4534% | 4.4731 ± 1.4062 | 12.4531 ± 4.3356% |

MLP 默认与选价节能分别为 61.1653% 和 86.2118%。默认 MLP 比类型规则准确率高 2.9742 个百分点，但能耗也更高；不宣称全维度优越。线性验证选价退化为全检测，不能按该结果说与 MLP 匹配了同一绝对准确率约束。MLP 选价的平均测试准确率比默认低 1.3075 个百分点，超过 1 pp；不违反预设，因为 1 pp 是验证集相对约束，不是测试保证。

目标训练得到 24 个 class/SNR 校准比率全部为 1.0（校准接受条件自然回退）。因此 train_only 和 none 的 train/validation/test 标签差异均为零，两模式相同种子的模型结果完全相同。已实际执行两组，不重复宣称为两个独立支持证据。Oracle 测试准确率仅作上界，为 80.6338%。

### 未完成与解释边界

- 源冻结迁移仍为 PENDING_SOURCE_FEATURE_AUDIT；本次不能写成 VisDrone→DroneVehicle 零样本泛化已验证。
- 未进行图像聚类推断/多重检验，不使用“显著优于”等统计措辞。
- Detector 缓存本身来自 VisDrone 训练检测器，本文没有新增目标检测器训练；目标重拟合特指小型路由器及校准。
- GPU/VLM 能耗仍是同接收器主数据代理，并非 DroneVehicle 实测能耗；没有新增GPU推理。
- 全部数值仍待与主文的新安全特征协议同步后再用于论文；本任务未修改论文或旧结果。
