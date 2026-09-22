# 基于现有结果的论文改写交接

日期：2026-09-07。用户要求以已经完成且有效的结果组织论文，不再补跑实验。

## 本轮范围

没有启动训练、VLM 推理、信道重放或新实验。仅读取已有 JSON/NPZ 结果，重新汇总、绘图、修改 LaTeX 并编译。未执行 senderfix、transfer 草稿。旧实验、旧图和日志均保留。

改写前备份：`/tmp/hppo-writing-before-xvcr73/`。这是临时备份，不应视为长期版本管理。

## 叙述主线

问题需求决定发送的证据，并同时决定接收端采用符号解码还是 VLM 推理。类型规则展示这一机制；逐样本路由提供不同的准确率—系统能耗工作点，其收益取决于数据集与接收器。文章不再以“非线性路由始终优于简单基线”为主张。

实验组织为：设置与核算范围 → 类型路由和证据互补性 → 逐样本路由及能耗权衡 → 接收器适配 → 局限。补充材料保留完整有效 AWGN 消融、未校准对照和接收器任务覆盖情况。

## 采用与排除的证据

- `formal_v1`：三信道固定分支、类型规则和 LUT；仅 AWGN 学习型路由。固定策略不使用有争议的发送端数量特征。
- `supplement_v1`：有效 AWGN 消融，包括完整模型不占优的结果。
- `dronevehicle_v1/target_refit`：已修正原始检测数量的目标数据拟合。不是跨数据集零样本迁移。
- `crossreceiver_v2`：公共样本与修正发送端特征下的接收器专用拟合和冻结源路由器比较。
- 不采用：依赖有问题数量特征的旧 Rayleigh/Rician 学习路由、含答案衍生输入的历史高分、被中断的 crossreceiver_v1，以及尚未执行的实验草稿。

保留弱结果并非删去主要贡献：VisDrone AWGN 的 MLP 不优于线性基线；DroneVehicle 固定检测能耗更低但准确率较低；验证选价的测试准确率下降可能超过验证容许值。这些边界已写入正文。

## 核心数字与核算边界

- VisDrone 类型规则：三信道准确率 67.47%、66.92%、67.27%，相对图像分支计入系统能耗降低约 55.4%。
- DroneVehicle：验证选价 MLP 为 74.60%、4.47 J/query；类型规则为 72.93%、11.21 J/query。
- DroneVehicle 选价 MLP 的 UAV 计入能耗约 0.45 J，高于固定图像的约 0.13 J。不能声称改善无人机续航。
- 接收器适配：完整公共测试集 104 张图像，但存在性和计数仅覆盖 26 张；不将公共集合均值与全量 VisDrone 均值直接比较。
- 十种子标准差是固定划分下的训练变化，不是独立数据集不确定性。不沿用旧置信区间或声称统计显著/等效。

## 修改与输出

英文修改：main.tex 摘要、01_introduction 的贡献、03_system_model、04_evidence_routing、06_experiments、08_conclusion、supplementary.tex。

中文同步：abstract_zh.tex、introduction_zh.tex、system_model_zh.tex、evidence_routing_zh.tex、experiments_zh.tex；新增 conclusion_zh.tex、supplementary_zh.tex。学术英文采用常用术语与直接句式，明确比较对象、条件和计量边界。

统一绘图脚本：`paper/figures/build_evidence_manuscript.py`；五张图位于 `paper/figures/evidence_final/`，均有 PDF/SVG/400 dpi PNG。原始文件哈希和能耗分解位于 `paper/outputs/evidence_manuscript/`。

最新编译文件在 `paper/build/`，不是旧的 `paper/main.pdf`。英文主稿 23 页；英文补充 4 页；中文实验 6 页。九份 LaTeX 编译均成功，无未定义引用或 overfull 错误；保留模板的 lettersize 提示和非阻塞 underfull 提示。已对主稿全部页面及实验/补充材料进行渲染检查，重点核验架构图、能耗图和接收器表。

## 保留的限制

未测量实际无人机端到端功耗或替代接收器能耗；链路可靠性与空口时间仍是模型核算；结构化问题与场景规模限制泛化。现稿据此收窄主张，不通过新增实验或更换测试集掩盖限制。本轮完成的是证据一致的写作版本，不等同于已完成投稿级外部技术审查。
