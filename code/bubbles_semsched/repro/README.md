# BUBBLES 语义调度实验 · 复现包

把到 2026-10-08 为止、**当前图表实际用到的全部实验**整理成一套可以直接重跑的脚本。
图表读取的每个结果文件对应哪条命令，见 `MANIFEST.md`。

> 仿真和训练只在服务器上跑，不要在 Mac 上跑。Mac 上只做出图、出表和本目录 `tools/` 里的检查（它们不运行任何仿真）。

## 一、三步上手

```bash
# 0) 把本目录拷到服务器，按需改 env.sh（解释器、进程数）；只需要 numpy + torch（CPU 即可）
# 1) 几分钟：按原样重算 6 个策略（测试流量 4 个、开发流量 2 个），与存档数值逐项比对
PY=python3 PROCS=16 bash exp/90_check_reproduction.sh
# 2) 一两个小时：把每一条命令都缩小规模执行一遍（2 组流量、1 个速率、训练 1 轮），只验证能跑通
SMOKE=1 bash run_all.sh
# 3) 全量复现（CPU 时间以天计）；评估默认使用 ckpt/ 里随包附带的已训练检查点
bash run_all.sh                 # 或只跑一部分：ONLY="31 34" bash run_all.sh
```

每条命令的输出在 `$OUT`（默认 `out/`），日志是同名 `.log`；已有输出会跳过（`FORCE=1` 强制重跑）。

## 二、目录

| 路径 | 内容 |
|---|---|
| `env.sh` | 可调设置：解释器 `PY`、评估进程数 `PROCS`、并行训练数 `JOBS`、输出目录 `OUT` |
| `lib.sh` | 所有脚本共用的执行函数：干净的环境变量、依赖检查、跳过已完成、冒烟模式 |
| `run_all.sh` | 按顺序执行 `exp/` 下全部脚本 |
| `exp/` | 按实验分组的命令脚本（**由 `tools/build.py` 生成，不要手改**） |
| `src/` | 仿真与学习代码的固定版本，以及语义侧汇总出的输入 `sim_inputs.json` |
| `ckpt/` | 当时训练得到的 58 个调度策略检查点（约 3 MB），存档结果就是用它们评估出来的；不含检测器和图像编码器的权重 |
| `ref/` | 复现检查用的期望值、结果文件应放回的目录 |
| `as_run/` | 当时在三台服务器上实际运行的链脚本，是命令的唯一来源（公开仓库里的副本隐去了注释中的主机地址，命令未改） |
| `stageA/` | 语义侧（检测器训练、符号层基线、门控 token 扫描、HEVC 对照）的脚本，原样归档 |
| `tools/` | 生成与核对工具（见第五节） |
| `MANIFEST.md` | 结果文件 → 复现脚本 → 当时的链脚本 → 服务器 → 用在哪张图 / 表 |

## 三、流程与脚本

```
语义侧（stageA/，需 GPU）──► src/sim_inputs.json ──► 训练（10–12）──► 评估（20–40）──► ../figs 出图出表
```

| 脚本 | 内容 | 命令数 |
|---|---|---|
| `exp/10_train_proposed.sh` | 提出的方法，最终协议，两组各 5 个训练种子（rollout 进程数 2 与 4） | 10 |
| `exp/11_train_baselines.sh` | 对比学习器 H-PPO、D3QN、TD3，各 5 个训练种子 | 15 |
| `exp/12_train_variants.sh` | 结构消融、一档 / 两档 token、其它召回要求下的重训等 | 见脚本 |
| `exp/20_validation_refs.sh` | 各规则在验证流量上的参照线（收敛图用） | 2 |
| `exp/21_grid_search.sh` | 结构参数 (p, V) 的网格搜索 | 见脚本 |
| `exp/30_dev_traffic.sh` | 开发流量（种子 401–448）上的系统级评估、折中曲线、HEVC | 见脚本 |
| `exp/31_test_traffic.sh` | **测试流量（种子 501–548）上的系统级评估：论文数字** | 见脚本 |
| `exp/32_ablation_levels.sh` | 结构消融、token 档数、全网平均约束对照 | 见脚本 |
| `exp/33_robustness_sensitivity.sh` | 不重训的鲁棒性、对建模假设的敏感性 | 见脚本 |
| `exp/34_recall_requirement.sh` | 容量随召回要求（训一次直接用 / 按要求重训） | 见脚本 |
| `exp/40_behaviour_timing.sh` | 行为诊断（何时发 token）、每次决策的计算时间 | 见脚本 |
| `exp/90_check_reproduction.sh` | 全保真的快速复现检查（不属于实验） | 2 |

依赖关系：评估脚本需要对应的检查点。`$OUT` 里没有时，自动取 `ckpt/` 里随包的那一个（`SHIPPED=0` 则要求先重训）。

## 四、把复现结果接到图表

```bash
python tools/collect.py out res_repro ../res      # 按 ../res 的目录结构摆放，语义侧的两个输入文件从 ../res 拷贝
BUB_RES=$PWD/res_repro python ../figs/make_figs.py
BUB_RES=$PWD/res_repro python ../figs/make_tables.py   # 末尾自动做可行性判据复核
```

## 五、这套脚本是怎么保证和当时一致的

脚本不是手写的，而是从当时的运行记录生成的，并做了三层核对：

1. **命令记录**：`tools/replay_as_run.py` 把 `as_run/` 里 65 个链脚本对着一个"假解释器"重放（不运行任何仿真），记下每条命令的参数和环境变量，共 492 条。
2. **筛选与元数据核对**：`tools/trace_figs.py` 记录当前图表实际打开的结果文件；`tools/build.py` 为每个文件找到产生它的命令，并与结果文件里存的元数据（策略、信道数、速率、缓冲、流量种子、召回要求 / 训练种子）比对。
3. **等价性**：`tools/check_equivalence.py` 把生成的 `exp/*.sh` 同样对着假解释器重放，逐条比较。

以后新增实验的更新方法：把新的链脚本放进 `as_run/`，结果拉回 `../res/`，然后

```bash
python tools/replay_as_run.py && python tools/trace_figs.py && python tools/build.py && python tools/check_equivalence.py
```

## 六、验证记录（2026-10-08）

| 检查 | 在哪里做 | 结果 |
|---|---|---|
| 生成脚本与运行记录逐条比较（`tools/check_equivalence.py`） | Mac（不运行仿真） | 199 条命令全部相同 |
| 命令与结果文件元数据比对（`tools/build.py`） | Mac | 185 条一致，12 条的结果文件无可比元数据，2 条为重建 |
| 全保真复现检查（`exp/90_check_reproduction.sh`，48 组流量，M=4、60 架次/h、无缓冲。测试流量：drift-plus-penalty、网格搜索规则、提出的方法种子 0、H-PPO 种子 0；开发流量：一条规则和早期两阶段协议的一个策略） | AutoDL | 42 个数值与存档**完全相同**（最大绝对差 0），用时约 10 分钟 |
| 全流程冒烟（`SMOKE=1 bash run_all.sh`） | 182 | 199 条命令全部执行通过。首轮 198 条通过、1 条失败（`main_M4_48seeds`，重建的命令有误，见第七节）；修正后补跑通过，11 个脚本均报告全部完成。首轮用时约 2 小时 20 分 |

## 七、已知事项

- **评估进程数不影响结果**，所以统一成 `PROCS`；**训练命令里的 rollout 进程数会影响随机数序列**，保持当时的数值不变。
- **提出的方法有两组训练，各 5 个种子**：`sppoX_fast5_s*`（rollout 进程数 2，在 3090 上训）与 `sppoX_fast_s*`（rollout 进程数 4，在 182 上训）。协议、种子、超参、训练量都相同，但 rollout 进程数不同，随机数序列就不同，所以两组训出的参数不同——这**不能只归因于机器差异**（整理本包时才发现这一点）。两组都随包附带，都进了结果表。
- **源码版本**：`src/` 是 3090 上的版本（与本地一致）。AutoDL 与 182 上当时的若干文件是较早版本，差别只是后来新增的、默认关闭的选项；`hppo_hold.py` 的差别是"token 档数不是 4 时观测特征越界"的修复，四档时行为不变。
- **两条命令是重建的**（`MANIFEST.md` 里标为 rebuilt），当时是手敲的、没有链脚本记录：`mrl5k_B_M4_s0`（H-PPO 种子 0 的训练）按种子 1–4 的记录命令重建；`main_M4_48seeds`（开发流量 M=4 网格）按有记录的同类命令（M=3、M=5）和结果文件里的元数据重建。后者第一版重建有误（把 5 个学出的策略写成了规则名），被冒烟测试查出；改正后它用的检查点由全保真复现检查确认（开发流量上两个单元与存档完全相同）。前者只核对了元数据，没有重训验证。
- **语义侧未在本包内重跑**：`stageA/` 是原样归档，需要 GPU、VisDrone 数据、SwinJSCC 权重和 x265。当时的命令是
  `BUB_DATA=~/bub_work BUB_ROOT=~/bub_store BUB_WORKERS=8 BUB_HEVC_WORKERS=12 python colab_run_all_cell.py`，
  之后用 `build_sim_inputs.py` 汇总成 `sim_inputs.json`。
- **图表没用到的探索性实验不在 `exp/` 里**，但它们的链脚本都在 `as_run/`，可按同样方法生成。
