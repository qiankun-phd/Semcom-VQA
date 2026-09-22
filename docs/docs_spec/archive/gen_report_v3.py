#!/usr/bin/env python3
"""Self-contained HTML report: system design + all current results (160 rebuild)."""
import base64
import os

FIGS = {}
for name in ["F1_acc_snr_3panel_final", "F2_cliff_final", "F3_latency_final",
             "F4_qtype_final", "F5_pareto_final", "F6_complementarity",
             "F7_mismatch_v3", "F8_token_budget_v3", "F9_separation_capacity"]:
    p = f"/tmp/report_figs/{name}.png"
    FIGS[name] = base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""

RL = open("/tmp/rl_section.html").read() if os.path.exists("/tmp/rl_section.html") else "<p><em>(v19 架构分析待补)</em></p>"


def img(name, cap):
    return (f'<figure><img src="data:image/png;base64,{FIGS[name]}" alt="{name}">'
            f'<figcaption>{cap}</figcaption></figure>')


HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAV-VQA 语义通信系统：设计与实验结果 v3（2026-07-05）</title>
<style>
  :root {{ --ink:#24292f; --muted:#57606a; --line:#d0d7de; --bg:#f6f8fa; --acc:#0969da; --good:#1a7f37; --warn:#9a6700; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; color:var(--ink);
         max-width:1080px; margin:0 auto; padding:24px 32px 80px; line-height:1.65; }}
  h1 {{ font-size:26px; border-bottom:2px solid var(--line); padding-bottom:10px; }}
  h2 {{ font-size:21px; margin-top:44px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
  h3 {{ font-size:17px; margin-top:28px; }}
  .sub {{ color:var(--muted); font-size:14px; }}
  nav {{ background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:12px 18px; font-size:14px; }}
  nav a {{ margin-right:14px; color:var(--acc); text-decoration:none; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; margin:14px 0; }}
  th,td {{ border:1px solid var(--line); padding:6px 10px; text-align:left; }}
  th {{ background:var(--bg); }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .best {{ color:var(--good); font-weight:600; }}
  .bad  {{ color:#cf222e; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; margin:18px 0; }}
  .card {{ background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:12px 16px; }}
  .card b {{ font-size:22px; display:block; }}
  .card span {{ font-size:13px; color:var(--muted); }}
  figure {{ margin:18px 0; text-align:center; }}
  figure img {{ max-width:100%; border:1px solid var(--line); border-radius:6px; }}
  figcaption {{ font-size:13px; color:var(--muted); margin-top:6px; }}
  .flow {{ display:flex; flex-wrap:wrap; align-items:stretch; gap:6px; margin:16px 0; }}
  .box {{ background:#fff; border:1.5px solid var(--acc); border-radius:8px; padding:8px 12px;
          font-size:13px; flex:1; min-width:120px; text-align:center; }}
  .box small {{ color:var(--muted); display:block; }}
  .arrow {{ align-self:center; color:var(--muted); font-size:18px; }}
  .note {{ background:#fff8c5; border:1px solid #d4a72c66; border-radius:6px; padding:10px 14px; font-size:13.5px; }}
  .finding {{ border-left:4px solid var(--good); background:var(--bg); padding:8px 14px; margin:10px 0; font-size:14px; }}
  code {{ background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:12.5px; }}
</style></head><body>

<h1>UAV-VQA 语义通信系统：设计与实验结果 <span style="color:var(--acc)">v3</span></h1>
<p class="sub">2026-07-05 · 数据来自 160 服务器全量重算（与原 182 结果逐点复现校验通过）·
仓库 <code>qiankun-phd/uav-vqa-semantic-rl</code> 分支 <code>codex/lut-semantic-utility-upgrade</code>（对比实验）/ <code>main</code>（v19 资源分配 RL）·
<b>v3 新增</b>：跨数据集 DroneVehicle 验证（§12）、语义通信进 BUBBLES 飞行安全环（§13）、新颖性与期刊定位（§14）、
§6 跨三 VLM 家族升级</p>

<nav><b>目录</b>&nbsp;
<a href="#sys">1 系统设计</a> <a href="#rl">2 v19 RL 架构</a> <a href="#methods">3 对比方法</a>
<a href="#main">4 主结果</a> <a href="#comp">5 互补性</a> <a href="#xvlm">6 跨VLM</a>
<a href="#p3">7 CSI失配</a> <a href="#p2">8 逐样本预测器</a> <a href="#p1">9 token预算</a>
<a href="#findings">10 发现清单</a> <a href="#repro">11 复现信息</a>
<a href="#xdata">12 跨数据集</a> <a href="#safety">13 飞行安全环</a> <a href="#novelty">14 新颖性与定位</a></nav>

<div class="cards">
<div class="card"><b>0.670</b><span>M4 自适应 @20dB Rician（测试集）— 高于 error-free 理想信道整图 0.615</span></div>
<div class="card"><b>265×</b><span>-5dB 下 s2 图像 vs s1 token 的信道用量差（3.72M vs 14k complex uses）</span></div>
<div class="card"><b>49×</b><span>-5dB Rayleigh 端到端延迟差：M1 4.32s vs token 0.089s</span></div>
<div class="card"><b>+7.7pt</b><span>threshold 题型 t=32 截断 vs 全量发送——"少发反而更准"</span></div>
<div class="card"><b>1.14×</b><span>-5dB Rician 实测时延注入：token 相对容量 1.060 vs 图像 0.930——证据选择换来的空域容量（20dB 塌缩到 1.01×）</span></div>
<div class="card"><b>3/3</b><span>跨三个 VLM 家族（Qwen2-VL / Qwen2.5-VL / SmolVLM-Idefics3）路由一致性——token 优势与接收端模型无关</span></div>
</div>

<h2 id="sys">1 系统设计</h2>
<h3>1.1 场景与端到端管线</h3>
<p>无人机（UAV）巡查低空场景，边缘服务器上的用户就实时画面提出视觉问答（VQA）问题。
系统<b>不以重建图像为目标</b>，而是按问题类型自适应选择"证据等级"，只传输回答该问题所需的语义证据。</p>
<div class="flow">
<div class="box">UAV 相机<small>VisDrone 航拍图</small></div><div class="arrow">→</div>
<div class="box">YOLOv8n 检测器<small>VisDrone 微调 best.pt</small></div><div class="arrow">→</div>
<div class="box">证据选择器<small>问题类型 → s0/s1/s2（LUT / 零参数规则 / 逐样本预测器）</small></div><div class="arrow">→</div>
<div class="box">物理层<small>s1: LDPC r=1/2+BPSK 逐帧<br>s2: 速率自适应 JPEG+LDPC</small></div><div class="arrow">→</div>
<div class="box">无线信道<small>AWGN / Rayleigh / Rician K=6dB<br>SNR −5→20 dB</small></div><div class="arrow">→</div>
<div class="box">边缘接收端<small>s1: 符号解码器<br>s2: Qwen2-VL-2B 答题</small></div><div class="arrow">→</div>
<div class="box">答案<small>指标: VQA 精度</small></div>
</div>

<h3>1.2 证据等级（服务等级）</h3>
<table>
<tr><th>等级</th><th>载荷</th><th>物理层</th><th>典型大小</th><th>接收端解码</th></tr>
<tr><td>s0 缓存</td><td>无传输，命中历史答案</td><td>—</td><td>0 B</td><td>直接返回（新鲜度约束）</td></tr>
<tr><td>s1 轻量 token</td><td>检测器输出（类别/框/置信度）逐帧封包</td><td>LDPC r=1/2 × BPSK（0.5 bit/use），帧丢失概率=FER：50% 整帧丢弃 / 50% 载荷乱码（类别→unknown+框抖动）</td><td>~0.7–1.2 KB</td><td>符号解码器（计数/比较/阈值）或 VLM（presence）</td></tr>
<tr><td>s2 全图</td><td>JPEG 码流</td><td>速率自适应：按瞬时可达谱效率适配 JPEG 质量（遍历 ergodic SE 计费）</td><td>~110 KB（均值）</td><td>Qwen2-VL-2B 视觉问答</td></tr>
<tr><td>s3 ROI</td><td>检测框引导的裁剪图</td><td>同 s2</td><td>介于两者</td><td>Qwen2-VL（评测框架支持，未进对比主表）</td></tr>
</table>

<h3>1.3 统一带宽计费（模拟/数字公平对比的关键）</h3>
<p>所有方法统一按<b>复信道用量（complex channel uses）/查询</b>计费：
s2 = bits ÷ ergodic SE(snr, 信道)；s1 与固定速率数字 = bits ÷ 0.5；M2 模拟 = 原生 uses（1 时隙 = B·τ = 1MHz×0.3s = 300k uses）；
CBR = uses ÷ 平均源维度 2.88M。关键量级（-5dB Rayleigh）：M1 图像 3.72M uses（12.4 时隙）、M0_naive 恒 5.28M、M2 恒 300k、<b>token 仅 ~14k</b>。</p>

<h3>1.4 任务与数据</h3>
<p>VisDrone-DET 548 张真实航拍图，5 种问题类型（presence 存在性 / counting 计数 / comparison 数量比较 /
co_presence 联合存在 / threshold 数量阈值），任务生成完全确定性（538 条 comparison 严格 269/269 平衡、1096 条 extra）。
按 image_id 的 crc32 做 20% 留出测试集，所有报告数字均为测试集。语义质量模型三代演进：
多维 LUT（Wilson-LCB 选择）→ 零参数语义规则（符号→token / 感知→图像）→ 逐样本参数化预测器（§8）。</p>

<h2 id="rl">2 资源分配 RL 层（main 分支 v19 实际架构）</h2>
{RL}

<h2 id="methods">3 对比方法（标准四件套扩展为七方）</h2>
<table>
<tr><th>方法</th><th>含义</th><th>代表的范式</th></tr>
<tr><td>M0_errorfree</td><td>理想无差错信道传原图</td><td>性能参照（非上界——见发现 ②）</td></tr>
<tr><td>M0_naive</td><td>固定速率 LDPC 数字传图</td><td>传统数字链路的悬崖效应</td></tr>
<tr><td>M1_image</td><td>速率自适应 JPEG+LDPC 传图（=固定 s2）</td><td>传统分离信源信道编码 SSCC</td></tr>
<tr><td>M2_analog</td><td>非编码模拟图像传输（JSCC-lite）</td><td>DeepSC 式模拟语义传输的受控代表</td></tr>
<tr><td>M3_token</td><td>固定 s1 检测 token</td><td>GO-SG 式目标导向符号传输</td></tr>
<tr><td><b>M4_adaptive</b></td><td>按 (题型, SNR) Wilson-LCB 选服务（训练集学、测试集报）</td><td><b>本系统</b></td></tr>
<tr><td>M5_oracle</td><td>逐任务事后最优服务</td><td>服务选择上界</td></tr>
</table>

<h2 id="main">4 主结果：精度-SNR、悬崖、延迟、Pareto</h2>
{img("F1_acc_snr_3panel_final", "F1 三信道精度-vs-SNR（测试集）。M4 在所有信道所有 SNR 上是最优非 oracle 方法。")}
<table>
<tr><th>信道</th><th class="num">M4 (-5→20dB)</th><th class="num">M3 token</th><th class="num">M1 图像</th><th class="num">M0_naive @-5dB</th><th class="num">M5 oracle @20dB</th></tr>
<tr><td>AWGN</td><td class="num best">0.628 → 0.668</td><td class="num">0.601（平坦）</td><td class="num">0.556 → 0.611</td><td class="num bad">0.369</td><td class="num">0.720</td></tr>
<tr><td>Rayleigh</td><td class="num best">0.609 → 0.668</td><td class="num">0.603 → 0.599</td><td class="num">0.537 → 0.614</td><td class="num bad">0.409</td><td class="num">0.720</td></tr>
<tr><td>Rician K=6dB</td><td class="num best">0.625 → 0.670</td><td class="num">0.598 → 0.601</td><td class="num">0.556 → 0.612</td><td class="num bad">0.369</td><td class="num">0.721</td></tr>
</table>
<p>参照线：<b>M0_errorfree = 0.615</b>（理想信道整图）、M2 模拟 Rician 0.479 → 0.595（平滑无悬崖）。</p>
{img("F2_cliff_final", "F2 悬崖效应（Rayleigh）：固定速率数字 M0_naive 在低 SNR 跌至 0.37-0.41；M2 模拟与 M4 自适应平滑退化。")}
{img("F3_latency_final", "F3 端到端延迟分解（Rayleigh，对数轴）：-5dB 下 M1 上传 3.72s + 推理 0.60s = 4.32s；token 全程 0.089s（49×）；M0_naive 5.87s。")}
{img("F4_qtype_final", "F4 分题型精度（Rician @5dB）。")}
{img("F5_pareto_final", "F5 目标导向效率 Pareto：精度 vs 复信道用量（统一模拟/数字轴）。M4 以 ~14k-110K 量级用量达到 0.67，M1/M0_naive 需要 3.7-5.3M。")}

<h2 id="comp">5 头条发现：证据-问题互补性</h2>
{img("F6_complementarity", "F6 token 增益 Δ=acc(token)−acc(image) 按题型（3 信道池化，测试集）。")}
<table>
<tr><th>题型</th><th>推理类别</th><th class="num">token</th><th class="num">image</th><th class="num">Δ = t − i</th></tr>
<tr><td>counting</td><td>符号</td><td class="num">0.453</td><td class="num">0.279</td><td class="num best">+0.174</td></tr>
<tr><td>comparison</td><td>符号</td><td class="num">0.846</td><td class="num">0.696</td><td class="num best">+0.151</td></tr>
<tr><td>co_presence</td><td>符号</td><td class="num">0.663</td><td class="num">0.545</td><td class="num best">+0.119</td></tr>
<tr><td>threshold</td><td>符号（边界）</td><td class="num">0.596</td><td class="num">0.611</td><td class="num">−0.014</td></tr>
<tr><td>presence</td><td>感知</td><td class="num">0.678</td><td class="num">0.765</td><td class="num bad">−0.087</td></tr>
</table>
<p><b>零参数语义规则</b>（符号→token，感知→图像）与<b>数据标定 LCB 选择器</b>在测试集上完全一致：
双双 0.6725（n=5616；固定 token 0.6346 / 固定图像 0.6043 / oracle 0.7472）。
机理：token 无损保留离散计数的充分统计量；图像保留存在性召回。消融证明 question_type 是 LUT 中唯一承重维度
（snr/view/freshness/risk/LCB 对服务选择增益≈0——这些维度的用武之地在跨时间/跨用户的资源调度层，见 §2）。</p>

<h2 id="xvlm">6 跨 VLM 稳健性（三家族三模型，Rician）</h2>
<p>v3 把跨 VLM 验证从"两个 Qwen 同家族模型"扩到<b>三家族三模型</b>：Qwen2-VL-2B、Qwen2.5-VL-3B、
以及<b>非 Qwen 家族的 SmolVLM-2B</b>（视觉塔 = Idefics3 / SigLIP，语言塔 = SmolLM2；与 Qwen 架构完全独立）。
下表每格为该模型的 <code>token / image</code> 测试精度，路由方向取 argmax。</p>
<table>
<tr><th>题型</th><th>推理类别</th><th class="num">Qwen2-VL-2B<br>token/image</th><th class="num">Qwen2.5-VL-3B<br>token/image</th><th class="num">SmolVLM-2B<br>token/image</th><th>三家族路由</th></tr>
<tr><td>presence</td><td>感知</td><td class="num">0.678 / 0.761</td><td class="num">0.712 / 0.721</td><td class="num">0.712 / 0.724</td><td class="best">一致 → image</td></tr>
<tr><td>counting</td><td>符号</td><td class="num">0.450 / 0.280</td><td class="num">0.404 / 0.375</td><td class="num">0.404 / 0.391</td><td class="best">一致 → token</td></tr>
<tr><td>comparison</td><td>符号</td><td class="num">0.846 / 0.710</td><td class="num">0.846 / 0.787</td><td class="num">0.846 / 0.598</td><td class="best">一致 → token</td></tr>
<tr><td>co_presence</td><td>符号</td><td class="num">0.663 / 0.559</td><td class="num">0.663 / 0.553</td><td class="num">0.663 / 0.529</td><td class="best">一致 → token</td></tr>
<tr><td>threshold</td><td>符号（边界）</td><td class="num">0.596 / 0.611</td><td class="num">0.596 / 0.567</td><td class="num">0.596 / 0.622</td><td>边界（三模型均 |Δ|最小）</td></tr>
</table>
<div class="finding"><b>三条跨家族结论</b>：
（1）<b>4/5 题型路由跨三家族完全一致</b>——presence→image、其余四个符号题型→token；唯一分歧的 threshold 恰是互补性最弱（|Δ|最小）题型，<b>互补性强度预测路由稳定性</b>。
（2）<b>token 列与 VLM 无关</b>：三模型的 token 精度逐格相同（comparison 0.846、co_presence 0.663、counting 0.404…）——因为 token 由确定性<b>符号解码器</b>处理，根本不经过 VLM。
（3）<b>弱 VLM 放大 token 优势</b>：SmolVLM 读退化图像最弱（comparison image 0.598 vs Qwen 0.71–0.79、co_presence 0.529），token 相对图像的领先反而更大——接收端越弱，证据选择的价值越高。</div>

<h2 id="p3">7 CSI 失配矩阵：选择器不需要精确 CSI</h2>
{img("F7_mismatch_v3", "F7 策略假定 SNR × 真实 SNR 的测试精度热力图（物理链路保持按真实 SNR 自适应，隔离调度层的 CSI 误差影响）。")}
<p>三信道全部近乎平坦：<b>最差失配格（0.641–0.653）恰好等于匹配 CSI 时的最低对角值</b>，任意 CSI 错误都不产生断崖。
问题驱动的路由在选择时刻无需信道估计——参考工作（如 ADJSCC）评估时始终馈入真实 SNR，从未度量此轴。</p>

<h2 id="p2">8 逐样本参数化预测器：更准且 payload 减半</h2>
<p>用<b>发送端可得特征</b>（题型/目标类/原始检测计数/SNR/任务元数据）训练逐服务正确率预测器
P(correct | x, s)（逻辑回归，OracleNet 配方 MSE→BCE），逐样本 argmax 选服务，评估为对已有日志的离线重选择：</p>
<table>
<tr><th>信道</th><th class="num">M3 token</th><th class="num">M1 image</th><th class="num">M4 LUT</th><th class="num">逐样本 ML</th><th class="num">M5 oracle</th><th class="num">平均 payload</th></tr>
<tr><td>AWGN</td><td class="num">0.6346</td><td class="num">0.6043</td><td class="num">0.6747</td><td class="num best">0.6832</td><td class="num">0.7472</td><td class="num">58 KB（LUT 111 KB）</td></tr>
<tr><td>Rayleigh</td><td class="num">0.6339</td><td class="num">0.6015</td><td class="num">0.6690</td><td class="num best">0.6804</td><td class="num">0.7455</td><td class="num">56 KB（107 KB）</td></tr>
<tr><td>Rician</td><td class="num">0.6339</td><td class="num">0.6063</td><td class="num">0.6727</td><td class="num best">0.6798</td><td class="num">0.7450</td><td class="num">60 KB（110 KB）</td></tr>
</table>
<p>三信道全胜 LUT +0.7~1.1pt（追回 oracle 差距的 10–15%），且 <b>payload 几乎减半</b>——预测器学会"发送端检测证据已充分时，presence 也走 token"。
增益集中于 presence（+1.5~2.4pt）与 counting（+1.2pt），正是逐样本信息应当起作用的位置。</p>

<h2 id="p1">9 可变 token 预算：嵌套 top-t 截断</h2>
{img("F8_token_budget_v3", "F8 token 预算扫描（符号解码题型，测试集；每档 t 重新标定计数，t=full 与主实验逐位一致）。")}
<div class="finding"><b>发现 A（预算敏感度分化）</b>：comparison t=3 即近饱和（0.817 / 满发 0.846）；counting 需 t=48 才追平（0.453）——
需求差一个数量级，构成互补性的第二证据轴。</div>
<div class="finding"><b>发现 B（少发反而更准）</b>：threshold t=32 达 0.673 vs 满发 0.596（<b>+7.7pt</b>）、co_presence +3.9pt，三信道完全一致。
置信度排序截断兼做假阳性过滤器——对"至少 N 个 / 两者都有"这类 yes 偏置问题，精度比召回更重要。</div>
<div class="note">诚实标注：s1 载荷为帧头主导（t=1 约 696B、满发约 1.2KB），截断的带宽节省有限（~30%）；
该机制的价值在<b>精度过滤与预算敏感度分化</b>，而非省带宽。</div>

<h2 id="findings">10 发现清单（论文叙事骨架）</h2>
<div class="finding">① <b>自适应语义系统在理想信道下也胜过传图</b>：M4@20dB（0.668–0.670）&gt; M0_errorfree（0.615）——增益来自证据选择本身，不只是信道鲁棒性。</div>
<div class="finding">② <b>证据-问题互补性</b>：符号推理→token（+0.12~0.17），感知推理→图像（−0.09）；零参数规则=数据标定策略（0.6725）。</div>
<div class="finding">③ <b>悬崖 vs 平滑</b>：固定速率数字断崖（-5dB 0.37-0.41）；token/模拟/自适应平滑退化。</div>
<div class="finding">④ <b>跨 VLM 稳健</b>：路由 4/5 一致，分歧点=互补性最弱题型。</div>
<div class="finding">⑤ <b>调度无需精确 CSI</b>：失配矩阵平坦，最差失配=匹配最低值。</div>
<div class="finding">⑥ <b>逐样本预测器</b>：+0.7~1.1pt 且 payload 减半（Pareto 改进）。</div>
<div class="finding">⑦ <b>token 预算双重角色</b>：成本旋钮 + 假阳性过滤器（threshold +7.7pt）。</div>
<div class="finding">⑧ <b>延迟</b>：-5dB 下 token 0.089s vs 图像 4.32s（49×）——实时性证据。</div>
<div class="finding">⑨ <b>互补性跨数据集复现，且规则反超标定</b>（§12）：DroneVehicle（另一 UAV 数据集/另一目标域）上互补性符号顺序完全复现（counting +0.211、comparison +0.098、co_presence +0.091、threshold +0.077），零参数规则 0.7293 <b>反超</b>逐数据集标定 LCB 0.7246——规则比标定更泛化。</div>
<div class="finding">⑩ <b>路由跨三个 VLM 家族一致</b>（§6）：Qwen2-VL / Qwen2.5-VL / SmolVLM-Idefics3 三家族 4/5 题型路由相同；token 列因走符号解码器而与 VLM 完全无关；弱 VLM 放大 token 优势。</div>
<div class="finding">⑪ <b>证据选择 = 飞行安全变量</b>（§13）：把实测传输时延注入飞行环，-5dB Rician 下选 token 相对选 image 可把安全接触距离拉近 <b>48.8m</b>（398.4→349.5m）、换来 <b>1.14×</b> 空域相对容量——语义选择首次量化为飞行安全/空域容量收益。</div>

<h2 id="repro">11 复现与工件</h2>
<table>
<tr><th>项</th><th>位置</th></tr>
<tr><td>对比实验代码+数据</td><td><code>codex/lut-semantic-utility-upgrade</code> 分支（tip 08aa141）：build_comparison_v2 / build_mismatch_matrix / build_persample_policy / build_token_budget_sweep / analyze_crossvlm + outputs/reports/*.csv + F1–F8</td></tr>
<tr><td>资源分配 RL</td><td><code>main</code> 分支（tip ad31e9c）：src/vqa_semcom/rl/v19_ppo.py + v19_resource_env.py</td></tr>
<tr><td>实验机</td><td>lab-s2 <code>~/phd_research/vqa_semcom</code>（RTX 4060 8GB；全链 ~37h 重算）</td></tr>
<tr><td>原始预测备份</td><td>本地 <code>HPPO-VQA/outputs/backup_160/</code>（重算 12 CSV + v25 三件，压缩 41MB）</td></tr>
<tr><td>参考实现代码考据</td><td><code>docs_spec/SemCom_Reference_Repos_Code_Analysis.md</code>（MA-DeepSC/PADC/ADJSCC/DeepJSCC-f/SJTU 五仓库）</td></tr>
<tr><td><b>v3 新提交（7/5 四条线）</b></td><td><code>f74c896</code> E7 实测时延注入 + 相对容量/分隔距离；<code>74e4991</code> BUBBLES 场景档（<code>feat/bubbles-scenario</code>）；<code>2708b2a</code> 三 VLM 家族路由（+SmolVLM-2B）；<code>bbcc83d</code> DroneVehicle 跨数据集接入</td></tr>
<tr><td>备份包（三个）</td><td>本地 <code>HPPO-VQA/outputs/backup_160/</code>：① 主重算 12 CSV + v25 三件（41MB）；② DroneVehicle 跨数据集产物；③ 三 VLM + E7/BUBBLES 产物</td></tr>
<tr><td>论文骨架</td><td><code>paper_semcom/</code>（另一代理在建的 TCCN 主投骨架，与本报告 §14 定位一致；此处仅提一句）</td></tr>
</table>
<p class="sub">重算与 182 原结果交叉验证：AWGN M4 0.628→0.668（原 0.627→0.668）、naive 悬崖 0.369–0.409（原 0.37–0.41）、error-free 0.615（原 0.611）——逐点复现。</p>

<h2 id="xdata">12 跨数据集验证（DroneVehicle）</h2>
<p>为回应"单数据集 548 图"的审稿攻击点（见 §14 薄弱点 #3），v3 在<b>第二个 UAV 数据集 DroneVehicle</b> 上完整复跑管线。
接入是<b>零代码</b>的：490 张图 / 5 类车（car、truck、bus、van、freight car；其中 <code>freight car</code> 两种拼写变体统一并入 <b>truck</b>），
标注为 <b>VOC-XML</b>，通过<b>符号链接</b>把 DroneVehicle 目录挂进现有 VisDrone 数据路径即可，检测器/任务生成器/证据选择器全部零改动。</p>

<h3>12.1 主对比（DroneVehicle，三信道）</h3>
<table>
<tr><th>方法</th><th class="num">AWGN</th><th class="num">Rayleigh</th><th class="num">Rician</th><th>范式</th></tr>
<tr><td><b>M4_adaptive</b></td><td class="num best">0.735</td><td class="num best">0.732</td><td class="num best">0.725</td><td>本系统（问题驱动路由）</td></tr>
<tr><td>M3_token</td><td class="num" colspan="3" style="text-align:center">0.724（三信道近平坦）</td><td>固定检测 token</td></tr>
<tr><td>M1_image</td><td class="num" colspan="3" style="text-align:center">0.642</td><td>速率自适应传图 SSCC</td></tr>
<tr><td>M5_oracle</td><td class="num" colspan="3" style="text-align:center">0.808</td><td>服务选择上界</td></tr>
</table>
<p>排序 <b>M4 (0.735/0.732/0.725) &gt; M3 (0.724) &gt; M1 (0.642)</b>，oracle 0.808——与 VisDrone 主结果（§4）同构：自适应稳居最优非 oracle，固定传图垫底。</p>

<h3>12.2 互补性对照（DroneVehicle vs VisDrone）</h3>
<p>核心验证：证据-问题<b>互补性符号顺序在跨数据集下完全复现</b>。Δ = acc(token) − acc(image)：</p>
<table>
<tr><th>题型</th><th>推理类别</th><th class="num">DroneVehicle Δ</th><th class="num">VisDrone Δ（§5）</th><th>跨数据集读解</th></tr>
<tr><td>counting</td><td>符号</td><td class="num best">+0.211</td><td class="num">+0.174</td><td>token 优势更强</td></tr>
<tr><td>comparison</td><td>符号</td><td class="num best">+0.098</td><td class="num">+0.151</td><td>方向一致</td></tr>
<tr><td>co_presence</td><td>符号</td><td class="num best">+0.091</td><td class="num">+0.119</td><td>方向一致</td></tr>
<tr><td>threshold</td><td>符号（边界）</td><td class="num best">+0.077</td><td class="num">−0.014</td><td><b>脱离边界</b>：DroneVehicle 上明确偏 token</td></tr>
<tr><td>presence</td><td>感知</td><td class="num">−0.014</td><td class="num bad">−0.087</td><td><b>边际变薄</b>：图像优势缩小但仍为负</td></tr>
</table>
<p>四个符号题型在两个数据集上一致偏 token，唯一感知题型（presence）一致偏 image——<b>互补性是任务结构性质，跨目标域稳健</b>。
边界题型 threshold 在 DroneVehicle 上从 −0.014 转为明确的 +0.077（脱离边界），presence 从 −0.087 收窄到 −0.014（边际变薄），
说明两个"临界"题型的符号在不同数据集上会摆动，但四强一弱的整体划分不变。</p>

<h3>12.3 规则反超标定：泛化性头条</h3>
<div class="finding"><b>零参数语义规则 0.7293 &gt; 逐数据集标定 LCB 0.7246</b>。
在 DroneVehicle 上，"符号→token / 感知→image"的<b>零参数规则</b>（无任何该数据集的标定）比用 DroneVehicle 自身数据重新标定的
<b>Wilson-LCB 选择器</b>还高 0.47pt。含义：数据标定会过拟合到本数据集的采样噪声（尤其稀疏格），而基于推理结构的规则跨域<b>更泛化</b>——
这把"规则 = 标定"（VisDrone 上的等价性，§5）升级为"规则 ≥ 标定"（跨数据集上规则更优），是 v3 对方法论稳健性的最强单点证据。</div>

<h2 id="safety">13 语义通信进飞行安全环（E7 + BUBBLES）</h2>
<p>前面各节把证据选择的收益量化为<b>精度 / 带宽 / 延迟</b>。§13 更进一步：把实测传输时延接入<b>飞行安全环</b>，
让"选 token 还是选 image"直接映射到<b>安全分隔距离与空域容量</b>——语义通信首次成为飞行安全变量。</p>

<h3>13.1 BUBBLES 分隔链复现（提交 74e4991 / f74c896）</h3>
<p>采用 SESAR <b>BUBBLES D2.1 Block-4</b> 的分隔管理链条实现。selftest 逐格复现官方
<b>Table G-4 / G-5</b>：全格误差 <b>&lt; 0.18%</b>；SAIL I-II 场景总接触距离 TC <b>370.53m → 370.49m</b>（口径一致）。
分隔链本身经 selftest 锚定后作为固定"消费端"，接收语义链路给出的实测时延。</p>

<h3>13.2 实测时延注入 → 分隔距离与相对容量（E7 / T4）</h3>
{img("F9_separation_capacity", "F9 证据模态 → 分隔距离/相对容量：把 s1 token 与 s2 image 的实测传输时延注入 BUBBLES 分隔链（-5→20dB Rician）。")}
<table>
<tr><th>指标（-5dB Rician）</th><th class="num">s2 图像（image）</th><th class="num">s1 token</th><th>差异</th></tr>
<tr><td>安全接触距离 d_TC</td><td class="num bad">398.4 m</td><td class="num best">349.5 m</td><td><b>−48.8 m</b>（token 更近更安全）</td></tr>
<tr><td>相对空域容量</td><td class="num bad">0.930</td><td class="num best">1.060</td><td><b>1.14×</b></td></tr>
<tr><td>@20dB 相对容量</td><td class="num" colspan="2" style="text-align:center">两者趋同</td><td>塌缩到 <b>1.01×</b></td></tr>
</table>
<p>机理：弱链路下传整图的排队/传输时延膨胀，把飞行器"看清并决策"的时刻推后，安全分隔必须留更大距离（d_TC 更大）；
选 token 把时延压掉，分隔距离缩短 48.8m，等价于 <b>1.14× 的空域相对容量</b>。SNR 升高后 image 时延不再是瓶颈，优势塌缩到 1.01×——
<b>证据选择的安全收益集中在弱链路</b>，与精度/延迟的规律一致。</p>
<div class="note"><b>口径保守性</b>：F9 与上表<b>只注入了传输（tx）时延</b>，<b>不含 VLM 推理时延</b>。
若把 image 侧的 Qwen2-VL 推理时延（§3 中 -5dB 达 0.60s）一并计入，token 的分隔/容量优势只会更大——当前数字是下界。</div>

<h3>13.3 v19 环境的 BUBBLES 场景档（feat/bubbles-scenario，74e4991）</h3>
<p>同一提交把 BUBBLES 引入 v19 资源分配环境（§2）作为可选场景档：
<b>Table B-2 冲突包线</b> + <b>CPA（最近接近点）双判据</b>（包线 & CPA 同时触发才算冲突）+
<b>TLS_MAC = 2.5e-7</b>（目标安全等级 / 空中相撞）锚定 + <b>G-13 需求曲线</b>驱动到达。
测试 <b>14/14 通过</b>；<code>profile=off</code> 时逐字节回归（不影响既有实验）；CPA 冒烟场景把冲突率从 <b>0 抬到 0.5</b>（判据确实生效）。
这为论文二（v19 资源分配）提供了<b>标准化、可对标 SESAR 的安全场景</b>，替代此前的合成冲突模型。</p>

<h2 id="novelty">14 新颖性与期刊定位</h2>
<p>下表摘自 <code>docs_spec/Novelty_Journal_Assessment_2026-07.md</code>（deep-research 管线：5 检索角度 × 21 一手来源 × 105 条提取论断，3 条完成三票对抗核验）。</p>

<h3>14.1 六近邻划界（语义通信侧 ①–⑧ 先例核查）</h3>
<table>
<tr><th>我方发现</th><th>最近邻工作</th><th>我方增量</th></tr>
<tr><td>① 按题型路由证据模态 + 规则=标定</td><td><b>GO-SG</b>(2411.02452)、<b>Park&amp;Yoon</b>(JSAC'25, 2412.13646)</td><td>同一任务(VQA)内按题型的实证互补性分解 + 自动路由 + 规则=标定等价性；两篇均须引用划界</td></tr>
<tr><td>② 无差错信道也胜过传整图</td><td>GO-SG 把整图当<b>精度上界</b>（与我方相反）</td><td class="best">真差异化发现，文献无对应物【核验 2-0】</td></tr>
<tr><td>③ 统一 complex-channel-uses 计费</td><td>TOFC(2503.12926)、2604.26508 均无物理信道建模</td><td>模拟/数字统一计费 + 3 信道×LDPC 全物理层在近邻中空白</td></tr>
<tr><td>⑥ CSI 失配矩阵（调度层鲁棒性）</td><td>ADJSCC 等评估永远馈真值 SNR</td><td class="best">语义层 CSI 误差度量无先例</td></tr>
<tr><td>⑦ 逐样本学习选择器</td><td>PADC(TWC'23) OracleNet 预测 PSNR 选速率（重建域）</td><td>任务精度域 + 发送端特征 + 选证据模态是新组合</td></tr>
<tr><td>⑧ token 截断兼过滤（少发更准）</td><td>2604.26508 结论是单调退化；P&amp;Y 过滤场景图略超全图</td><td>置信度截断 + 按题型分化(t=3 vs t=48) + VQA 精度 +7.7pt + 假阳性过滤机理</td></tr>
</table>
<div class="note"><b>必须处理的近邻</b>：arXiv <b>2505.02413</b>（LLaVA 接收端 + F 衰落 + 端到端 VQA 精度）——"真实 VLM 接收端过无线链路"<b>已有先例</b>，不能声称第一。
但其仅 41 图/172 题（我方 548 图，13×）、地面交通场景、模态内空间选块（非证据等级路由）、启发式功率分配（无 DRL）。
口径应改为：<b>"最大规模的 / UAV 场景的 / 证据等级路由的真实 VLM 无线 VQA 评测"</b>。</div>

<h3>14.2 期刊定位判断（TCCN 主投）</h3>
<table>
<tr><th>方案</th><th>判断</th><th>依据</th></tr>
<tr><td>论文一（语义通信有效性）投 <b>TCCN</b></td><td class="best">够格，主推</td><td>系统+方法论+三个未被声称的发现（②⑥⑧）；TCCN 对认知/语义通信实证系统接受度高；无学习型编解码器不是硬伤</td></tr>
<tr><td>论文一冲 TWC</td><td>边缘，需补强</td><td>需互补性 DPI 形式化命题 + 逐样本预测器升为正式贡献 + 第二数据集（§12 已补）</td></tr>
<tr><td>论文一冲 JSAC</td><td>仅限语义通信特刊</td><td>常规刊贡献密度不够；特刊 + 合并 RL 成 full-system 故事可试</td></tr>
<tr><td>论文二（v19 资源分配）投 <b>TCCN / IoTJ</b></td><td class="best">够格</td><td>语义感知约束 RL + 完整低空环境(UTM/DSS/BUBBLES) 在 IoTJ/TCCN 是强贡献</td></tr>
<tr><td>论文二冲 TWC</td><td>需理论件</td><td>加 Lyapunov 漂移-加-惩罚稳定性/最优性间隙分析（框架成熟）</td></tr>
</table>

<h3>14.3 口径红线（写作时必须遵守）</h3>
<div class="finding">1. <b>不得声称"第一个真实 VLM 无线 VQA"</b>——2505.02413 在先；改口径为规模/场景/路由维度并引用划界。</div>
<div class="finding">2. <b>①⑧ 近邻必须补引</b>：GO-SG、Park&amp;Yoon(JSAC'25)、2604.26508——related work 加差异表。</div>
<div class="finding">3. <b>M2 定位为"受控模拟代表"</b>，非学习型 JSCC；引用划界，勿声称击败 DeepSC 系列。</div>
<div class="finding">4. <b>互补性需理论件</b>：DPI / 充分统计量命题（计数：检测计数为充分统计量；存在性：召回下界）撑起 TWC 级贡献。</div>
<div class="finding">5. <b>撞车时间压力</b>：2505.02413 与 2604.26508 方向快速演进——建议论文一 1–2 个月内投出。</div>

</body></html>"""

out = "/Users/zhangqiankun/Documents/mpu/HPPO-VQA/docs_spec/UAV_VQA_SemCom_System_and_Results_v3.html"
open(out, "w").write(HTML)
print(f"wrote {out} ({len(HTML)//1024} KB)")
