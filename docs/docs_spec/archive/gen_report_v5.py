#!/usr/bin/env python3
"""Self-contained HTML report v5 (叙事重构版): Part A-F narrative reorder.

改造自 gen_report_v4.py。v4 的 15 节 / 9 图 / 全部实测数字全部保留，
按用户指定的叙事线重排为 Part A-F，并大幅详写：
  - Part C（BUBBLES 场景设计）
  - Part D（求解算法细节）
图来源 /tmp/report_figs/（9 张 PNG）。
"""
import base64
import os

FIGS = {}
for name in ["F1_acc_snr_3panel_final", "F2_cliff_final", "F3_latency_final",
             "F4_qtype_final", "F5_pareto_final", "F6_complementarity",
             "F7_mismatch_v3", "F8_token_budget_v3", "F9_separation_capacity"]:
    p = f"/tmp/report_figs/{name}.png"
    FIGS[name] = base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""


def img(name, cap):
    return ('<figure><img src="data:image/png;base64,%s" alt="%s">'
            '<figcaption>%s</figcaption></figure>') % (FIGS[name], name, cap)


HEAD = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAV-VQA 语义通信系统：设计与实验结果 v5 叙事重构版（2026-07-05）</title>
<style>
  :root { --ink:#24292f; --muted:#57606a; --line:#d0d7de; --bg:#f6f8fa; --acc:#0969da; --good:#1a7f37; --warn:#9a6700; --part:#8250df; }
  * { box-sizing:border-box; }
  body { font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; color:var(--ink);
         max-width:1080px; margin:0 auto; padding:24px 32px 80px; line-height:1.65; }
  h1 { font-size:26px; border-bottom:2px solid var(--line); padding-bottom:10px; }
  h2 { font-size:21px; margin-top:44px; border-bottom:1px solid var(--line); padding-bottom:6px; }
  h2.part { font-size:23px; color:#fff; background:linear-gradient(90deg,var(--part),#a371f7);
            border:none; border-radius:8px; padding:12px 18px; margin-top:56px; }
  h3 { font-size:17px; margin-top:28px; }
  .sub { color:var(--muted); font-size:14px; }
  nav { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:12px 18px; font-size:14px; }
  nav a { margin-right:12px; color:var(--acc); text-decoration:none; }
  nav .pt { color:var(--part); font-weight:700; }
  table { border-collapse:collapse; width:100%; font-size:13.5px; margin:14px 0; }
  th,td { border:1px solid var(--line); padding:6px 10px; text-align:left; vertical-align:top; }
  th { background:var(--bg); }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .best { color:var(--good); font-weight:600; }
  .bad  { color:#cf222e; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; margin:18px 0; }
  .card { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:12px 16px; }
  .card b { font-size:22px; display:block; }
  .card span { font-size:13px; color:var(--muted); }
  figure { margin:18px 0; text-align:center; }
  figure img { max-width:100%; border:1px solid var(--line); border-radius:6px; }
  figcaption { font-size:13px; color:var(--muted); margin-top:6px; }
  .flow { display:flex; flex-wrap:wrap; align-items:stretch; gap:6px; margin:16px 0; }
  .box { background:#fff; border:1.5px solid var(--acc); border-radius:8px; padding:8px 12px;
          font-size:13px; flex:1; min-width:120px; text-align:center; }
  .box small { color:var(--muted); display:block; }
  .arrow { align-self:center; color:var(--muted); font-size:18px; }
  .note { background:#fff8c5; border:1px solid #d4a72c66; border-radius:6px; padding:10px 14px; font-size:13.5px; }
  .finding { border-left:4px solid var(--good); background:var(--bg); padding:8px 14px; margin:10px 0; font-size:14px; }
  .eqmono { background:#fbfaff; border:1px solid #d8cffa; border-radius:6px; padding:10px 14px; margin:12px 0;
        font-family:"SF Mono",ui-monospace,Menlo,monospace; font-size:13px; overflow-x:auto; }
  .eq { font-family:Georgia,"Times New Roman",serif; font-style:italic; background:#f6f8fa;
        border-left:3px solid #0969da; padding:8px 14px; margin:8px 0; font-size:15px; overflow-x:auto; }
  .eq sub, .eq sup { font-style:normal; }
  .eq .lbl { color:var(--muted); font-style:normal; font-size:12.5px; margin-left:10px; }
  code { background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:12.5px; }
  ol.roman { font-size:14px; }
</style></head><body>

<h1>UAV-VQA 语义通信系统：设计与实验结果 <span style="color:var(--part)">v5</span>
<span style="font-size:15px;color:var(--muted)">（叙事重构版 · A2 详写版）</span></h1>
<p class="sub">2026-07-05 · 数据来自 160 服务器全量重算（与原 182 结果逐点复现校验通过）·
仓库 <code>qiankun-phd/uav-vqa-semantic-rl</code> 分支 <code>codex/lut-semantic-utility-upgrade</code>（对比实验）/
<code>main</code> + <code>feat/bubbles-scenario</code>（v19 资源分配 RL）·
<b>v5 相对 v4</b>：内容不删、数字不变，按 <b>Part A 系统设计 → B 语义通信实验 → C BUBBLES 场景设计 → D 求解算法细节 → E RL 实验结果 → F 收尾</b>
的叙事线重排；<b>Part C（场景设计）与 Part D（求解算法）为本版详写重点</b>，整合 BUBBLES D2.1 精读与 v19 双模型审查全部素材。</p>
"""

CARDS = """
<div class="cards">
<div class="card"><b>0.670</b><span>M4 自适应 @20dB Rician（测试集）— 高于 error-free 理想信道整图 0.615</span></div>
<div class="card"><b>265×</b><span>-5dB 下 s2 图像 vs s1 token 的信道用量差（3.72M vs 14k complex uses）</span></div>
<div class="card"><b>49×</b><span>-5dB Rayleigh 端到端延迟差：M1 4.32s vs token 0.089s</span></div>
<div class="card"><b>+7.7pt</b><span>threshold 题型 t=32 截断 vs 全量发送——"少发反而更准"</span></div>
<div class="card"><b>1.14×</b><span>-5dB Rician 实测时延注入：token 相对容量 1.060 vs 图像 0.930——证据选择换来的空域容量（20dB 塌缩到 1.01×）</span></div>
<div class="card"><b>3/3</b><span>跨三个 VLM 家族（Qwen2-VL / Qwen2.5-VL / SmolVLM-Idefics3）路由一致性——token 优势与接收端模型无关</span></div>
<div class="card"><b>&minus;0.168</b><span>慢速移动头因果降低冲突率（BUBBLES 主线 B1 消融，v2/v3 两轮稳定）——约束满足由结构承担的最强单项证据（Part D/E）</span></div>
</div>

<nav><b>目录</b>&nbsp;
<span class="pt">Part A 系统设计</span>
<a href="#A1">A1 管线</a> <a href="#A2">A2 证据等级</a> <a href="#A3">A3 统一计费</a> <a href="#A4">A4 任务与质量模型</a><br>
<span class="pt">Part B 语义通信实验</span>
<a href="#B1">B1 对比方法</a> <a href="#B2">B2 主结果</a> <a href="#B3">B3 互补性</a> <a href="#B4">B4 跨VLM</a>
<a href="#B5">B5 CSI失配</a> <a href="#B6">B6 逐样本预测器</a> <a href="#B7">B7 token预算</a> <a href="#B8">B8 跨数据集</a><br>
<span class="pt">Part C BUBBLES 场景设计</span>
<a href="#C1">C1 为何选 U-space</a> <a href="#C2">C2 场景要素</a> <a href="#C3">C3 我们的实现</a>
<a href="#C4">C4 安全桥 E7</a> <a href="#C5">C5 引用策略</a><br>
<span class="pt">Part D 求解算法细节</span>
<a href="#D1">D1 问题形式化</a> <a href="#D2">D2 网络与训练</a> <a href="#D3">D3 五层约束</a>
<a href="#D4">D4 正确性修复</a> <a href="#D5">D5 对偶消融定论</a><br>
<span class="pt">Part E RL 实验结果</span>
<a href="#E1">E1 双工况A/B</a> <a href="#E2">E2 承重件与权衡</a> <a href="#E3">E3 进行中：实验矩阵</a><br>
<span class="pt">Part F 收尾</span>
<a href="#F1">F1 发现清单</a> <a href="#F2">F2 新颖性与定位</a> <a href="#F3">F3 复现信息</a>
</nav>
"""

# ============================== PART A ==============================
PART_A = """
<h2 class="part" id="partA">Part A · 语义通信系统设计</h2>

<h3 id="A1">A1 场景与端到端管线</h3>
<p>无人机（UAV）巡查低空场景，边缘服务器上的用户就实时画面提出视觉问答（VQA）问题。
系统<b>不以重建图像为目标</b>，而是按问题类型自适应选择"证据等级"，只传输回答该问题所需的语义证据。</p>
<div class="flow">
<div class="box">UAV 相机<small>VisDrone 航拍图</small></div><div class="arrow">&rarr;</div>
<div class="box">YOLOv8n 检测器<small>VisDrone 微调 best.pt</small></div><div class="arrow">&rarr;</div>
<div class="box">证据选择器<small>问题类型 &rarr; s0/s1/s2（LUT / 零参数规则 / 逐样本预测器）</small></div><div class="arrow">&rarr;</div>
<div class="box">物理层<small>s1: LDPC r=1/2+BPSK 逐帧<br>s2: 速率自适应 JPEG+LDPC</small></div><div class="arrow">&rarr;</div>
<div class="box">无线信道<small>AWGN / Rayleigh / Rician K=6dB<br>SNR &minus;5&rarr;20 dB</small></div><div class="arrow">&rarr;</div>
<div class="box">边缘接收端<small>s1: 符号解码器<br>s2: Qwen2-VL-2B 答题</small></div><div class="arrow">&rarr;</div>
<div class="box">答案<small>指标: VQA 精度</small></div>
</div>

<h3 id="A2">A2 证据等级（服务等级）</h3>
<table>
<tr><th>等级</th><th>载荷</th><th>物理层</th><th>典型大小</th><th>接收端解码</th></tr>
<tr><td>s0 缓存</td><td>无传输，命中历史答案</td><td>&mdash;</td><td>0 B</td><td>直接返回（新鲜度约束）</td></tr>
<tr><td>s1 轻量 token</td><td>检测器输出（类别/框/置信度）逐帧封包</td><td>LDPC r=1/2 &times; BPSK（0.5 bit/use），帧丢失概率=FER：50% 整帧丢弃 / 50% 载荷乱码（类别&rarr;unknown+框抖动）</td><td>~0.7&ndash;1.2 KB</td><td>符号解码器（计数/比较/阈值）或 VLM（presence）</td></tr>
<tr><td>s2 全图</td><td>JPEG 码流</td><td>速率自适应：按瞬时可达谱效率适配 JPEG 质量（遍历 ergodic SE 计费）</td><td>~110 KB（均值）</td><td>Qwen2-VL-2B 视觉问答</td></tr>
<tr><td>s3 ROI</td><td>检测框引导的裁剪图</td><td>同 s2</td><td>介于两者</td><td>Qwen2-VL（评测框架支持，未进对比主表）</td></tr>
</table>
<p class="sub">上表为总览；下面把每个等级展开成完整实现细节——先一句"这级在干什么"，再讲发送端 / 信道 / 接收端 / 计费与时延。</p>

<h4 id="A2-s0">A2.1 &nbsp;s0 缓存应答 &mdash; 命中即零传输、零暴露</h4>
<p><b>一句话：这级根本不发东西。</b>系统维护一个语义缓存，把过去回答过的问题连同答案存起来；
若当前问题能在缓存里命中，就直接返回历史答案，<b>零信道传输、零 operational intent</b>
（在 BUBBLES 安全场景下即"零冲突暴露"——不发起飞行/通信动作就不会新增碰撞风险），<b>时延 &asymp; 0</b>。</p>
<p><b>缓存键与新鲜度。</b>缓存以 <code>(区域, 问题类型)</code> 为键存历史答案；每个条目随时间老化，
经历 <b>fresh &rarr; stale &rarr; expired</b> 三态，过期条目不可用（必须重新走 s1/s2 采集新证据）。
这保证"零传输"不会以牺牲时效为代价：只有仍新鲜的答案才允许被复用。</p>
<p><b>缓存精度模型。</b>命中并不等于满分——最好情况也只有查表（LUT）标定出的精度，且随命中质量折减：</p>
<div class="eq">A<sub>s0</sub> = A<sub>LUT</sub> &middot; (0.85 + 0.15 &middot; p<sub>hit</sub>)<span class="lbl">p<sub>hit</sub> = 命中置信度 &isin; [0,1]</span></div>
<p>其含义：即便命中置信度 p<sub>hit</sub>=1，精度上限也只是 A<sub>LUT</sub>；命中质量越差（p<sub>hit</sub> 越小），
精度按 0.85&rarr;1.0 的系数线性折减。<b>风险与防护：</b>当语义缺口（gap）过大时，缓存复用会"塌缩"——
把不该复用的旧答案强行套到新画面上。算法层用 <b>cache-override 投影</b>强制升级证据等级（禁止在大 gap 下停留于 s0），
对应 RL 奖励里的 <b>&minus;16&middot;gap 塌缩罚</b>，从价值面把"缓存塌缩"这条捷径堵死。</p>
<p><i>一句话总结：命中时零成本、零暴露、零时延，但精度天花板只到 LUT——它是省资源的捷径，不是提精度的手段，且必须用塌缩罚看住。</i></p>

<h4 id="A2-s1">A2.2 &nbsp;s1 轻量语义 token &mdash; 几百字节的符号证据</h4>
<p><b>一句话：不发图，只发"检测器看到了什么"。</b>发送端用 <b>YOLOv8n</b>（VisDrone 微调）在原图上检测，
把每个检测框封装成一帧（类别 / 边框 / 置信度），再逐帧走物理层送到接收端。</p>
<p><b>发送端 &rarr; 物理层。</b>每帧经 <b>LDPC 码率 R<sub>c</sub>=1/2 + BPSK（M=2）</b>逐帧独立传输。
信道用确定性哈希实现（可复现）：逐帧的帧错误概率 FER(&gamma;, ch) 取自 LDPC-BPSK 链路模型；
一旦该帧判为丢失，则二选一（各 50%）——<b>帧头损坏 &rarr; 整帧丢弃</b>；<b>载荷损坏 &rarr; 类别置 unknown + 边框抖动</b>
（坐标乘 U[0.7,1.3] 的随机缩放）。因此单帧存活概率为：</p>
<div class="eq">P(帧存活) = 1 &minus; FER(&gamma;, ch)</div>
<p><b>信道用量（谱效率 0.5 bit/复用）。</b>每字节 8 bit，经 R<sub>c</sub>=1/2、BPSK 后每复用承载 0.5 bit，故：</p>
<div class="eq">N<sub>s1</sub> = 8 &middot; B<sub>tok</sub> / (R<sub>c</sub> &middot; log<sub>2</sub>M) = 16 &middot; B<sub>tok</sub><span class="lbl">R<sub>c</sub>=1/2，BPSK M=2</span></div>
<p>载荷以帧头为主导：t=1 个 token 约 <b>696 B</b>、满发约 <b>1.2 KB</b> &rarr; N<sub>s1</sub> &asymp; <b>1.1 万 ~ 1.9 万复用</b>；
对比 s2 在低 SNR 时高达 <b>372 万复用</b>，相差约 <b>265&times;</b>。</p>
<p><b>接收端解码分两路。</b>（i）<b>符号解码器</b>处理 counting / comparison / co_presence / threshold 题型——
直接对幸存帧做计数/比较/阈值判断，<b>不需要 VLM</b>，因此 <b>token 路的精度与接收端 VLM 无关</b>（这也是 B4 跨-VLM 一致性的根因）；
（ii）<b>VLM 提示词路</b>处理 presence 题型——把 token 文本拼进 Qwen2-VL 的提示词里辅助判断。</p>
<p><b>计数标定（纠正信道丢帧的系统性低估）。</b>丢帧会让接收端计数偏低；对校准集按 <code>(目标类, 信道档)</code>
拟合一个比例系数 &rho;，推理时按其放大计数：</p>
<div class="eq">&rho; = &Sigma; 真值计数 / &Sigma; 收到计数 &nbsp;&nbsp;&rarr;&nbsp;&nbsp; &ccirc; = round(&rho; &middot; c<sub>rx</sub>)<span class="lbl">仅当 c<sub>rx</sub> &ge; c<sub>min</sub>=3 且标定在校准集不劣于原始计数才启用</span></div>
<p>两个护栏保证标定只在可靠时生效：收到帧太少（c<sub>rx</sub> &lt; 3）时不外推，且若标定在校准集上反而更差就回退到原始计数。</p>
<p><b>可变 token 预算（嵌套前缀截断）。</b>按检测置信度降序只保留 top-t 帧再发送——<b>t 就是"语义速率"旋钮</b>。
实验发现适度截断兼做假阳性过滤：threshold 题型取 <b>t=32 比满发高 7.7 分</b>（"少发反而更准"）。</p>
<p><b>端到端时延。</b>-5dB Rayleigh 全程仅 <b>0.089 s</b>（上传 0.014 + 发端处理 0.037 + 解码 0.037），
比传图（s2）低约两个数量级。</p>
<p><i>一句话总结：用几百字节的符号证据回答一半以上的问题——精度与接收端模型无关，时延和带宽比传图低两个数量级。</i></p>

<h4 id="A2-s2">A2.3 &nbsp;s2 全图 &mdash; 速率自适应 JPEG，把悬崖换成平滑下降</h4>
<p><b>一句话：真发图，但"装得下多少发多少"。</b>发送端按瞬时信道可达速率适配 JPEG 质量因子，
低 SNR 时自动降质、高 SNR 时提质，用速率自适应把数字通信的"精度悬崖"抹平成"质量平滑下降"。</p>
<p><b>可达速率用遍历谱效率。</b>可发的比特预算由信道的遍历（ergodic）谱效率决定：</p>
<div class="eq">SE(&gamma;&#772;) = E<sub>h</sub>[ log<sub>2</sub>(1 + &gamma;&#772; &middot; |h|<sup>2</sup>) ]<span class="lbl">bit/复用；AWGN 时 |h|=1；Rayleigh/Rician 对 h 取期望</span></div>
<p>该期望对衰落 h 数值积分预计算成表（如 rician -5dB &rarr; <b>0.383</b>、20dB &rarr; <b>6.354</b> bit/复用）。</p>
<p><b>信道用量与上传时延</b>（带宽 B<sub>w</sub>=1 MHz，1 复用 = 1 &micro;s）：</p>
<div class="eq">N<sub>s2</sub> = 8 &middot; B<sub>JPEG</sub> / SE(&gamma;&#772;) &nbsp;&nbsp;&nbsp; T<sub>up</sub> = N<sub>s2</sub> / B<sub>w</sub></div>
<p>均值 B<sub>JPEG</sub> &asymp; <b>110 KB</b>；低 SNR 下 JPEG 质量自动下调但有下限（min_payload &asymp; 1.5 KB，防彻底断流）。
-5dB Rayleigh 实测 <b>N<sub>s2</sub> = 372 万复用、T<sub>up</sub> = 3.72 s</b>（= 12.4 个 300k-复用时隙）。</p>
<p><b>接收端。</b>Qwen2-VL-2B 直接对退化后的图像做 VQA（推理 <b>0.60 s</b>）&rarr; 全程 <b>4.32 s</b>，是 s1 的 <b>49&times;</b>。</p>
<p><b>为什么要速率自适应（对照 M0_naive 固定速率数字）。</b>固定速率不自适应 &rarr; 低 SNR 大量丢包 &rarr; 精度悬崖
（-5dB 跌至 <b>0.37</b>）；速率自适应则把这道悬崖换成"质量平滑下降"，代价是带宽/时延随信道恶化而抬升。</p>
<p><b>统一计费口径（模拟/数字公平比较的关键）。</b>所有方法一律按复信道用量计费：</p>
<div class="eq">CBR = N / (3HW)<span class="lbl">源维度 3HW &asymp; 288 万实数</span></div>
<p><i>一句话总结：以带宽和时延为代价换取信道恶化时的精度平滑——速率自适应是把数字通信悬崖抹平的关键工程手段。</i></p>

<h4 id="A2-s3">A2.4 &nbsp;s3 ROI 裁剪 &mdash; 介于 token 与全图之间的中间档</h4>
<p><b>一句话：只发"目标附近那块图"。</b>用检测框引导裁剪出目标区域（对目标框扩边取窗），
走与 s2 完全相同的速率自适应链路，载荷大小介于 s1（几百字节）与 s2（约 110 KB）之间。</p>
<p>评测框架已支持（<code>_build_roi_image</code> + VLM 提示注明 detector-guided crop），但<b>未进主对比表</b>——
主实验刻意聚焦 s1 token 与 s2 全图这对"二元权衡"，把叙事收窄。s3 在 RL 层保留为<b>可选动作</b>，
供资源分配策略在中间工况调用。</p>
<p><i>一句话总结：一个已实现、暂未主打的中间档——需要局部细节又付不起全图带宽时的备选，留给 RL 策略按需启用。</i></p>

<h4 id="A2-qm">A2.5 &nbsp;这些等级怎么被选中：质量模型驱动</h4>
<p>服务选择由<b>质量模型</b>驱动：LUT 把每个格子 <code>(题型 &times; 服务 &times; SNR &times; …)</code> 的历史正确率记为
<b>Wilson 置信区间</b>，选择时不用点估计 p&#770;，而用其<b>下置信界 LCB</b>，防止稀疏格子（样本少）因偶然高分被虚高选中：</p>
<div class="eq">LCB = ( p&#770; + z<sup>2</sup>/2n &minus; z &middot; &radic;( p&#770;(1&minus;p&#770;)/n + z<sup>2</sup>/4n<sup>2</sup> ) ) / (1 + z<sup>2</sup>/n)<span class="lbl">z=1.96（95%）；n = 该格样本数</span></div>
<p>样本 n 越小，惩罚项越大、LCB 被压得越低——迫使策略在证据不足时保守。这套选择器经历三代演进：
<b>多维 LUT &rarr; 零参数语义规则 &rarr; 逐样本预测器</b>，其机制、等价性与升级路径见 <a href="#A4">A4</a>。</p>

<h3 id="A3">A3 统一带宽计费（模拟/数字公平对比的关键）</h3>
<p>所有方法统一按<b>复信道用量（complex channel uses）/查询</b>计费：
s2 = bits &divide; ergodic SE(snr, 信道)；s1 与固定速率数字 = bits &divide; 0.5；M2 模拟 = 原生 uses（1 时隙 = B&middot;&tau; = 1MHz&times;0.3s = 300k uses）；
CBR = uses &divide; 平均源维度 2.88M。关键量级（-5dB Rayleigh）：M1 图像 3.72M uses（12.4 时隙）、M0_naive 恒 5.28M、M2 恒 300k、<b>token 仅 ~14k</b>。</p>

<h3 id="A4">A4 任务、数据与语义质量模型三代演进</h3>
<p>VisDrone-DET 548 张真实航拍图，5 种问题类型（presence 存在性 / counting 计数 / comparison 数量比较 /
co_presence 联合存在 / threshold 数量阈值），任务生成完全确定性（538 条 comparison 严格 269/269 平衡、1096 条 extra）。
按 image_id 的 crc32 做 20% 留出测试集，所有报告数字均为测试集。</p>
<table>
<tr><th>代</th><th>机制</th><th>选择依据</th><th>作用层</th></tr>
<tr><td>第一代 多维 LUT</td><td>(题型, SNR, view, freshness, risk) &rarr; 服务，Wilson-LCB 下界选择</td><td>数据标定的保守下界</td><td>单查询服务选择 + RL 奖励接口</td></tr>
<tr><td>第二代 零参数语义规则</td><td>符号推理&rarr;token / 感知推理&rarr;image，无任何标定</td><td>推理结构（充分统计量）</td><td>与 LCB 选择器在测试集上完全等价（Part B3）</td></tr>
<tr><td>第三代 逐样本参数化预测器</td><td>P(correct | 发送端特征, 服务) 逐样本 argmax</td><td>发送端可得特征（题型/目标类/检测计数/SNR）</td><td>更准且 payload 减半（Part B6），RL LUT 接口的直接升级候选</td></tr>
</table>
"""

# ============================== PART B ==============================
PART_B = ("""
<h2 class="part" id="partB">Part B · 语义通信实验</h2>

<h3 id="B1">B1 对比方法（标准四件套扩展为七方）</h3>
<table>
<tr><th>方法</th><th>含义</th><th>代表的范式</th></tr>
<tr><td>M0_errorfree</td><td>理想无差错信道传原图</td><td>性能参照（非上界——见发现 &#9313;）</td></tr>
<tr><td>M0_naive</td><td>固定速率 LDPC 数字传图</td><td>传统数字链路的悬崖效应</td></tr>
<tr><td>M1_image</td><td>速率自适应 JPEG+LDPC 传图（=固定 s2）</td><td>传统分离信源信道编码 SSCC</td></tr>
<tr><td>M2_analog</td><td>非编码模拟图像传输（JSCC-lite）</td><td>DeepSC 式模拟语义传输的受控代表</td></tr>
<tr><td>M3_token</td><td>固定 s1 检测 token</td><td>GO-SG 式目标导向符号传输</td></tr>
<tr><td><b>M4_adaptive</b></td><td>按 (题型, SNR) Wilson-LCB 选服务（训练集学、测试集报）</td><td><b>本系统</b></td></tr>
<tr><td>M5_oracle</td><td>逐任务事后最优服务</td><td>服务选择上界</td></tr>
</table>

<h3 id="B2">B2 主结果：精度-SNR、悬崖、延迟、Pareto</h3>
"""
+ img("F1_acc_snr_3panel_final", "F1 三信道精度-vs-SNR（测试集）。M4 在所有信道所有 SNR 上是最优非 oracle 方法。")
+ """
<table>
<tr><th>信道</th><th class="num">M4 (-5&rarr;20dB)</th><th class="num">M3 token</th><th class="num">M1 图像</th><th class="num">M0_naive @-5dB</th><th class="num">M5 oracle @20dB</th></tr>
<tr><td>AWGN</td><td class="num best">0.628 &rarr; 0.668</td><td class="num">0.601（平坦）</td><td class="num">0.556 &rarr; 0.611</td><td class="num bad">0.369</td><td class="num">0.720</td></tr>
<tr><td>Rayleigh</td><td class="num best">0.609 &rarr; 0.668</td><td class="num">0.603 &rarr; 0.599</td><td class="num">0.537 &rarr; 0.614</td><td class="num bad">0.409</td><td class="num">0.720</td></tr>
<tr><td>Rician K=6dB</td><td class="num best">0.625 &rarr; 0.670</td><td class="num">0.598 &rarr; 0.601</td><td class="num">0.556 &rarr; 0.612</td><td class="num bad">0.369</td><td class="num">0.721</td></tr>
</table>
<p>参照线：<b>M0_errorfree = 0.615</b>（理想信道整图）、M2 模拟 Rician 0.479 &rarr; 0.595（平滑无悬崖）。</p>
"""
+ img("F2_cliff_final", "F2 悬崖效应（Rayleigh）：固定速率数字 M0_naive 在低 SNR 跌至 0.37-0.41；M2 模拟与 M4 自适应平滑退化。")
+ img("F3_latency_final", "F3 端到端延迟分解（Rayleigh，对数轴）：-5dB 下 M1 上传 3.72s + 推理 0.60s = 4.32s；token 全程 0.089s（49&times;）；M0_naive 5.87s。")
+ img("F4_qtype_final", "F4 分题型精度（Rician @5dB）。")
+ img("F5_pareto_final", "F5 目标导向效率 Pareto：精度 vs 复信道用量（统一模拟/数字轴）。M4 以 ~14k-110K 量级用量达到 0.67，M1/M0_naive 需要 3.7-5.3M。")
+ """
<h3 id="B3">B3 头条发现：证据-问题互补性</h3>
"""
+ img("F6_complementarity", "F6 token 增益 &Delta;=acc(token)&minus;acc(image) 按题型（3 信道池化，测试集）。")
+ """
<table>
<tr><th>题型</th><th>推理类别</th><th class="num">token</th><th class="num">image</th><th class="num">&Delta; = t &minus; i</th></tr>
<tr><td>counting</td><td>符号</td><td class="num">0.453</td><td class="num">0.279</td><td class="num best">+0.174</td></tr>
<tr><td>comparison</td><td>符号</td><td class="num">0.846</td><td class="num">0.696</td><td class="num best">+0.151</td></tr>
<tr><td>co_presence</td><td>符号</td><td class="num">0.663</td><td class="num">0.545</td><td class="num best">+0.119</td></tr>
<tr><td>threshold</td><td>符号（边界）</td><td class="num">0.596</td><td class="num">0.611</td><td class="num">&minus;0.014</td></tr>
<tr><td>presence</td><td>感知</td><td class="num">0.678</td><td class="num">0.765</td><td class="num bad">&minus;0.087</td></tr>
</table>
<p><b>零参数语义规则</b>（符号&rarr;token，感知&rarr;image）与<b>数据标定 LCB 选择器</b>在测试集上完全一致：
双双 <b>0.6725</b>（n=5616；固定 token 0.6346 / 固定图像 0.6043 / oracle 0.7472）。
机理：token 无损保留离散计数的充分统计量；图像保留存在性召回。消融证明 question_type 是 LUT 中唯一承重维度
（snr/view/freshness/risk/LCB 对服务选择增益&asymp;0——这些维度的用武之地在跨时间/跨用户的资源调度层，见 Part D/E）。</p>

<h3 id="B4">B4 跨 VLM 稳健性（三家族三模型，Rician）</h3>
<p>把跨 VLM 验证从"两个 Qwen 同家族模型"扩到<b>三家族三模型</b>：Qwen2-VL-2B、Qwen2.5-VL-3B、
以及<b>非 Qwen 家族的 SmolVLM-2B</b>（视觉塔 = Idefics3 / SigLIP，语言塔 = SmolLM2；与 Qwen 架构完全独立）。
下表每格为该模型的 <code>token / image</code> 测试精度，路由方向取 argmax。</p>
<table>
<tr><th>题型</th><th>推理类别</th><th class="num">Qwen2-VL-2B<br>token/image</th><th class="num">Qwen2.5-VL-3B<br>token/image</th><th class="num">SmolVLM-2B<br>token/image</th><th>三家族路由</th></tr>
<tr><td>presence</td><td>感知</td><td class="num">0.678 / 0.761</td><td class="num">0.712 / 0.721</td><td class="num">0.712 / 0.724</td><td class="best">一致 &rarr; image</td></tr>
<tr><td>counting</td><td>符号</td><td class="num">0.450 / 0.280</td><td class="num">0.404 / 0.375</td><td class="num">0.404 / 0.391</td><td class="best">一致 &rarr; token</td></tr>
<tr><td>comparison</td><td>符号</td><td class="num">0.846 / 0.710</td><td class="num">0.846 / 0.787</td><td class="num">0.846 / 0.598</td><td class="best">一致 &rarr; token</td></tr>
<tr><td>co_presence</td><td>符号</td><td class="num">0.663 / 0.559</td><td class="num">0.663 / 0.553</td><td class="num">0.663 / 0.529</td><td class="best">一致 &rarr; token</td></tr>
<tr><td>threshold</td><td>符号（边界）</td><td class="num">0.596 / 0.611</td><td class="num">0.596 / 0.567</td><td class="num">0.596 / 0.622</td><td>边界（三模型均 |&Delta;|最小）</td></tr>
</table>
<div class="finding"><b>三条跨家族结论</b>：
（1）<b>4/5 题型路由跨三家族完全一致</b>——presence&rarr;image、其余四个符号题型&rarr;token；唯一分歧的 threshold 恰是互补性最弱（|&Delta;|最小）题型，<b>互补性强度预测路由稳定性</b>。
（2）<b>token 列与 VLM 无关</b>：三模型的 token 精度逐格相同（comparison 0.846、co_presence 0.663、counting 0.404&hellip;）——因为 token 由确定性<b>符号解码器</b>处理，根本不经过 VLM。
（3）<b>弱 VLM 放大 token 优势</b>：SmolVLM 读退化图像最弱（comparison image 0.598 vs Qwen 0.71&ndash;0.79、co_presence 0.529），token 相对图像的领先反而更大——接收端越弱，证据选择的价值越高。</div>

<h3 id="B5">B5 CSI 失配矩阵：选择器不需要精确 CSI</h3>
"""
+ img("F7_mismatch_v3", "F7 策略假定 SNR &times; 真实 SNR 的测试精度热力图（物理链路保持按真实 SNR 自适应，隔离调度层的 CSI 误差影响）。")
+ """
<p>三信道全部近乎平坦：<b>最差失配格（0.641&ndash;0.653）恰好等于匹配 CSI 时的最低对角值</b>，任意 CSI 错误都不产生断崖。
问题驱动的路由在选择时刻无需信道估计——参考工作（如 ADJSCC）评估时始终馈入真实 SNR，从未度量此轴。</p>

<h3 id="B6">B6 逐样本参数化预测器：更准且 payload 减半</h3>
<p>用<b>发送端可得特征</b>（题型/目标类/原始检测计数/SNR/任务元数据）训练逐服务正确率预测器
P(correct | x, s)（逻辑回归，OracleNet 配方 MSE&rarr;BCE），逐样本 argmax 选服务，评估为对已有日志的离线重选择：</p>
<table>
<tr><th>信道</th><th class="num">M3 token</th><th class="num">M1 image</th><th class="num">M4 LUT</th><th class="num">逐样本 ML</th><th class="num">M5 oracle</th><th class="num">平均 payload</th></tr>
<tr><td>AWGN</td><td class="num">0.6346</td><td class="num">0.6043</td><td class="num">0.6747</td><td class="num best">0.6832</td><td class="num">0.7472</td><td class="num">58 KB（LUT 111 KB）</td></tr>
<tr><td>Rayleigh</td><td class="num">0.6339</td><td class="num">0.6015</td><td class="num">0.6690</td><td class="num best">0.6804</td><td class="num">0.7455</td><td class="num">56 KB（107 KB）</td></tr>
<tr><td>Rician</td><td class="num">0.6339</td><td class="num">0.6063</td><td class="num">0.6727</td><td class="num best">0.6798</td><td class="num">0.7450</td><td class="num">60 KB（110 KB）</td></tr>
</table>
<p>三信道全胜 LUT +0.7~1.1pt（追回 oracle 差距的 10&ndash;15%），且 <b>payload 几乎减半</b>——预测器学会"发送端检测证据已充分时，presence 也走 token"。
增益集中于 presence（+1.5~2.4pt）与 counting（+1.2pt），正是逐样本信息应当起作用的位置。</p>

<h3 id="B7">B7 可变 token 预算：嵌套 top-t 截断</h3>
"""
+ img("F8_token_budget_v3", "F8 token 预算扫描（符号解码题型，测试集；每档 t 重新标定计数，t=full 与主实验逐位一致）。")
+ """
<div class="finding"><b>发现 A（预算敏感度分化）</b>：comparison t=3 即近饱和（0.817 / 满发 0.846）；counting 需 t=48 才追平（0.453）——
需求差一个数量级，构成互补性的第二证据轴。</div>
<div class="finding"><b>发现 B（少发反而更准）</b>：threshold t=32 达 0.673 vs 满发 0.596（<b>+7.7pt</b>）、co_presence +3.9pt，三信道完全一致。
置信度排序截断兼做假阳性过滤器——对"至少 N 个 / 两者都有"这类 yes 偏置问题，精度比召回更重要。</div>
<div class="note">诚实标注：s1 载荷为帧头主导（t=1 约 696B、满发约 1.2KB），截断的带宽节省有限（~30%）；
该机制的价值在<b>精度过滤与预算敏感度分化</b>，而非省带宽。</div>

<h3 id="B8">B8 跨数据集验证（DroneVehicle）</h3>
<p>为回应"单数据集 548 图"的审稿攻击点（见 F2 薄弱点 #3），在<b>第二个 UAV 数据集 DroneVehicle</b> 上完整复跑管线。
接入是<b>零代码</b>的：490 张图 / 5 类车（car、truck、bus、van、freight car；其中 <code>freight car</code> 两种拼写变体统一并入 <b>truck</b>），
标注为 <b>VOC-XML</b>，通过<b>符号链接</b>把 DroneVehicle 目录挂进现有 VisDrone 数据路径即可，检测器/任务生成器/证据选择器全部零改动。</p>
<h4>主对比（DroneVehicle，三信道）</h4>
<table>
<tr><th>方法</th><th class="num">AWGN</th><th class="num">Rayleigh</th><th class="num">Rician</th><th>范式</th></tr>
<tr><td><b>M4_adaptive</b></td><td class="num best">0.735</td><td class="num best">0.732</td><td class="num best">0.725</td><td>本系统（问题驱动路由）</td></tr>
<tr><td>M3_token</td><td class="num" colspan="3" style="text-align:center">0.724（三信道近平坦）</td><td>固定检测 token</td></tr>
<tr><td>M1_image</td><td class="num" colspan="3" style="text-align:center">0.642</td><td>速率自适应传图 SSCC</td></tr>
<tr><td>M5_oracle</td><td class="num" colspan="3" style="text-align:center">0.808</td><td>服务选择上界</td></tr>
</table>
<p>排序 <b>M4 (0.735/0.732/0.725) &gt; M3 (0.724) &gt; M1 (0.642)</b>，oracle 0.808——与 VisDrone 主结果同构。</p>
<h4>互补性对照 + 规则反超标定</h4>
<table>
<tr><th>题型</th><th>推理类别</th><th class="num">DroneVehicle &Delta;</th><th class="num">VisDrone &Delta;（B3）</th><th>跨数据集读解</th></tr>
<tr><td>counting</td><td>符号</td><td class="num best">+0.211</td><td class="num">+0.174</td><td>token 优势更强</td></tr>
<tr><td>comparison</td><td>符号</td><td class="num best">+0.098</td><td class="num">+0.151</td><td>方向一致</td></tr>
<tr><td>co_presence</td><td>符号</td><td class="num best">+0.091</td><td class="num">+0.119</td><td>方向一致</td></tr>
<tr><td>threshold</td><td>符号（边界）</td><td class="num best">+0.077</td><td class="num">&minus;0.014</td><td><b>脱离边界</b>：明确偏 token</td></tr>
<tr><td>presence</td><td>感知</td><td class="num">&minus;0.014</td><td class="num bad">&minus;0.087</td><td><b>边际变薄</b>：图像优势缩小但仍为负</td></tr>
</table>
<div class="finding"><b>零参数语义规则 0.7293 &gt; 逐数据集标定 LCB 0.7246</b>。
在 DroneVehicle 上，无任何该数据集标定的<b>零参数规则</b>比用其自身数据重新标定的 Wilson-LCB 还高 0.47pt——
数据标定会过拟合本数据集采样噪声，基于推理结构的规则跨域<b>更泛化</b>。把"规则 = 标定"（VisDrone 等价性）升级为"规则 &ge; 标定"（跨数据集规则更优）。</div>
""")

# ============================== PART C ==============================
PART_C = ("""
<h2 class="part" id="partC">Part C · BUBBLES 场景设计（详写）</h2>
<p class="sub">本章整合 <code>docs_spec/BUBBLES_D2.1_Analysis.md</code>（Opus 子代理逐页精读 SESAR JU BUBBLES D2.1 Ed 04.00.00
全部 156 页产出）。目的：把 v19 资源分配环境（Part D）的每一个场景参数锚定到<b>权威可引出处</b>，
并搭起"语义通信质量 &rarr; 飞行安全约束"的理论桥。</p>

<h3 id="C1">C1 为什么选 U-space / BUBBLES（单场景主线决策）</h3>
<p>论文二（v19 资源分配）此前用的是<b>自设合成冲突场景</b>：空域大小、UAV 运动参数、任务到达率、冲突阈值都是"拍脑袋"取值。
这在审稿时是明确的攻击面——"这些数字从哪来？为什么冲突约束阈值是这个数？"无法回答。因此本版做出一个明确的<b>单场景主线决策</b>：
把整套场景要素<b>逐项替换为 SESAR JU BUBBLES D2.1 有明文出处的数值</b>，让每一个参数都能指向一页原文。</p>
<table>
<tr><th>决策维度</th><th>自设场景（旧）</th><th>BUBBLES 主线（新）</th><th>收益</th></tr>
<tr><td>空域几何</td><td>任取 N&times;N 网格</td><td>5&times;5 km、3 飞行层、20&ndash;120 m（Appendix G）</td><td>可引、量级自洽</td></tr>
<tr><td>UAV 运动包线</td><td>自设速度/尺寸</td><td>Appendix B 性能包线表（10 类交通分类）</td><td>可引、按 SAIL 分级</td></tr>
<tr><td>任务到达率</td><td>自设 Poisson 率</td><td>G-13 需求曲线（日均并发 8.17、峰值 20）</td><td>可引、双工况有据</td></tr>
<tr><td>冲突判据</td><td>单一距离阈值</td><td>CPA 双判据 + Block-4 分隔链</td><td>可引、术语对齐 SESAR</td></tr>
<tr><td>约束阈值锚</td><td>自设 conflict 上限</td><td>TLS_MAC = 2.5e-7 FAT/FH</td><td>可引、安全预算有据</td></tr>
</table>
<div class="note">BUBBLES D2.1 是 U-space <b>间隔管理服务（SMS）ConOps</b> 交付物（Grant 893206，PU 公开），
其 §4.5 明确定义了 <b>CSPM&rarr;DSA 闭环</b>（通信/监视性能监视 &rarr; 动态分隔调整）——
这恰好是"把语义通信质量接进飞行安全约束"的权威服务框架，是我们把语义通信写成飞行安全变量（C4）的制度依据。</div>

<h3 id="C2">C2 场景要素逐项详写</h3>

<h4>C2.1 交通分类与性能包线（Appendix B, p99 &rarr; Table B-2）</h4>
<p>D2.1 Appendix B 给出 <b>10 类交通分类</b>及每类的性能包线（巡航 5&ndash;25 m/s、爬升 3&ndash;5 m/s、尺寸 0.5&ndash;5 m）。
我们的主线取<b>中段 SAIL III&ndash;IV</b> 档位（小型旋翼巡查 UAS），落到环境里的 <code>Table B-2</code> 冲突包线为：</p>
<table>
<tr><th>包线参数</th><th class="num">取值</th><th>出处/理由</th></tr>
<tr><td>巡航速度 Vc</td><td class="num">14 m/s</td><td>Appendix B 中段（5&ndash;25 区间）</td></tr>
<tr><td>爬升率</td><td class="num">5 m/s</td><td>Appendix B 上限（3&ndash;5）</td></tr>
<tr><td>下降率</td><td class="num">4 m/s</td><td>Appendix B 区间内</td></tr>
<tr><td>机体特征尺寸（水平/垂直）</td><td class="num">2.0 / 1.0 m</td><td>Appendix B（0.5&ndash;5 区间）</td></tr>
<tr><td>SAIL 等级</td><td class="num">III&ndash;IV</td><td>JARUS SORA 分级；决定分隔严格度</td></tr>
</table>

<h4>C2.2 空域几何（Appendix G, p123&ndash;129）</h4>
<p>数值算例给出：<b>5&times;5 km</b> 作业空域、<b>3 个飞行层</b>、高度带 <b>20&ndash;120 m</b>。
这套几何直接决定了 v19 环境的 UAV 位置状态归一化尺度与 A2G 路损的距离量级，使"环境尺度"不再是自设量。</p>

<h4>C2.3 G-13 需求曲线（任务到达率的出处）</h4>
<p>Table G-13 给出该空域的运营需求：<b>日均并发 8.17 架</b>、<b>峰值 20 架</b>、<b>日操作 784 次</b>、<b>平均操作时长 15 min</b>。
我们据此把 <code>bubbles_daily</code> 到达过程标定为两档预设：</p>
<table>
<tr><th>工况</th><th>并发/到达</th><th>对应 G-13</th><th>用途</th></tr>
<tr><td><b>utm_conflict（峰值）</b></td><td>峰值 20 架并发，到达密集</td><td>G-13 峰值列</td><td>压力测试：冲突约束是否被真正激活</td></tr>
<tr><td><b>nominal（平峰）</b></td><td>日均 8.17 架并发</td><td>G-13 日均列</td><td>标称评估：默认工况</td></tr>
</table>
<p>操作时长 15 min &rarr; 单架任务在空域内的驻留步数与语义缓存新鲜度窗口（fresh&rarr;stale&rarr;expired）由此校准。</p>

<h4>C2.4 分隔最小值链式公式（§3.3.4 Block 4, p60&ndash;61）</h4>
<p>严重度阶梯（p103 Fig 46）：<b>MAC &rarr; NMAC &rarr; IC &rarr; SL &rarr; TC</b>，每级一个 barrier。
战术分隔最小值（tactical contact 距离 d_TC）由下式逐级叠加得到——从物理碰撞 d_NMAC 出发，逐项加入各类误差与时间窗位移，直到战术接触距离：</p>
<div class="eqmono">
d_TC = d_NMAC + TSE + Vc&middot;(T1+T2) + SSE + Vc&middot;T_res + Vc&middot;(T3+T4)
</div>
<p>其中 d_NMAC = 近碰撞距离（下游 barrier）；TSE = 总系统误差；SSE = 监视系统误差；
T1/T2/T3 = 检测/决策/机动响应时间窗；<b>T_res</b> = 分隔恢复时间。链条从 d_NMAC 逐级向上：
<code>d_NMAC &rarr; d_IC &rarr; d_SL &rarr; d_TC</code> 每一级对应阶梯上一个 barrier 的时间/误差余量。
关键项是 <b>T4</b>：</p>
<div class="finding"><b>T4 三正态分解（通信延迟在环）</b>：
T4 = <b>tracking 延迟 + separation 通信延迟 + pilot 执行延迟</b> 三个正态分布之和
（Table G-2 算例：<b>通信延迟均值 1.8 s / &sigma; 1.0</b>，T4 总均值 <b>7.82 s / &sigma; 4.24</b>）。
因果链完整成文（p39 性能&rarr;分隔原则）：<b>通信时延&uarr; &rarr; T4&uarr; &rarr; Vc&middot;T4 项&uarr; &rarr; d_TC&uarr; &rarr; 空域容量&darr; &rarr; 冲突约束更易违反</b>。
这正是"语义通信质量映射到飞行安全约束"的机制核心——通信延迟不是外生噪声，而是分隔链里一个可加的正态分量。</div>

<h4>C2.5 TLS 锚定与 conflict limit 推导（p40 / p110 / p112）</h4>
<p>目标安全等级（Target Level of Safety）：<b>总 TLS = 1e-6 FAT/FH</b>（每飞行小时致命事故），
其中<b>空中相撞分项 TLS_MAC = 2.5e-7 FAT/FH</b>（p40, p112 明文）。碰撞风险按气体动理学模型 &prop; N(N-1)（p112, p116），
严重度阶梯每级一个缓解 barrier（Providence 小型 UAS 取 0.01, p104）。据此把安全预算逐级"上溯"为可用作 RL 约束的<b>每-slot 战术冲突预算</b>：</p>
<div class="eqmono">
TLS_MAC 2.5e-7 FAT/FH ──(barrier 阶梯 MAC←NMAC←IC←SL←TC，逐级 mitigation)──▶
上游 TC(战术接触)违反是可接受的高频事件（每级 barrier 把风险下压约 1~2 数量级）
──(按峰值并发 N、操作时长归一化到 decision slot)──▶ 每-slot conflict limit ≈ 0.08
</div>
<p>即：真正致命的 MAC 概率被压在 2.5e-7；TC 距离被侵犯（战术冲突）是阶梯<b>最上游</b>、被下游多层 barrier 保护的事件，
因此允许以远高于 MAC 的频率发生。把该允许频率按峰值并发密度与 slot 时间窗归一化，得到 v19 拉格朗日 <b>conflict 约束限值 &asymp; 0.08</b>——
这使"冲突约束阈值"不再是自设常数，而是从 TLS_MAC 经 barrier 阶梯与运营密度<b>推导</b>出的量（Part D3 中 λ_conflict 的 limit）。</p>
<div class="note">诚实标注：BUBBLES <b>不给</b> RCP/时延/可靠性硬数字（fit-for-purpose 原则，p62 引 ED-261）。
τ/ε 的机制性解释 = "保证 T4 时间窗以给定置信度成立"（2σ/1σ/0σ 置信档）；0.08 的具体归一化系数是我们在
"barrier 阶梯 + G-13 密度"框架内的<b>工程映射</b>，写作时需声明为设计选择而非 D2.1 直接给出的数值。</div>

<h4>C2.6 CPA 双判据战术冲突（§3.3.4, p38 / p61）</h4>
<p>把旧的"单一距离阈值"冲突判据替换为 SESAR 的<b>战术冲突（Tactical Conflict）双判据</b>：
一对 UAV 只有<b>同时满足</b>下面两条才计为冲突——</p>
<table>
<tr><th>判据</th><th>条件</th><th>说明</th></tr>
<tr><td>空间</td><td>CPA 距离 &lt; d_TC</td><td>最近接近点距离低于战术接触分隔最小值</td></tr>
<tr><td>时间</td><td>time-to-CPA &lt; 60 s</td><td>迫近性门限：远期才接近的不算当前战术冲突</td></tr>
</table>
<p>CPA（Closest Point of Approach）由两机的<b>运动学</b>外推。<b>关键实现细节</b>：CPA 必须用
<b>assigned-UAV（被指派执行任务的 UAV）</b>的运动学，而非"最近的任意 UAV"——
否则冲突对策略几乎外生（策略唯一规避手段退化成选 cache），这是 Part D4 修复的 P1-env 根因。</p>

<h3 id="C3">C3 我们的实现（bubbles_separation.py）</h3>
<p>分隔链、TLS 锚定、CPA 双判据落地为 <code>bubbles_separation.py</code>，并以 <b>G-4 复现</b>为验证门。</p>
<table>
<tr><th>实现件</th><th>内容</th><th>验证门</th></tr>
<tr><td>分隔链 selftest</td><td>逐格复现官方 <b>Table G-4 / G-5</b></td><td class="best">全格误差 &lt; <b>0.18%</b></td></tr>
<tr><td>TC 口径校核</td><td>SAIL I&ndash;II 场景总接触距离 TC</td><td class="best"><b>370.53 m &rarr; 370.49 m</b>（口径一致）</td></tr>
<tr><td><code>scenario_profile</code> 门控</td><td>BUBBLES 作为<b>可选场景档</b>接入 v19 环境</td><td><code>profile=off</code> 时<b>逐字节回归</b>（不影响既有实验）</td></tr>
<tr><td>CPA 双判据生效性</td><td>冒烟场景注入迫近对</td><td class="best">冲突率从 <b>0 抬到 0.5</b>（判据确实触发）</td></tr>
<tr><td>整体单测</td><td>场景档全套单元测试</td><td class="best"><b>14/14 通过</b></td></tr>
</table>
<p><b>双工况设计</b>：分隔链本身经 selftest 锚定后作为固定"消费端"，接收语义链路给出的实测时延；
运营层按 C2.3 分为<b>峰值 <code>utm_conflict</code></b>（冲突约束被真正激活）与<b>平峰 <code>nominal</code></b>（标称评估）两档，
供 Part E 的 A/B 消融在"约束紧 / 约束松"两种压力下分别读数。这套场景档替代了此前的合成冲突模型，
为论文二提供<b>标准化、可对标 SESAR 的安全场景</b>。</p>

<h3 id="C4">C4 安全桥 E7：语义通信作为飞行安全变量</h3>
<p>前面 Part B 把证据选择的收益量化为<b>精度 / 带宽 / 延迟</b>。E7 更进一步：把<b>实测传输时延</b>注入 C2.4 的分隔链，
让"选 token 还是选 image"直接映射到<b>安全分隔距离与空域容量</b>——语义通信首次成为飞行安全变量。</p>
"""
+ img("F9_separation_capacity", "F9 证据模态 &rarr; 分隔距离/相对容量：把 s1 token 与 s2 image 的实测传输时延注入 BUBBLES 分隔链（-5&rarr;20dB Rician）。")
+ """
<table>
<tr><th>指标（-5dB Rician）</th><th class="num">s2 图像（image）</th><th class="num">s1 token</th><th>差异</th></tr>
<tr><td>安全接触距离 d_TC</td><td class="num bad">398.4 m</td><td class="num best">349.5 m</td><td><b>&minus;48.8 m</b>（token 更近更安全）</td></tr>
<tr><td>相对空域容量</td><td class="num bad">0.930</td><td class="num best">1.060</td><td><b>1.14&times;</b></td></tr>
<tr><td>@20dB 相对容量</td><td class="num" colspan="2" style="text-align:center">两者趋同</td><td>塌缩到 <b>1.01&times;</b></td></tr>
</table>
<p>机理（对应 C2.4 因果链）：弱链路下传整图的排队/传输时延膨胀 &rarr; 通信延迟分量&uarr; &rarr; T4&uarr; &rarr; Vc&middot;T4&uarr; &rarr; d_TC 更大，
把飞行器"看清并决策"的时刻推后，安全分隔必须留更大距离。选 token 把时延压掉，d_TC 缩短 48.8 m，
等价于 <b>1.14&times; 的空域相对容量</b>。SNR 升高后 image 时延不再是瓶颈，优势塌缩到 1.01&times;——
<b>证据选择的安全收益集中在弱链路</b>，与精度/延迟规律一致。</p>
<div class="note"><b>口径保守性</b>：F9 与上表<b>只注入了传输（tx）时延</b>，<b>不含 VLM 推理时延</b>。
若把 image 侧 Qwen2-VL 推理时延（-5dB 达 0.60s，见 B2）一并计入，token 的分隔/容量优势只会更大——当前数字是下界。</div>

<h3 id="C5">C5 引用策略（D2.1 + Weinert 2022 + EU 法规双引）</h3>
<p>D2.1 可引（SESAR JU 官方 PU 交付物），但<b>非同行评审</b>——关键论断必须配<b>双引</b>：</p>
<table>
<tr><th>论断</th><th>主引（D2.1 出处）</th><th>双引（peer-reviewed / 法规）</th></tr>
<tr><td>sNMAC / 碰撞风险模型</td><td>D2.1 p104 / p112</td><td><b>Weinert et al., AIAA JAIS 2022, DOI 10.2514/1.D0260</b></td></tr>
<tr><td>场景合法性（U-space 运营）</td><td>D2.1 ConOps</td><td><b>CIR (EU) 2021/664 / 665 / 666</b> 法规原文</td></tr>
<tr><td>ConOps / SORA 分级</td><td>D2.1 §3&ndash;4</td><td>CORUS ConOps、JARUS SORA、ICAO Doc 9854/9426</td></tr>
<tr><td>UTM 领域已采用 RL（方法背书）</td><td>D2.1 p140&ndash;141, p144（RL+GAN 学分隔最小值 + Multi-UAS OpenAI Gym 环境）</td><td>引为"UTM 已用 RL"先例</td></tr>
</table>
<div class="finding"><b>写作红线</b>：（1）通信 RCP/时延硬数字 D2.1 不给，需引 EUROCAE ED-282/ED-129B、3GPP C2 KPI 或自设并声明；
（2）conflict limit 0.08 的归一化系数标注为设计选择；（3）BUBBLES 期刊版（UPV 作者 Balbastre Tejedor / Vera Vélez 等）待查补引。</div>
""")

# ============================== PART D ==============================
PART_D = """
<h2 class="part" id="partD">Part D · 求解算法细节（详写）</h2>
<p class="sub">算法准确名称：<b>risk-aware Lagrangian 约束混合动作 PPO</b>，带语义-LCB Lyapunov 虚拟队列与语义可行性安全投影层；
双时间尺度变体增加慢速移动 actor（checkpoint 内部命名 <code>two_timescale_mobility_semantic_actor_critic</code>，
实验方法名 <code>proposed_semantic_cognitive_rl</code>）。<b>不是 TCH-PPO</b>——TCH-PPO 属旧仓库（vqa_semcom_v0），本仓库
<code>hybrid_ppo_lagrangian</code> baseline &asymp; 其重实现，v19 是在它之上多轮演进的产物
（解决"塌缩到 cache"：LCB 奖励缩放 &rarr; 熵退火 &rarr; BC 热启动 &rarr; 资源 floor &rarr; 安全投影 &rarr; 双时间尺度）。
本章回答"它实际在做什么"：形式化（D1）&rarr; 网络与训练（D2）&rarr; 五层约束（D3）&rarr; 正确性修复（D4）&rarr; 对偶消融定论（D5）。</p>

<h3 id="D1">D1 问题形式化（约束 MDP）</h3>
<p>建模为约束马尔可夫决策过程（CMDP）&#10216;S, A, P, r, {c_i}, {d_i}, &gamma;&#10217;，
目标 = 在满足所有约束的期望预算下最大化折扣语义效用：</p>
<div class="eqmono">
max<sub>&pi;</sub>  E[ &Sigma;<sub>t</sub> &gamma;<sup>t</sup> r(s<sub>t</sub>, a<sub>t</sub>) ]
  s.t.  E[ c<sub>i</sub> ] &le; d<sub>i</sub> ,  i &isin; {quality-n, quality-c, deadline-n, deadline-c, conflict, battery, GPU}
</div>

<h4>D1.1 状态（state_v2_fixed &asymp; 137 维）</h4>
<p>论文推荐 <code>state_v2_fixed</code>（v1 为 68 维），维度构成要点：</p>
<ul style="font-size:14px">
<li>任务：question_type one-hot + SNR / deadline / 缓存命中率等标量；</li>
<li>每服务等级（s0/s1/s2/s3）：LUT payload / accuracy + <b>7 个语义可行性特征</b>
（accuracy_LCB / 不确定度 / quality gap / 联合可行性等）；</li>
<li>每 UAV 7 维（位置 / 电量 / 状态）+ 每边缘 4 维（CPU / GPU / 显存 / 队列）；</li>
<li>5 条 Lyapunov 虚拟队列水位；</li>
<li>机动特征 + 各类 mask 特征（供更新端重放，见 D4）。</li>
</ul>

<h4>D1.2 混合动作空间（快头 / 慢头，联合分解策略）</h4>
<table>
<tr><th>头</th><th>类型</th><th>控制的资源</th><th>时间尺度</th></tr>
<tr><td>service_level</td><td>离散 {0,1,2(,3)}</td><td><b>语义证据路由</b>：0=缓存复用 / 1=检测 token / 2=图像（payload 由 LUT 给出）</td><td><b>快</b>：每 slot</td></tr>
<tr><td>resource（4 维高斯&rarr;sigmoid&rarr;[floor,max]）</td><td>连续</td><td>A2G 带宽份额（&times;1MHz）、发射功率 [0.05,1] W、边缘 CPU 份额、边缘 GPU 份额</td><td><b>快</b>：每 slot</td></tr>
<tr><td>mobility_uav / mobility_mode</td><td>离散</td><td>任务-UAV 指派；5 种机动模式（stay/serve/reposition/avoid_conflict/return_base）</td><td><b>慢</b>：每 K=3 slot</td></tr>
<tr><td>waypoint_delta / altitude_delta</td><td>连续</td><td>tanh&times;80m 水平位移、tanh&times;20m 高度增量（飞行时延/能耗）</td><td><b>慢</b>：每 K=3 slot</td></tr>
</table>

<h4>D1.3 约束集</h4>
<table>
<tr><th>约束 c_i</th><th>含义</th><th class="num">限值 d_i</th><th>来源</th></tr>
<tr><td>quality (normal)</td><td>语义质量 gap（风险 normal 任务）</td><td class="num">0.05</td><td>风险感知设定</td></tr>
<tr><td>quality (critical)</td><td>语义质量 gap（critical 任务）</td><td class="num">0.02</td><td>风险感知设定（更紧）</td></tr>
<tr><td>deadline (normal/critical)</td><td>超时违约率</td><td class="num">0.05 / 0.02</td><td>SLA</td></tr>
<tr><td>conflict</td><td>战术冲突率</td><td class="num">&asymp; 0.08</td><td><b>由 TLS_MAC 2.5e-7 经 barrier 阶梯推导（Part C2.5）</b></td></tr>
<tr><td>battery</td><td>电量耗尽风险</td><td class="num">&mdash;</td><td>UAV 能耗模型</td></tr>
<tr><td>GPU</td><td>边缘显存超限风险</td><td class="num">&mdash;</td><td>模型缓存 LRU</td></tr>
</table>

<h4>D1.4 奖励与环境</h4>
<p><b>奖励</b>（semantic_utility 模式，重度 shaped）：语义成功 / accuracy-LCB / 裕量正项（风险加权 critical&times;1.6）
&minus; 质量 gap 惩罚 &minus; 时延/能耗/payload 成本 &minus; 机动与安全成本
&minus; <b>缓存塌缩惩罚</b>（s=0 有语义缺口时 &minus;16&middot;gap 级）&minus; 拉格朗日项 &minus; 可选 Lyapunov 漂移；
另有 oracle/greedy 示范 BC 热启动 + 衰减辅助损失。
<b>环境</b>：每 episode 24 个决策时刻（slot 1s）；VisDrone VQA 任务流（staggered/burst/wave 到达）；
A2G 对数距离路损 + 仰角 LoS + 并发干扰 &rarr; SINR &rarr; 速率，SNR 量化到 {-5..20}dB bin；
时延六项分解（fly+sense+tx+queue+infer+load）、能耗四项、GPU 显存 + 模型缓存 LRU、语义缓存 fresh&rarr;stale&rarr;expired、UTM/DSS 冲突状态机。</p>

<h3 id="D2">D2 网络与训练</h3>
<table>
<tr><th>组件</th><th>配置</th></tr>
<tr><td>共享 encoder</td><td>MLP <b>(128, 128)</b> + ReLU；策略与价值<b>共享</b> encoder</td></tr>
<tr><td>各动作头</td><td>单层线性输出；连续头用<b>状态无关可学习 log_std</b></td></tr>
<tr><td>优化器</td><td>Adam，lr = 3e-4</td></tr>
<tr><td>PPO clip</td><td>0.2；每 episode <b>4 个 epoch</b></td></tr>
<tr><td>优势估计</td><td><b>无 GAE</b>：MC 折扣回报 &minus; V 基线 + batch 归一化</td></tr>
<tr><td>熵</td><td>系数 <b>0.05 &rarr; 0.005 线性退火</b>（抑制过早塌缩到 cache）</td></tr>
<tr><td>BC 热启动</td><td>oracle/greedy 示范辅助损失；先验权重 0.65 衰减，<b>bc_aux_weight 加时间衰减</b>（修复后，见 D4）</td></tr>
<tr><td>rollout 采集</td><td>包 <code>no_grad</code>（修复后）</td></tr>
</table>
<p><b>为什么无 GAE</b>：episode 短（24 步）且回报稠密（每 slot 有 shaped 奖励），GAE 的偏差-方差折中收益有限；
用 MC 回报 &minus; 共享 V 基线足以，且省一套 &lambda;_GAE 超参。代价是方差偏高——这也是 D5 里"欠训练 + 高方差"P2 的部分来源。</p>

<h3 id="D3">D3 五层约束机制（修复后版本）</h3>
<p>约束满足由五层机制<b>叠加</b>实现。注意：D5 的消融证明真正"承重"的是结构层（2&ndash;4），对偶层（1）收敛为定价。</p>
<table>
<tr><th>层</th><th>机制</th><th>细节（修复后）</th></tr>
<tr><td>① 拉格朗日对偶</td><td>7 个乘子 &lambda;</td><td>对偶上升 &lambda; &larr; clip(&lambda; + 0.05&middot;(cost &minus; limit), 0, 20)；
风险感知 limit（quality-n 0.05 / critical 0.02、conflict &asymp;0.08）；
<b>修复：加泄漏项（棘轮解除，&lambda; 可降）+ 分通道 &lambda; 上限 + 可达 limit</b></td></tr>
<tr><td>② 动作 mask</td><td>硬可行性掩码</td><td>GPU 显存可行性 mask 服务等级、电池 mask UAV、机动模式 mask</td></tr>
<tr><td>③ 语义可行投影</td><td>硬违规重选</td><td>硬违规时按语义安全分重选服务；<b>cache-override</b>（缓存缺口大强制升级证据）+
<b>deadline token&rarr;cache fallback</b>（事实上的拒绝/降级机制）</td></tr>
<tr><td>④ 资源 floor 投影</td><td>下限投影</td><td>token/image/ROI 各自的带宽/功率/CPU/GPU 下限；s=0 资源清零</td></tr>
<tr><td>⑤ Lyapunov 虚拟队列 &times;5</td><td>漂移-加-惩罚</td><td>quality/deadline/energy/risk/UTM 五队列进观测与奖励漂移惩罚；<b>修复：改增量式更新</b></td></tr>
</table>

<h3 id="D4">D4 正确性修复记录（双模型审查 &rarr; P0/P1 修复 &rarr; ratio&equiv;1 门）</h3>
<p>两个代理（Opus + Fable）对 <code>main @ ad31e9c</code> + BUBBLES worktree + 160 旧训练产物做<b>独立只读审查</b>，
交叉验证后结论<b>零矛盾</b>、关键发现互证。合并结论见 <code>docs_spec/V19_Design_Review_2026-07.md</code>。</p>

<h4>D4.1 九疑点判定总表</h4>
<table>
<tr><th>疑点</th><th>判定</th><th>要点</th></tr>
<tr><td>sigmoid squash 缺 Jacobian</td><td class="best">伪疑点</td><td>策略定义在 pre-squash 高斯上，新旧口径一致，Jacobian 在 ratio 中严格相消——无需修</td></tr>
<tr><td>慢头 credit assignment</td><td class="bad">真 P0</td><td>慢速移动头 log_prob <b>每步计入（&times;K=3）</b>；mobility mask 采样端 masked / 更新端 unmasked &rarr; ratio&ne;1 虚假触发 clip，熵也全步计入</td></tr>
<tr><td>两时标 mobility 代价双计</td><td class="bad">真 P0</td><td>semantic_utility 奖励内已含 mobility_cost，两时标路径又叠加 <code>_mobility_reward_adjustment</code>——旧"two-timescale vs monolithic"消融<b>失真，须修后重跑</b></td></tr>
<tr><td>&lambda; 棘轮</td><td class="bad">真 P1</td><td>limit=0 且 cost&ge;0 &rarr; &lambda; 单调不减、爬升顶格；旧 trace 实证 &lambda;_quality <b>0&rarr;5.41 全程单调</b>、违约率 0.9 始终不降——&lambda; 是障碍项不是影子价格</td></tr>
<tr><td>冲突多重罚</td><td class="bad">真 P1</td><td>BUBBLES 冲突同时置 airspace+utm 双 flag &rarr; shaped <b>&minus;3.5 重罚</b>，再叠 &lambda; 项 + Lyapunov 水位罚 + 隐性第 4 通道（success 翻负）；后期冲突步净罚 &minus;30~&minus;50 vs 正项上限 +23，cache 塌缩是理性解</td></tr>
<tr><td>BC 贴示范锁死</td><td class="bad">真 P1</td><td><code>bc_aux_weight=0.28</code> <b>永不衰减</b>（只有 0.65 先验衰减）；实证 low_snr 下 proposed&asymp;教师（0.955 vs 0.950）</td></tr>
<tr><td>CPA 用 nearest UAV</td><td class="bad">真 P1-env</td><td>CPA 运动学用 <b>nearest 而非 assigned UAV</b>（multi_uav_env.py:1918）&rarr; 冲突对策略基本外生，唯一规避手段是选 cache——冲突可控性的环境侧根因</td></tr>
<tr><td>advantage 归一化中和 &lambda;</td><td class="bad">真 P1</td><td>per-episode 归一化把 episode 内近常数的 &lambda; 罚减均值消掉 &rarr; 对偶机制一阶效应被架空</td></tr>
<tr><td>欠训练 + 高方差</td><td class="bad">真 P2</td><td>120ep 回报仍在涨、末期 std&gt;|mean|、容量非瓶颈；rollout 采集缺 no_grad</td></tr>
</table>

<h4>D4.2 P0 + P1 修复（feat/bubbles-scenario：541c39c / 428aec7 / 2cd5212 / 3999ad6）</h4>
<table>
<tr><th>修复项</th><th>内容</th></tr>
<tr><td>慢头 credit assignment（P0）</td><td>log_prob 决策步计一次（非决策步置 0）+ mobility mask 存储供更新端重放 + 熵乘 decision mask</td></tr>
<tr><td>mobility 双计（P0）</td><td>删两时标路径的 <code>_mobility_reward_adjustment</code> 叠加</td></tr>
<tr><td><b>单测门 ratio&equiv;1</b></td><td><b>epoch-0 ratio&equiv;1</b>：采集后立即重算 log_prob，断言与 old_log_probs 逐步相等；<b>133 项测试全过</b></td></tr>
<tr><td>&lambda; 通道（P1）</td><td>&lambda; 更新加<b>泄漏项</b> + 分通道 &lambda; 上限（棘轮解除，&lambda; 可降）</td></tr>
<tr><td>罚通道去重（P1）</td><td>冲突同源只罚一次（utm 通道权重归零），Lyapunov 队列改增量式</td></tr>
<tr><td>BC 衰减（P1）</td><td>bc_aux_weight 加时间衰减，不再永久贴教师</td></tr>
<tr><td>assigned-UAV CPA（P1-env）</td><td>CPA 运动学改用 assigned UAV，恢复冲突对策略的可控性</td></tr>
<tr><td>归一化 / 效率</td><td>advantage 改 <b>scale-only 归一化</b>（保均值、不消 &lambda; 罚）；rollout 采集包 no_grad</td></tr>
</table>

<h3 id="D5">D5 三轮对偶消融与定论："结构承重 + 对偶定价"</h3>
<ol class="roman">
<li><b>第一轮（v2）</b>：关掉 &lambda;_conflict 通道的 <b>B2</b> 与全量 <b>A2</b> 行为完全相同——首次发现对偶通道可能不承重；
当时怀疑是多通道罚冗余掩盖。</li>
<li><b>第二轮（v3，罚通道单通道化后）</b>：去重后复测，<b>B2 仍与 A2 逐位相同</b>（冲突率 <b>0.3027</b> 逐位同、全指标逐位同）——
排除"冗余通道顶替"解释。剩余假设：&lambda; 从 0 起步"迟到"，等它长起来时策略已被结构件带进无冲突区。</li>
<li><b>第三轮（A2warm，提交 b176b52 / 1b9902d）</b>：&lambda;_conflict = 4.0 <b>热启动、全程在场</b>——
结果<b>仍与 A2 逐位相同</b>。"迟到"假设被否证。</li>
</ol>
<div class="finding"><b>深挖（为何梯度在流、行为却不变）</b>：A2warm 与 A2 的网络权重确有差异
（max|&Delta;w| = 0.35，&lambda; 罚的梯度确实流过并改了参数），但策略行为<b>重新收敛到同一解</b>；
且 &lambda; 三种起法（0 起步 / 单通道 / 4.0 热启动）均收敛到 <b>3.7&ndash;4.7 同一均衡带</b>——
对偶变量把"价格"算对了，但 primal 侧被安全投影、mask、资源 floor 等硬结构钉死，价格信号改不动行为：
<b>价格对、控制无效——primal 钝化</b>（primal saturation）。</div>
<p><b>&lambda; 影子价格分层</b>（收敛区间随约束紧迫度分层，说明<b>定价功能正常</b>）：</p>
<table>
<tr><th>Arm</th><th class="num">&lambda;_conflict 收敛带</th><th>解读</th></tr>
<tr><td>A2 全量（结构齐全）</td><td class="num">3.7&ndash;4.7（末段均值 &asymp;4.31）</td><td>约束不太紧 &rarr; 价格中等</td></tr>
<tr><td>B1 无慢速移动头（结构缺位）</td><td class="num bad">6.6&ndash;7.1</td><td><b>约束更紧 &rarr; 影子价格更高</b>（方向正确）</td></tr>
<tr><td>A1 legacy（尺度不同）</td><td class="num">0.8&ndash;1.1</td><td>奖励尺度不可比，仅参照</td></tr>
</table>
<p>"约束越紧、影子价格越高"正是对偶变量作为价格应有的行为——这是 &lambda; <b>定价功能正常</b>的证据，
尽管它<b>不控制</b>行为。承重件分解见下表：</p>
<table>
<tr><th>承重件</th><th class="num">因果贡献</th><th>证据</th></tr>
<tr><td><b>慢速移动头</b>（结构）</td><td class="num best">冲突率 &minus;0.168</td><td>B1 消融，v2/v3 两轮稳定复现（0.4707 &rarr; 0.3027）</td></tr>
<tr><td><b>安全投影层</b>（结构）</td><td class="num best">semSucc &minus;0.21</td><td>旧组件消融最大单项（去掉后向 cache 塌缩）</td></tr>
<tr><td><b>隐式通道</b>（奖励结构）</td><td class="num">冲突&rarr;任务失败丢 +6 正项</td><td>冲突步 success 翻负，本身就是强规避信号——无需 &lambda; 也在罚</td></tr>
<tr><td>&lambda;_conflict（对偶）</td><td class="num">行为贡献 &asymp; 0</td><td>三轮消融逐位相同；但收敛为<b>稳定影子价格</b>（分层正确），可作约束紧迫度度量</td></tr>
</table>
<div class="note"><b>论文二叙事口径</b>：不声称"Lagrangian 对偶控制了冲突"；改为
<b>"结构化安全机制承担约束满足（constraint satisfaction by construction），对偶变量收敛为可解释的影子价格，
用作约束紧迫度的在线度量"</b>。三轮递进消融（发现 &rarr; 排除冗余解释 &rarr; 否证迟到假设 &rarr; 权重-行为深挖）
本身构成论文二核心证据链——一个方法论上干净的 <b>primal 钝化</b>案例。
<br><b>Future work（救活对偶控制，标注不实施）</b>：（1）减弱安全投影，把部分约束满足责任让渡给 &lambda;；
（2）credit assignment 对准慢头决策步，让 &lambda; 罚梯度作用在真正可控的自由度上。</div>
"""

# ============================== PART E ==============================
PART_E = """
<h2 class="part" id="partE">Part E · RL 实验结果</h2>
<p class="sub">数据：<code>outputs/rl/ab_bubbles_v3/ab_summary.md</code>（160，BUBBLES 主线 v3 双工况全表 + &lambda; 轨迹）。
Arms：<b>A2</b>（bubbles 全量，修复后）/ <b>B1</b>（无慢速移动头）/ <b>B2</b>（关 &lambda;_conflict 通道）/
<b>A1</b>（legacy 档，尺度仅参照）/ <b>Cgreedy</b>、<b>Ccache</b>（非学习基线）。</p>

<h3 id="E1">E1 单场景双工况 A/B</h3>
<p><b>峰值工况（utm_conflict）</b>：</p>
<table>
<tr><th>Arm</th><th class="num">acc</th><th class="num">conflict 率</th><th class="num">cache 率</th><th>读解</th></tr>
<tr><td><b>A2 全量</b></td><td class="num best">0.5628</td><td class="num">0.3027</td><td class="num">0.3907</td><td>学习方主线：acc 大幅高于两基线，冲突率压到 greedy 的一半以下</td></tr>
<tr><td>B2 关 &lambda;_conflict</td><td class="num">0.5628</td><td class="num">0.3027</td><td class="num">0.3907</td><td><b>与 A2 逐位相同</b>——对偶通道不承重（见 D5）</td></tr>
<tr><td>B1 无慢速移动头</td><td class="num">0.5815</td><td class="num bad">0.4707</td><td class="num">0.2173</td><td>冲突率 +0.168：慢头是冲突控制的<b>结构承重件</b></td></tr>
<tr><td>A1 legacy（尺度不可比）</td><td class="num">0.5332</td><td class="num">0.1147</td><td class="num">&mdash;</td><td>奖励尺度不同，仅参照，不进结论</td></tr>
<tr><td>Cgreedy</td><td class="num">0.3859</td><td class="num bad">0.6540</td><td class="num">&mdash;</td><td>贪心全发：冲突失控</td></tr>
<tr><td>Ccache</td><td class="num bad">0.3127</td><td class="num best">0.0000</td><td class="num">&mdash;</td><td>全缓存：零冲突但精度塌缩</td></tr>
</table>
<p><b>nominal 工况</b>：</p>
<table>
<tr><th>Arm</th><th class="num">acc</th><th class="num">semSucc</th><th class="num">taskSucc</th><th class="num">cache 率</th></tr>
<tr><td><b>A2 全量</b></td><td class="num best">0.6334</td><td class="num best">0.9463</td><td class="num">0.2760</td><td class="num">0.5923</td></tr>
<tr><td>Cgreedy</td><td class="num">0.6241</td><td class="num">0.8960</td><td class="num best">0.3680</td><td class="num">&mdash;</td></tr>
<tr><td>Ccache</td><td class="num bad">0.2655</td><td class="num bad">0.2970</td><td class="num">&mdash;</td><td class="num">&mdash;</td></tr>
</table>

<h3 id="E2">E2 慢头因果与精度-安全权衡</h3>
<div class="finding"><b>慢速移动头因果 &minus;0.168</b>（峰值工况 B1 消融）：去掉慢头，冲突率从 A2 的 0.3027 抬到 B1 的 0.4707；
这是"约束满足由结构承担"的最强单项证据，v2/v3 两轮稳定复现。</div>
<p><b>精度-安全权衡</b>：注意 B1（无慢头）的 acc <b>0.5815 反而略高于</b> A2 的 0.5628——
放开慢头约束后策略更"激进地发图 / 少缓存"（cache 率 0.2173 &lt; 0.3907），精度略升但<b>冲突率翻到 1.55&times;</b>。
A2 用略微的精度让步换来冲突率减半，是安全-精度前沿上的正确取舍；两条非学习基线（Cgreedy 冲突失控 / Ccache 精度塌缩）
则是前沿的两个坏端点。&lambda;_conflict 分层（A2 3.7&ndash;4.7 / B1 6.6&ndash;7.1）在此定量印证"结构缺位&rarr;约束更紧&rarr;影子价格更高"。</p>

<h3 id="E3">E3 进行中：标准化实验矩阵 matrix_v1（结果待更新）</h3>
<div class="note"><b>状态</b>：截至 2026-07-05，160 上 <code>outputs/rl/matrix_v1/</code> <b>尚未生成产物</b>（已只读核查：
现有目录为 <code>semantic_scenario_benchmark_v1..v4</code>、<code>low_snr_deadline_tuning</code>、
<code>two_timescale_mobility_formal</code> 等中间产物）。下表为按
<code>docs_spec/RL_Experiment_Standards_Survey.md</code>（7 篇 TWC/JSAC/IoTJ 深读）拟定的<b>实验矩阵计划</b>，
标注"运行中，结果待更新"；数字将在 matrix_v1 跑出后回填。</div>
<p>训练协议：3&ndash;5 seeds &times; &ge;1000 ep；测试每点 32+ rollouts；默认 = BUBBLES 标称工况。</p>
<table>
<tr><th>编号</th><th>内容</th><th>跑什么</th><th>状态</th></tr>
<tr><td>Fig.1 三联</td><td>(a) reward vs ep (b) 约束成本 + 界线 (c) &lambda; 轨迹；proposed / no_lagrangian / fixed-penalty / service_only 同图，&plusmn;std</td><td>trace 直接画 + fixed-penalty 补训</td><td>运行中</td></tr>
<tr><td>Fig.2</td><td>多规模收敛（UAV 数 3 档，仅 proposed）</td><td>补训 2 点</td><td>计划</td></tr>
<tr><td>Fig.3&ndash;5</td><td>性能 vs UAV 数 / 到达率 / SNR（全方法 8 条线）</td><td>零样本评估扫描</td><td>计划</td></tr>
<tr><td>Fig.6</td><td>约束满足对比：违约率柱状（双工况）+ 效用-违约 Pareto</td><td>汇总 3&ndash;5 数据</td><td>部分（E1 已有双工况数据）</td></tr>
<tr><td>Fig.7</td><td>零样本泛化专图（未见 profile / &times;1.5 负载 / 低 SNR，vs 重训参考点）</td><td>评估 + 少量重训</td><td>计划（差异化加分）</td></tr>
<tr><td>Table I</td><td>主对比：双工况 &times; 8 方法，mean&plusmn;std，含 oracle/random</td><td>汇总</td><td>运行中</td></tr>
<tr><td>Table II</td><td>消融（已有 v3 数据，加 mean&plusmn;std）</td><td>已有（E1）</td><td class="best">数据已在</td></tr>
<tr><td>Table III</td><td>样本效率（达 95% 最终 reward 的 ep 数 + 墙钟）</td><td>trace 计算</td><td>计划</td></tr>
</table>
<p><b>当前训练瓶颈</b>（main tip "low-snr deadline tuning"，只读引用）：formal 矩阵 5 seeds &times; 500 ep（目标 1000），
<code>edge_overload</code> 场景最强（任务成功 <b>0.649</b>、零超时）；主要瓶颈 <code>low_snr_blockage</code> 超时违约 <b>0.854</b>——
诊断为弱链路下 token 传输时延主导（违约任务 tx_delay 占比 1.0），T1/T2/T3 调参仅改善到 0.846，
结论是需要<b>链路/压缩层面的改动而非纯 RL 调参</b>——正是 B7 可变 token 预算（t 截断把 payload 再压 ~30%、低 SNR 帧更少更稳）
与 P4 置信度门控补传的接入位置。</p>
<p>优先级：Fig.1 &rarr; Table I 补臂 &rarr; Fig.3&ndash;5 &rarr; Fig.6 &rarr; 扁平 PPO &rarr; Fig.2/7 &rarr;（可选）DQN。
基线套餐已达 7 个（proposed / no_lagrangian / fixed-penalty / service_only / greedy / cache / oracle），超过多数样本，SAC/TD3 判定"不值得"。</p>
"""

# ============================== PART F ==============================
PART_F = """
<h2 class="part" id="partF">Part F · 收尾</h2>

<h3 id="F1">F1 发现清单（论文叙事骨架，12 条）</h3>
<div class="finding">&#9312; <b>自适应语义系统在理想信道下也胜过传图</b>：M4@20dB（0.668&ndash;0.670）&gt; M0_errorfree（0.615）——增益来自证据选择本身，不只是信道鲁棒性。</div>
<div class="finding">&#9313; <b>证据-问题互补性</b>：符号推理&rarr;token（+0.12~0.17），感知推理&rarr;图像（&minus;0.09）；零参数规则=数据标定策略（0.6725）。</div>
<div class="finding">&#9314; <b>悬崖 vs 平滑</b>：固定速率数字断崖（-5dB 0.37-0.41）；token/模拟/自适应平滑退化。</div>
<div class="finding">&#9315; <b>跨 VLM 稳健</b>：路由 4/5 一致，分歧点=互补性最弱题型。</div>
<div class="finding">&#9316; <b>调度无需精确 CSI</b>：失配矩阵平坦，最差失配=匹配最低值。</div>
<div class="finding">&#9317; <b>逐样本预测器</b>：+0.7~1.1pt 且 payload 减半（Pareto 改进）。</div>
<div class="finding">&#9318; <b>token 预算双重角色</b>：成本旋钮 + 假阳性过滤器（threshold +7.7pt）。</div>
<div class="finding">&#9319; <b>延迟</b>：-5dB 下 token 0.089s vs 图像 4.32s（49&times;）——实时性证据。</div>
<div class="finding">&#9320; <b>互补性跨数据集复现，且规则反超标定</b>（B8）：DroneVehicle 上符号顺序完全复现，零参数规则 0.7293 <b>反超</b>逐数据集标定 LCB 0.7246。</div>
<div class="finding">&#9321; <b>路由跨三个 VLM 家族一致</b>（B4）：Qwen2-VL / Qwen2.5-VL / SmolVLM-Idefics3 三家族 4/5 题型路由相同；token 列与 VLM 无关；弱 VLM 放大 token 优势。</div>
<div class="finding">&#9322; <b>证据选择 = 飞行安全变量</b>（C4）：把实测传输时延注入飞行环，-5dB Rician 下选 token 相对 image 拉近安全接触距离 <b>48.8m</b>（398.4&rarr;349.5m）、换来 <b>1.14&times;</b> 空域相对容量。</div>
<div class="finding">&#9323; <b>对偶通道不承重——"结构承重 + 对偶定价"</b>（D5）：三轮递进消融下 B2 与 A2 逐位相同，冲突约束满足实际由慢速移动头（&minus;0.168）与安全投影（&minus;0.21）承担；&lambda; 收敛为稳定影子价格——一个干净的 <b>primal 钝化</b>案例。</div>

<h3 id="F2">F2 新颖性与期刊定位</h3>
<p>摘自 <code>docs_spec/Novelty_Journal_Assessment_2026-07.md</code>（deep-research 管线：5 检索角度 &times; 21 一手来源 &times; 105 条提取论断，3 条完成三票对抗核验）。</p>
<h4>六近邻划界（语义通信侧 &#9312;&ndash;&#9319; 先例核查）</h4>
<table>
<tr><th>我方发现</th><th>最近邻工作</th><th>我方增量</th></tr>
<tr><td>&#9312; 按题型路由证据模态 + 规则=标定</td><td><b>GO-SG</b>(2411.02452)、<b>Park&amp;Yoon</b>(JSAC'25, 2412.13646)</td><td>同一任务(VQA)内按题型的实证互补性分解 + 自动路由 + 规则=标定等价性；两篇均须引用划界</td></tr>
<tr><td>&#9313; 无差错信道也胜过传整图</td><td>GO-SG 把整图当<b>精度上界</b>（与我方相反）</td><td class="best">真差异化发现，文献无对应物【核验 2-0】</td></tr>
<tr><td>&#9314; 统一 complex-channel-uses 计费</td><td>TOFC(2503.12926)、2604.26508 均无物理信道建模</td><td>模拟/数字统一计费 + 3 信道&times;LDPC 全物理层在近邻中空白</td></tr>
<tr><td>&#9317; CSI 失配矩阵（调度层鲁棒性）</td><td>ADJSCC 等评估永远馈真值 SNR</td><td class="best">语义层 CSI 误差度量无先例</td></tr>
<tr><td>&#9318; 逐样本学习选择器</td><td>PADC(TWC'23) OracleNet 预测 PSNR 选速率（重建域）</td><td>任务精度域 + 发送端特征 + 选证据模态是新组合</td></tr>
<tr><td>&#9319; token 截断兼过滤（少发更准）</td><td>2604.26508 结论是单调退化；P&amp;Y 过滤场景图略超全图</td><td>置信度截断 + 按题型分化(t=3 vs t=48) + VQA 精度 +7.7pt + 假阳性过滤机理</td></tr>
</table>
<div class="note"><b>必须处理的近邻</b>：arXiv <b>2505.02413</b>（LLaVA 接收端 + F 衰落 + 端到端 VQA 精度）——"真实 VLM 接收端过无线链路"<b>已有先例</b>，不能声称第一。
但其仅 41 图/172 题（我方 548 图，13&times;）、地面交通场景、模态内空间选块（非证据等级路由）、启发式功率分配（无 DRL）。
口径应改为：<b>"最大规模的 / UAV 场景的 / 证据等级路由的真实 VLM 无线 VQA 评测"</b>。</div>
<h4>期刊定位判断（TCCN 主投）</h4>
<table>
<tr><th>方案</th><th>判断</th><th>依据</th></tr>
<tr><td>论文一（语义通信有效性）投 <b>TCCN</b></td><td class="best">够格，主推</td><td>系统+方法论+三个未被声称的发现（&#9313;&#9317;&#9319;）；TCCN 对认知/语义通信实证系统接受度高</td></tr>
<tr><td>论文一冲 TWC</td><td>边缘，需补强</td><td>需互补性 DPI 形式化命题 + 逐样本预测器升为正式贡献 + 第二数据集（B8 已补）</td></tr>
<tr><td>论文二（v19 资源分配）投 <b>TCCN / IoTJ</b></td><td class="best">够格</td><td>语义感知约束 RL + 完整低空环境(UTM/DSS/BUBBLES) 在 IoTJ/TCCN 是强贡献</td></tr>
<tr><td>论文二冲 TWC</td><td>需理论件</td><td>加 Lyapunov 漂移-加-惩罚稳定性/最优性间隙分析（框架成熟）</td></tr>
</table>
<h4>口径红线</h4>
<div class="finding">1. <b>不得声称"第一个真实 VLM 无线 VQA"</b>——2505.02413 在先；改口径为规模/场景/路由维度并引用划界。</div>
<div class="finding">2. <b>&#9312;&#9319; 近邻必须补引</b>：GO-SG、Park&amp;Yoon(JSAC'25)、2604.26508——related work 加差异表。</div>
<div class="finding">3. <b>M2 定位为"受控模拟代表"</b>，非学习型 JSCC；引用划界，勿声称击败 DeepSC 系列。</div>
<div class="finding">4. <b>互补性需理论件</b>：DPI / 充分统计量命题（计数：检测计数为充分统计量；存在性：召回下界）撑起 TWC 级贡献。</div>
<div class="finding">5. <b>撞车时间压力</b>：2505.02413 与 2604.26508 方向快速演进——建议论文一 1&ndash;2 个月内投出。</div>

<h3 id="F3">F3 复现与工件</h3>
<table>
<tr><th>项</th><th>位置</th></tr>
<tr><td>对比实验代码+数据</td><td><code>codex/lut-semantic-utility-upgrade</code> 分支：build_comparison_v2 / build_mismatch_matrix / build_persample_policy / build_token_budget_sweep / analyze_crossvlm + outputs/reports/*.csv + F1&ndash;F8</td></tr>
<tr><td>资源分配 RL</td><td><code>main</code> 分支：src/vqa_semcom/rl/v19_ppo.py + v19_resource_env.py；<code>feat/bubbles-scenario</code> 分支 8 提交（场景档 + P0/P1 修复 + A2warm 消融）</td></tr>
<tr><td>实验机</td><td>lab-s2 <code>~/phd_research/vqa_semcom</code>（RTX 4060 8GB；全链 ~37h 重算）</td></tr>
<tr><td>原始预测备份</td><td>本地 <code>HPPO-VQA/outputs/backup_160/</code>（重算 12 CSV + v25 三件；DroneVehicle 产物；三 VLM + E7/BUBBLES 产物）</td></tr>
<tr><td>BUBBLES 精读</td><td><code>docs_spec/BUBBLES_D2.1_Analysis.md</code>（156 页逐页精读）；<code>bubbles_separation.py</code>（selftest &lt;0.18%）</td></tr>
<tr><td>双模型审查</td><td><code>docs_spec/V19_Design_Review_2026-07.md</code>（Opus + Fable 独立只读审查，零矛盾）</td></tr>
<tr><td>A/B 消融汇总（160）</td><td><code>outputs/rl/ab_bubbles_v3/ab_summary.md</code>（BUBBLES 主线 v3 双工况全表 + &lambda; 轨迹）</td></tr>
<tr><td>RL 实验矩阵依据</td><td><code>docs_spec/RL_Experiment_Standards_Survey.md</code>（7 篇 TWC/JSAC/IoTJ 深读；matrix_v1 计划见 E3）</td></tr>
<tr><td>参考实现代码考据</td><td><code>docs_spec/SemCom_Reference_Repos_Code_Analysis.md</code>（MA-DeepSC/PADC/ADJSCC/DeepJSCC-f/SJTU）</td></tr>
</table>
<p class="sub">重算与 182 原结果交叉验证：AWGN M4 0.628&rarr;0.668（原 0.627&rarr;0.668）、naive 悬崖 0.369&ndash;0.409（原 0.37&ndash;0.41）、error-free 0.615（原 0.611）——逐点复现。</p>

</body></html>"""

HTML = HEAD + CARDS + PART_A + PART_B + PART_C + PART_D + PART_E + PART_F

out = "/Users/zhangqiankun/Documents/mpu/HPPO-VQA/docs_spec/UAV_VQA_SemCom_System_and_Results_v5.html"
open(out, "w").write(HTML)
print(f"wrote {out} ({len(HTML)//1024} KB)")

# ---- 自校验 ----
empty_figs = [k for k, v in FIGS.items() if not v]
parts = HTML.count('h2 class="part"')
key = ["0.6725", "1.14&times;", "&minus;0.168", "0.3027", "370.49", "2.5e-7", "primal 钝化"]
missing = [k for k in key if k not in HTML]
# Part C/D 字数 vs v4 对应节（v4 §13 safety + §2 rl + §15 audit 近似基线）
lenC = len(PART_C)
lenD = len(PART_D)
print(f"figs: {len(FIGS)} total, empty={empty_figs or '无'}")
print(f"part banners: {parts} (expect 6: A-F)")
print(f"key numbers missing: {missing or '无 — 全部命中'}")
print(f"Part C chars: {lenC} | Part D chars: {lenD}")
# v4 对应节粗基线：§13 ~2600、§2+§15 ~6800（详写目标应显著超出）
print(f"Part C vs v4 §13 (~2600): {'PASS 显著更多' if lenC > 5000 else 'FAIL'}")
print(f"Part D vs v4 §2+§15 (~6800): {'PASS 显著更多' if lenD > 9000 else 'FAIL'}")
assert not empty_figs, "有空图"
assert parts == 6, f"part banners != 6 ({parts})"
assert not missing, f"关键数字缺失 {missing}"
assert lenC > 5000 and lenD > 9000, "Part C/D 详写字数不足"
print("v5 校验通过")
