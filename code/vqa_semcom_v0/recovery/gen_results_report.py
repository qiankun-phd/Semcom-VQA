#!/usr/bin/env python3
import csv, base64, os
from collections import defaultdict

SC = "/private/tmp/claude-501/-Users-zhangqiankun-Documents-mpu-HPPO-VQA-vqa-semcom-v0/eee254b4-d6b6-4e0d-b945-718af1a4cdc7/scratchpad"
OUT = "/Users/zhangqiankun/Documents/mpu/HPPO-VQA/docs_spec"
CSV = f"{SC}/comparison_all.csv"

rows = list(csv.DictReader(open(CSV)))
def cell(ch, m, snr, qt="all", split="test", field="accuracy"):
    for r in rows:
        if r["channel"]==ch and r["method"]==m and abs(float(r["snr_db"])-snr)<1e-6 and r["qtype"]==qt and r["split"]==split:
            return float(r[field])
    return None

SNRS = [-5,0,5,10,15,20]
CHANNELS = [("awgn","AWGN"),("rayleigh","Rayleigh"),("rician","Rician K=6 dB")]
METHODS = [  # id, label
    ("M5_oracle","M5 Oracle 上界"),
    ("M4_adaptive","M4 我们(自适应)"),
    ("M3_token","M3 GO-SG token"),
    ("M1_image","M1 传统(率自适应图)"),
    ("M2_analog","M2 DeepSC 模拟"),
    ("M0_naive","M0 naive 数字(定速)"),
]
OURS = "M4_adaptive"

def cost(ch, m):
    vs = [float(r["mean_payload_bytes"]) for r in rows if r["channel"]==ch and r["method"]==m and r["qtype"]=="all" and r["split"]=="test"]
    return sum(vs)/len(vs) if vs else None

# ---------- markdown ----------
md = []
md.append("# UAV-VQA 语义通信 — 对比实验结果报告\n")
md.append("> held-out test set（按 image_id 20% 留出）· 6 方法 × 3 信道 × 6 SNR · VisDrone-DET 548 图 · Qwen2-VL-2B 统一后端\n")
md.append("> 2026-06-29 · 配套 `UAV_VQA_SemCom_Comparison_Plan`\n")

md.append("\n## 摘要\n")
md.append("- **M4(我们的自适应服务选择）在三信道、每个 SNR 都超过所有固定基线**，贴近 Oracle 上界（差约 0.05）。\n")
md.append("- **cliff-effect 成立**：naive 定速数字在 −5dB 断崖至 0.37–0.41；DeepSC 模拟与我们的自适应优雅降级。\n")
md.append("- **机理**：计数任务 token 完胜（整图 0.28 vs token 0.46），存在性任务整图更优；M4 逐任务路由取两者之长。\n")
md.append("- **效率**：M4 较传统整图省约 34% 字节，准确率反而更高。\n")

md.append("\n## T1 主结果：准确率 vs SNR（test, all qtypes）\n")
for ch,chl in CHANNELS:
    md.append(f"\n**{chl}**\n")
    md.append("| 方法 | " + " | ".join(f"{s}dB" for s in SNRS) + " |")
    md.append("|---|" + "---|"*len(SNRS))
    for m,lbl in METHODS:
        vals = []
        for s in SNRS:
            v = cell(ch,m,s)
            vals.append(f"{v:.3f}" if v is not None else "-")
        star = " ⭐" if m==OURS else ""
        md.append(f"| {lbl}{star} | " + " | ".join(vals) + " |")

md.append("\n## T2 传输代价（mean payload bytes/query, test, avg over SNR）\n")
md.append("| 方法 | AWGN | Rayleigh | Rician |")
md.append("|---|---|---|---|")
for m,lbl in METHODS:
    c = [cost(ch,m) for ch,_ in CHANNELS]
    md.append(f"| {lbl} | " + " | ".join(f"{x:,.0f}" if x is not None else "-" for x in c) + " |")
md.append("\n注：M2 模拟占满整个时隙（≈300,000 复信道符号，单位非字节，列为等效）；M4 vs M1 同为数字、字节可比，省约 34%。\n")

md.append("\n## 机理:按问题类型(Rician, SNR=5dB, test)\n")
md.append("| 任务 | M1 整图 | M3 token | M4 选 | Oracle |")
md.append("|---|---|---|---|---|")
for qt,qtl in [("presence","presence 存在性"),("counting","counting 计数")]:
    r = [cell("rician",m,5,qt=qt) for m in ["M1_image","M3_token","M4_adaptive","M5_oracle"]]
    md.append(f"| {qtl} | " + " | ".join(f"{x:.3f}" if x is not None else "-" for x in r) + " |")
md.append("\n计数任务整图惨败(Qwen 数不清退化航拍图),检测器 token 本就在数物体 → 完胜;存在性反之。\n")

md.append("\n## 图表\n")
md.append("- **F1** 三面板 Acc-SNR(AWGN/Rayleigh/Rician)\n- **F2** cliff-effect 招牌图(Rayleigh)\n- **F4** 按问题类型柱图\n- **F5** Pareto(准确率×代价)\n")
md.append("\n图文件:`outputs/figures/comparison/F{1,2,4,5}_*_final.{png,pdf}`(服务器 182);汇总数据 `outputs/reports/comparison_all.csv`。\n")

md.append("\n## 诚实 caveat\n")
md.append("1. M2 DeepSC 模拟在中低 SNR 不如离散 token(噪声图喂 Qwen 不稳),但低 SNR 无悬崖(−5dB 0.46 > naive 0.41)——帮我们讲透'数字/模拟/自适应'三方。\n")
md.append("2. 计数任务整体偏难(连 Oracle 才 ~0.50),为 VQA 计数固有难度。\n")
md.append("3. M4 为规则版 LCB 贪心(已 train/test 切分,排除过拟合);RL 资源分配(实验组 2)仍待做。\n")

open(f"{OUT}/UAV_VQA_SemCom_Results_Report.md","w").write("\n".join(md))

# ---------- html ----------
def b64(path):
    return base64.b64encode(open(path,"rb").read()).decode()
imgs = {k:b64(f"{SC}/{v}") for k,v in {"F1":"F1_final.png","F2":"F2_final.png","F4":"F4_final.png","F5":"F5_final.png"}.items()}

def t1_html():
    h=[]
    for ch,chl in CHANNELS:
        h.append(f'<h3>{chl}</h3><table><tr><th>方法</th>'+"".join(f"<th>{s}dB</th>" for s in SNRS)+"</tr>")
        for m,lbl in METHODS:
            cls=' class="ours"' if m==OURS else ''
            tds=""
            for s in SNRS:
                v=cell(ch,m,s); tds+=f"<td>{v:.3f}</td>" if v is not None else "<td>-</td>"
            h.append(f"<tr{cls}><td>{lbl}</td>{tds}</tr>")
        h.append("</table>")
    return "".join(h)

def t2_html():
    h=['<table><tr><th>方法</th><th>AWGN</th><th>Rayleigh</th><th>Rician</th></tr>']
    for m,lbl in METHODS:
        cls=' class="ours"' if m==OURS else ''
        c=[cost(ch,m) for ch,_ in CHANNELS]
        h.append(f"<tr{cls}><td>{lbl}</td>"+"".join(f"<td>{x:,.0f}</td>" if x is not None else "<td>-</td>" for x in c)+"</tr>")
    h.append("</table>")
    return "".join(h)

def qtype_html():
    h=['<table><tr><th>任务</th><th>M1 整图</th><th>M3 token</th><th>M4 选</th><th>Oracle</th></tr>']
    for qt,qtl in [("presence","presence 存在性"),("counting","counting 计数")]:
        r=[cell("rician",m,5,qt=qt) for m in ["M1_image","M3_token","M4_adaptive","M5_oracle"]]
        h.append(f"<tr><td>{qtl}</td>"+"".join(f"<td>{x:.3f}</td>" if x is not None else "<td>-</td>" for x in r)+"</tr>")
    h.append("</table>")
    return "".join(h)

html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAV-VQA 语义通信 — 对比实验结果</title>
<style>
:root{{--bg:#0f1419;--panel:#161c26;--panel2:#1d2530;--ink:#e6edf3;--muted:#9aa7b4;--acc:#4ea1ff;--acc2:#5ad19a;--warn:#ffb454;--bad:#ff6b6b;--line:#2a3340;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft Yahei",sans-serif;line-height:1.65;font-size:15px}}
.wrap{{max-width:980px;margin:0 auto;padding:32px 22px 80px}}
header.hero{{border:1px solid var(--line);border-radius:14px;padding:26px 28px;background:linear-gradient(135deg,#16202e,#11161d);margin-bottom:26px}}
h1{{font-size:25px;margin:0 0 8px}} .sub{{color:var(--muted);font-size:13.5px}}
.pill{{display:inline-block;background:var(--panel2);border:1px solid var(--line);color:var(--acc);border-radius:999px;padding:2px 10px;font-size:12px;margin:4px 6px 0 0}}
h2{{font-size:20px;margin:34px 0 12px;padding-left:11px;border-left:4px solid var(--acc)}}
h3{{font-size:15px;margin:18px 0 6px;color:var(--acc2)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}}
th,td{{border:1px solid var(--line);padding:6px 9px;text-align:center}}
th{{background:var(--panel2);color:var(--acc)}} td:first-child,th:first-child{{text-align:left}}
tr:nth-child(even) td{{background:#131922}}
tr.ours td{{background:#10241b!important;color:var(--acc2);font-weight:600}}
code{{background:#0b0f14;color:#9fe6c4;padding:1px 6px;border-radius:5px;font-size:12.5px}}
ul{{margin:8px 0 8px 2px;padding-left:20px}} li{{margin:4px 0}}
.good{{color:var(--acc2)}} .warn{{color:var(--warn)}} .bad{{color:var(--bad)}} .mut{{color:var(--muted)}}
img{{width:100%;border:1px solid var(--line);border-radius:10px;margin:8px 0;background:#fff}}
.note{{border-left:3px solid var(--warn);background:#1d1a12;padding:10px 14px;border-radius:8px;margin:12px 0;font-size:13.5px}}
.star{{border-left:3px solid var(--acc2);background:#10201a;padding:10px 14px;border-radius:8px;margin:12px 0;font-size:13.5px}}
</style></head><body><div class="wrap">
<header class="hero"><h1>UAV-VQA 语义通信 · 对比实验结果报告</h1>
<div class="sub">held-out test set · 6 方法 × 3 信道 × 6 SNR · VisDrone-DET 548 图 · Qwen2-VL-2B 统一后端</div>
<div style="margin-top:10px"><span class="pill">标准四方 + 双上界</span><span class="pill">AWGN/Rayleigh/Rician</span><span class="pill">cliff-effect 成立</span><span class="pill">2026-06-29</span></div></header>

<div class="star"><b>一句话结论：</b>任务感知的自适应语义服务选择（M4）在三种信道、每个 SNR 都压制所有固定基线（传统整图 / DeepSC 模拟 / GO-SG token / naive 定速数字），贴近 Oracle 上界，并省约 34% 带宽；传统定速数字在低 SNR 断崖式崩溃，我们与模拟基线优雅降级。</div>

<h2>1. 招牌图 — cliff-effect（Rayleigh）</h2>
<div class="card"><img src="data:image/png;base64,{imgs['F2']}" alt="F2 cliff">
<p class="mut" style="font-size:12.5px">naive 定速数字（红）在 −5dB 断崖至 0.41（LDPC 解不出→图全丢）；DeepSC 模拟（紫）与我们 M4（绿）优雅降级，M4 全程最高（非 Oracle）。</p></div>

<h2>2. 三面板 Acc-SNR</h2>
<div class="card"><img src="data:image/png;base64,{imgs['F1']}" alt="F1 3-panel"></div>

<h2>3. T1 主结果：准确率 vs SNR（test, all qtypes）</h2>
<div class="card">{t1_html()}</div>

<h2>4. T2 传输代价（bytes/query, avg over SNR）</h2>
<div class="card">{t2_html()}
<div class="note">M2 模拟占满整个时隙（≈300,000 复信道符号，单位非字节）；M4 vs M1 同为数字、字节可比，<b>省约 34%</b>。</div></div>

<h2>5. 机理 — 按问题类型（Rician, 5dB）</h2>
<div class="card">{qtype_html()}
<p>计数任务整图惨败（Qwen 数不清退化航拍图），检测器 token 本就在数物体 → 完胜；存在性反之。M4 逐任务路由取两者之长，这是任务感知语义通信最直观的卖点。</p>
<img src="data:image/png;base64,{imgs['F4']}" alt="F4 qtype"></div>

<h2>6. Pareto（准确率 × 代价）</h2>
<div class="card"><img src="data:image/png;base64,{imgs['F5']}" alt="F5 pareto"></div>

<h2>7. 诚实 caveat</h2>
<div class="card"><ul>
<li>M2 DeepSC 模拟在中低 SNR 不如离散 token（噪声图喂 Qwen 不稳），但低 SNR 无悬崖（−5dB 0.46 &gt; naive 0.41）——帮我们讲透"数字 / 模拟 / 自适应"三方。</li>
<li>计数任务整体偏难（连 Oracle 才 ~0.50），为 VQA 计数固有难度，论文如实说明。</li>
<li>M4 为规则版 LCB 贪心，已 <b>train/test 切分</b>排除过拟合；RL 资源分配（实验组 2）仍待做。</li>
</ul></div>
<p class="mut" style="text-align:center;font-size:12px;margin-top:30px">UAV-VQA 语义通信对比实验结果 · 2026-06-29 · 数据 outputs/reports/comparison_all.csv</p>
</div></body></html>"""

open(f"{OUT}/UAV_VQA_SemCom_Results_Report.html","w").write(html)
print("wrote:")
print(f"  {OUT}/UAV_VQA_SemCom_Results_Report.md")
print(f"  {OUT}/UAV_VQA_SemCom_Results_Report.html ({len(html)//1024} KB)")
