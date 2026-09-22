"""Build the cached-only diagnostic analysis bundle; no model loading or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPLITS = ("train", "validation", "legacy_dev")
NAMES = {"train": "训练480（拟合内）", "validation": "新内部验证120", "legacy_dev": "旧开发120（复用）"}
ENGLISH = {"train": "Train (480)", "validation": "Internal validation (120)", "legacy_dev": "Reused development (120)"}


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def paired_stats(a: list[int], b: list[int]) -> dict:
    if not a or len(a) != len(b):
        raise ValueError("Expected nonempty paired images")
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    gained, lost = int((differences > 0).sum()), int((differences < 0).sum())
    n = gained + lost
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2**n) if n else 1.
    rng = np.random.default_rng(20260922)
    samples = differences[rng.integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
    return {"n": len(a), "gained": gained, "lost": lost, "difference": float(differences.mean()),
            "ci95": np.quantile(samples, [.025, .975]).tolist(), "mcnemar_exact_p": p,
            "bootstrap_replicates": 2000, "unit": "paired image", "scope": "post-run exploratory, conditional on frozen policies"}


def holm(family: dict[str, dict]) -> None:
    previous = 0.
    ordered = sorted(family, key=lambda key: family[key]["mcnemar_exact_p"])
    for index, key in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - index) * family[key]["mcnemar_exact_p"]))
        family[key]["holm_adjusted_p"] = previous


def run(root: Path) -> dict:
    opportunities = read(root / "opportunities.json")
    scores = read(root / "score_metrics.json")
    features = read(root / "feature_diagnostics.json")
    contributions = read(root / "first_layer_contributions.json")
    rows = read(root / "score_diagnostic_rows.json")
    export = read(root / "export_manifest.json")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    metrics = {split: scores["splits"][split]["variants"]["joint9"]["replicates"]["ensemble"] for split in SPLITS}
    paired = {}
    for split in ("validation", "legacy_dev"):
        subset = [row for row in rows if row["split"] == split]
        assert len(subset) == len({row["id"] for row in subset}) == 120
        current = [row["correct"][row["joint_selected_actions"]["ensemble"]["lambda005"]] for row in subset]
        unpenalized = [row["correct"][row["joint_selected_actions"]["ensemble"]["lambda0"]] for row in subset]
        cheapest = [row["correct"][0] for row in subset]
        paired[split] = {"joint_minus_cheapest": paired_stats(current, cheapest),
                         "lambda0_minus_registered": paired_stats(unpenalized, current)}
        holm(paired[split])
        assert sum(current) == opportunities["splits"][split]["deployed_methods"]["joint"]["summary"]["correct"]
    (root / "paired_diagnostic_stats.json").write_text(json.dumps(paired, indent=2) + "\n")

    table = ["|指标|训练480|新内部验证120|旧开发120|", "|---|---:|---:|---:|"]
    definitions = [
        ("九档都答对", lambda s: opportunities["splits"][s]["grid_agreement"]["all_correct"]),
        ("九档都答错", lambda s: opportunities["splits"][s]["grid_agreement"]["all_wrong"]),
        ("九档正确性有差异", lambda s: opportunities["splits"][s]["grid_agreement"]["disagreement"]),
        ("廉价2k/low正确数", lambda s: opportunities["splits"][s]["cells"]["2000_low"]["correct"]),
        ("廉价档错误中可被其他档救回", lambda s: opportunities["splits"][s]["opportunities_from_2000_low"]["counts"]["any_grid_rescuable"]),
        ("联合原策略正确数", lambda s: metrics[s]["selected_lambda005"]["correct"]),
        ("联合原策略救回/误伤", lambda s: f"{metrics[s]['selected_lambda005']['rescued_vs_2000_low']}/{metrics[s]['selected_lambda005']['harmed_vs_2000_low']}"),
        ("九路答案辅助上限（非实测路由）", lambda s: opportunities["splits"][s]["oracles"]["all_families"]["nine_way"]["correct"]),
        ("同图动作排序AUC", lambda s: f"{metrics[s]['within_image_macro_auc_mixed_only']:.4f}"),
        ("取消惩罚的正确数（仅诊断）", lambda s: metrics[s]["selected_lambda0"]["correct"]),
        ("取消惩罚平均图像帧 B", lambda s: f"{metrics[s]['selected_lambda0']['mean_framed_image_bytes']:.2f}"),
    ]
    for name, getter in definitions:
        table.append("|" + name + "|" + "|".join(str(getter(split)) for split in SPLITS) + "|")

    report = ["# EXP-012：为什么联合资源选择没有产生稳定收益？", "",
              "结论：收益空间存在，但当前模型没有可靠学到同图动作的相对收益。低成本图像统计与问题词哈希的尺度失衡、有限的动作分歧样本以及绝对正确率监督，是有证据支持的排查方向；尚不能把任意一项认定为唯一因果。", "",
              "## 本轮做了什么", "",
              "只读取已有600图×9配置和旧开发120图×9配置缓存；对冻结MLP作CPU前向导出，不运行VLM/codec，不更新权重，不改变策略或lambda，不读取test300。既有checkpoint/输入哈希前后不变，3840个既有报告选档完全一致。",
              "比较问题、lambda=0单点机制探针与统计单位见诊断范围文件。结果是事后开发分析，不是独立测试成绩。", "",
              "## 精确诊断表", "", *table, "",
              "## 1. 并非所有错误都能通过更多资源解决", "",
              "训练480图只有84图的九档正确性不同，396图为全对或全错；因此4320条训练动作记录不是4320个独立的选档样本，也不都有区分动作的准确率信号。全对样本仍有学习省资源的价值，不能简单全部删除。",
              "新验证的廉价起点错误38题中，29题在当前九档/接收器下始终错，只有9题可救；旧开发对应32题中20题始终错、12题可救。这里的“不可救”仅限当前动作网格，不说明更好的接收器或编码器也无效。",
              "新验证仅增加码率可救7、仅增加视觉预算可救4，两者交集2，必须同时增加两轴才能救回的为0；旧开发分别6、6、交集1、必须双增1。",
              "两组各有11个廉价档原本正确的样本，在某个资源更高的档位变错。不能把高码率/大视觉预算强制视为总是更优。", "",
              "## 2. 路由没有抓住少数需要升级的题", "",
              "联合原策略在新验证救回1题、误伤1题；旧开发救回2题、误伤2题，净正确数均与固定2k/low相同。联合却传更多字节、处理更多视觉token，因此本次有限样本上的总体结果并未超过这个廉价固定控制；不是逐题输出完全相同，也不是统计等价证明。",
              "答案辅助九路上限为91/120和100/120；遍历固定另一轴后的最强码率单轴上限为89/120和97/120，最强视觉单轴上限为89/120和98/120。九路相对最强单轴的额外正确数仅2题/2题。所有oracle都使用了答案，只衡量当前候选空间，不能写成路由准确率。", "",
              "## 3. 主要评分问题：同图内排序泛化不足", "",
              "只在九档有正确也有错误的图上，计算预测把正确动作排在错误动作之前的AUC。训练为0.641，新验证0.495，旧开发0.558；0.5是随机排序基准，验证/旧开发区间都包含0.5，不能声称可靠分辨。样本分别84/20/23图，区间较宽。",
              "验证集joint的跨图/动作混合AUC也仅0.441，因此不能说它已经学好了题目难度。问题单独输入控制的混合AUC约0.755，但同图AUC只有0.429，说明“预测哪些题容易”与“判断哪种资源动作更适合这题”是不同目标。",
              "尤其4k/low相对2k/low，joint集成预测平均正确概率低约3.1个百分点，而实际正确率在训练/新验证/旧开发分别高1.67/2.50/0.83个百分点；该有用中间档在三组均未被joint集成选择。这是动作评分偏差的直接现象，不是单纯通信惩罚过大。", "",
              "## 4. 取消惩罚不能单独修复", "",
              f"固定概率只把lambda从0.05切到0，不做扫描和模型选择。新验证通信量从2147 B升至7494 B，正确数82→81；旧开发从2097 B升至{metrics['legacy_dev']['selected_lambda0']['mean_framed_image_bytes']:.0f} B，正确数88→92，仍未达到4k/medium的93且多传约{100*(metrics['legacy_dev']['selected_lambda0']['mean_framed_image_bytes']/opportunities['splits']['legacy_dev']['cells']['4000_medium']['mean_image_bytes']-1):.1f}%的字节。该机制对照不能被替换为正式部署策略。",
              "新验证最终漏救的8个廉价档可救题，4个在无惩罚排序时就错，4个被惩罚从正确改错；旧开发漏救10个分别为6和4。惩罚也会纠正部分错误，所以不能只数“惩罚改坏”而忽略改好与资源成本。", "",
              "## 5. 输入与训练的具体风险", "",
              "冻结训练集标准化器已独立重算一致，未发现拟合验证数据的问题。问题块L2固定约1，图像标准化统计块平均L2约8.8–9.0，约98.5%的拼接输入平方范数来自图像块。首层W×x分解后，图像/问题贡献L2比仍约9.6–11.2。它证明尺度失衡传到了首层，但不是最终决策重要性或失败的因果证明；没有擅自重归一化后重训。",
              "joint三种子最佳验证轮次1/1/2，实际均训练10轮；训练BCE从约0.66降到0.39–0.41，验证BCE升到0.69–0.71。延长轮数尚无依据。三种子选档不全一致约95%–96%，但逐题正确性差异远小于此，因为许多动作答题效果相同；不能把选档抖动直接等同答案抖动。", "",
              "## 下一步建议（本轮未执行）", "",
              "暂停九档联合方案的扩量、SNR和正式测试。若继续，先做仅用现有缓存的小型2×2机制检验：原始/分组尺度均衡输入 × 绝对正确率/相对廉价基线的动作增益目标；保持其他训练条件与对照一致。它用于区分尺度与训练目标的作用，不预设能成功，不同时更换VLM或压缩器。",
              "验收应直接看：相对固定廉价控制救回多少题、误伤多少题、增加多少字节，以及能否超过学习式单轴和强固定。优先保留简单码率选择作为比较对象；联合控制只有带来稳定额外收益时才值得增加复杂性。当前内部验证和旧开发已反复用于诊断，不能作为后续独立确认性证据。", "",
              "## 图与证据", "", "图1区分实际固定/学习式正确数和答案辅助上限；图2检验同图动作排序而不是跨题难度预测。区间与配对统计见 [stats-appendix.md](stats-appendix.md)，图注/限制见 [figure-catalog.md](figure-catalog.md)。",
              "本轮仅本地汇总报告与诊断代码；原论文未改、原EXP-012完成结果未覆盖。全过程时延/能耗仍未得到。"]
    (root / "analysis-report.md").write_text("\n".join(report) + "\n")

    appendix = ["# 诊断统计附录", "", "所有指标从已冻结模型和缓存生成。train是拟合内，validation选过checkpoint与固定轴，legacy反复开发；推断只做探索性描述。不得把九档或三个seed作为九倍/三倍独立图像样本。", "",
                "## 同图动作AUC与Brier", "", "AUC仅使用候选集中同时存在正确/错误的图，先在每图比较所有正负动作对，概率相等记0.5，再跨图等权。bootstrap重采样图像2000次，不重采样独立动作。Brier先对每图候选取均值再跨图。无正态假设，无基于pooled动作数的显著性检验。", "",
                "不同单轴family的mixed图子集不同，不直接按其AUC高低比较优劣。joint与question-only采用相同九动作和mixed图，但各自区间不是二者差值检验。",
                "|拆分|模型|有分歧图数|同图AUC [95%区间]|跨种子AUC均值±SD|描述性pooled AUC|Brier|", "|---|---|---:|---|---|---:|---:|"]
    for split in SPLITS:
        for variant in ("joint9", "question_only9", "rate_at_low", "compute_at_4000"):
            v = scores["splits"][split]["variants"][variant]
            e = v["replicates"]["ensemble"]
            lo, hi = e["within_image_auc_image_bootstrap"]["image_bootstrap_95_percentile_interval"]
            seeds = [v["replicates"][str(seed)]["within_image_macro_auc_mixed_only"] for seed in (7, 17, 27)]
            appendix.append(f"|{NAMES[split]}|{variant}|{e['mixed_candidate_images']}|{e['within_image_macro_auc_mixed_only']:.4f} [{lo:.4f}, {hi:.4f}]|{statistics.mean(seeds):.4f}±{statistics.stdev(seeds):.4f}|{e['pooled_all_candidate_heads_auc_descriptive']:.4f}|{e['brier_all_candidate_heads_image_mean']:.6f}|")
    appendix += ["", "## 配对正确数检验", "", "比较方向为表中前者减后者，单位图像；二元配对使用双侧精确McNemar，无正态/方差齐性要求。每拆分的两项比较使用Holm校正。效应量为准确率百分点差，区间为2000次配对图像bootstrap。区间未作多重校正，且不涵盖训练/选择不确定性。", "",
                 "|拆分|比较|修复/损失|差 pp [95%区间]|精确 p|Holm p|", "|---|---|---:|---|---:|---:|"]
    for split, family in paired.items():
        for contrast, value in family.items():
            lo, hi = value["ci95"]
            appendix.append(f"|{NAMES[split]}|{contrast}|{value['gained']}/{value['lost']}|{100*value['difference']:.3f} [{100*lo:.3f}, {100*hi:.3f}]|{value['mcnemar_exact_p']:.6f}|{value['holm_adjusted_p']:.6f}|")
    appendix += ["", "## 资源惩罚机制分解", "", "|拆分|lambda0正确|原策略正确|取消惩罚 B|原策略 B|原惩罚改好/改坏（全部图）|", "|---|---:|---:|---:|---:|---:|"]
    for split, m in metrics.items():
        d = m["penalty_decomposition"]["all_images"]
        appendix.append(f"|{NAMES[split]}|{m['selected_lambda0']['correct']}|{m['selected_lambda005']['correct']}|{m['selected_lambda0']['mean_framed_image_bytes']:.2f}|{m['selected_lambda005']['mean_framed_image_bytes']:.2f}|{d['penalty_changed_wrong_to_correct']}/{d['penalty_changed_correct_to_wrong']}|")
    appendix += ["", "## 完整性与不能推断的内容", "", "冻结24个checkpoint及输入哈希在导出前后保持一致，四个模型族的12个已选权重只运行CPU前向，3840次既有选档全部复现。数据均为480+120+120图像身份不重叠的既有集合。",
                 "没有无线交付率/能耗新结果。没有重新拟合、搜lambda、重新选择checkpoint或正式测试；跨种子SD只描述该训练过程，不是独立数据集的方差。",
                 "特征范数和首层线性信号比只是数值尺度，不证明有效语义信息或因果贡献。oracle完全依赖答案；最强单轴另一轴在各split事后取优只用于构成更强上限，不更改已冻结的单轴基线。",
                 "导出概率与后加首层分析对应不同诊断脚本版本，分别在各自manifest留hash；旧概率导出源码另存export_scores_forward_frozen.py。原训练源码/模型不变，不声称全部诊断输出由同一脚本版本生成。"]
    (root / "stats-appendix.md").write_text("\n".join(appendix) + "\n")

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, split in zip(axes, ("validation", "legacy_dev")):
        op = opportunities["splits"][split]
        counts = [op["cells"]["2000_low"]["correct"], op["deployed_methods"]["joint"]["summary"]["correct"],
                  op["deployed_methods"]["rate_only"]["summary"]["correct"],
                  op["oracles"]["strongest_fixed_other_axis"]["rate_only"]["summary"]["correct"],
                  op["oracles"]["all_families"]["nine_way"]["correct"]]
        bars = ax.bar(range(5), counts, color=["#999999", "#0072B2", "#009E73", "#F0E442", "#E69F00"], edgecolor="black", linewidth=.5)
        for index, (bar, count) in enumerate(zip(bars, counts)):
            if index >= 3:
                bar.set_hatch("///")
            ax.text(bar.get_x() + bar.get_width()/2, count+1, str(count), ha="center", fontsize=9)
        ax.set_xticks(range(5), ["Fixed\n2k/low", "Learned\njoint", "Learned\nrate only", "Oracle\nrate only", "Oracle\nnine-way"], fontsize=8)
        ax.set(title=ENGLISH[split], ylabel="Correct answers / 120", ylim=(0, 120))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"figure-01-opportunity-vs-realized.{extension}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for offset, variant, color, marker, title in ((-.10, "joint9", "#0072B2", "o", "Joint image + question"),
                                               (.10, "question_only9", "#D55E00", "s", "Question-only control")):
        values, lower, upper = [], [], []
        for split in SPLITS:
            m = scores["splits"][split]["variants"][variant]["replicates"]["ensemble"]
            value = m["within_image_macro_auc_mixed_only"]
            lo, hi = m["within_image_auc_image_bootstrap"]["image_bootstrap_95_percentile_interval"]
            values.append(value)
            lower.append(value-lo)
            upper.append(hi-value)
        ax.errorbar(np.arange(3)+offset, values, yerr=[lower, upper], color=color, marker=marker,
                    ls="none", capsize=4, label=title)
    ax.axhline(.5, color=".4", ls="--", lw=1)
    ax.set_xticks(range(3), ["Train\n84 mixed images", "Internal validation\n20 mixed images", "Reused development\n23 mixed images"])
    ax.set(ylabel="Within-image correct-action ranking AUC", ylim=(0, 1), xlim=(-.45, 2.45))
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"figure-02-within-image-ranking.{extension}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    (root / "figure-catalog.md").write_text("""# 诊断图目录

## figure-01-opportunity-vs-realized.pdf / .png

- 目的：区分廉价固定、已训练路由的实际表现与答案辅助的候选空间。
- 来源：opportunities.json，两组各120图，正确数精确计数，无误差条；不声称正式测试或显著优胜。
- 图注：实心柱为固定/冻结学习策略，斜线柱为使用答案的事后oracle；码率oracle遍历另一轴取最强者。图像帧均含1 B档位，问题传输不计入本轮预算。
- 观察：joint与2k/low总正确数相同，九路oracle仍有空间；单轴oracle已接近九路。
- 含义：应先证明识别资源收益的能力，不能把答案辅助上限当成方法结果，也不应默认需要九路联合。
- 检查：柱轴从0起、分母一致、oracle显式区别；同正确数不等于逐题相同或已证非劣。

## figure-02-within-image-ranking.pdf / .png

- 目的：直接检查同图内动作选择，而不是跨题总体难度预测。
- 来源：score_metrics.json。每图AUC按正负动作对计算，ties=0.5，再对mixed图等权平均；仅84/20/23图有正确性差异可定义AUC。
- 图注：点为三个种子概率集成的同图AUC，误差线为2000次按图bootstrap条件95%区间；虚线0.5为随机排序基准。训练是拟合内，验证已用于模型选择，旧开发复用，不合并推断。
- 观察：joint新验证约0.495、旧开发约0.558，未显现稳定可泛化排序；question-only也未跨split稳定领先。
- 含义：训练目标和输入表示需要受控诊断，不能用较高的pooled AUC代替选档能力。
- 检查：无全对/全错图伪造AUC，不把每个action当独立样本；三种子不是三套独立数据。
""")
    files = ["opportunities.json", "score_metrics.json", "score_diagnostic_rows.json", "feature_diagnostics.json", "first_layer_contributions.json", "export_manifest.json"]
    (root / "analysis_source_hashes.json").write_text(json.dumps({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}, indent=2) + "\n")
    assert features["scaler_reproduced_from_train_only"] and contributions["source_unchanged"]
    return {"reports": 3, "figures": 2, "paired_tests": 4, "model_training": False, "source_export_schema": export["schema"] if "schema" in export else "see manifest"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root)))
