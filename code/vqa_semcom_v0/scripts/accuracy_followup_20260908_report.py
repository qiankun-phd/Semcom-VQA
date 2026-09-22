#!/usr/bin/env python3
"""Decision-oriented internal report from completed, separately verified analyses."""
import argparse
import json
from pathlib import Path

NAME = '2026-09-08--accuracy-followup--r00--matched-baseline-vlm-validation.md'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    read = lambda p: json.loads((root / p).read_text())
    assert read('phase_ab/status.json')['state'] == read('phase_c/status.json')['state'] == 'COMPLETE'
    assert not (root / NAME).exists()
    ab = read('analysis_ab_v2/comparisons.json')
    c = read('analysis_c/results.json')
    stats = read('analysis_c/exploratory_paired_statistics.json')
    v3, v7 = c['3b_nf4']['overall'], c['7b_nf4']['overall']
    differences = {qt: 100*(c['7b_nf4']['by_type'][qt]['accuracy']-v['accuracy']) for qt, v in c['3b_nf4']['by_type'].items()}
    low, high = min(differences, key=differences.get), max(differences, key=differences.get)
    snr_range = {m: [100*min(v['accuracy'] for v in c[m]['by_snr'].values()),
                     100*max(v['accuracy'] for v in c[m]['by_snr'].values())] for m in ('3b_nf4', '7b_nf4')}
    lines = ['---', 'type: results-report', 'date: 2026-09-08', 'experiment_line: accuracy-followup',
             'round: 0', 'purpose: matched-baseline-vlm-validation', 'status: complete',
             'source_artifacts:', '  - analysis_ab_v2/analysis-report.md', '  - analysis_ab_v2/stats-appendix.md',
             '  - analysis_c/analysis-report.md', '  - analysis_c/exploratory_paired_statistics.json',
             'linked_experiments: []', 'linked_results: []', '---', '',
             '# Accuracy Follow-up / Round 0 / Matched Baseline and VLM Validation / 2026-09-08', '',
             'r00 是待统一项目轮次编号的显式占位，不代表第零次实验。报告仅写入本轮授权目录；'
             '未进行 Obsidian 回写、论文修改或旧结果覆盖。results-report 的主文件可用，但所引用的模板文件缺失，'
             '因此按其主文件十项结构整理；严格统计来自已完成的 results-analysis 产物。', '',
             '## 1. Executive Summary', '',
             '相同增强输入下，Qwen2 的线性模型已达到网络水平，不能把此前全部增益归因于网络非线性。'
             'Qwen2.5 的网络增益较小，Smol 较大，但逐类型仍有负结果。统一验证准确率约束并不保证测试集达到同一准确率；'
             '必须同时报告实际测试准确率与图像使用率。', '',
             f'完整验证集 3B NF4 为 {v3["accuracy"]*100:.3f}%，7B NF4 为 {v7["accuracy"]*100:.3f}%，'
             f'差值 {stats["effect_pp"]:+.3f} 个百分点。该对照包含全部 2646 个公共验证键，而非只复述原 120 条小试。', '',
             '## 2. Experiment Identity and Decision Context', '',
             '本轮回答三个决策问题：增强输入的收益是否仍超过一个公平的线性对照；相同验证准确率目标下如何权衡图像使用率；'
             '7B 相对 3B 的小试观察能否在完整公共验证集保持。输入、分支判分与划分冻结，不扩展非线性架构搜索，'
             '不把 7B 新答案混进已训练路由器。', '',
             '## 3. Setup and Evaluation Protocol', '',
             '- A/B：train/validation/test 为 9150/2646/2808 个决策键；Qwen2 使用 combined52，Qwen2.5/Smol 使用 semantics42，'
             '训练拟合的标准化不变。新拟合 6 个双逻辑模型与 2 个固定 ridge 模型，复用 30 个网络种子模型和 3 个旧18维线性模型。',
             '- 历史 GD400 与收敛 L-BFGS 求解同一个固定 L2 逻辑目标，全部保留；ridge 使用相同优势监督，'
             '但不宣称与网络的参数正则化完全相同。没有验证/测试驱动的正则搜索。',
             '- B：26 个预声明无量纲 κ，image iff score>κ；共同验证目标 74%。原各自零价减1pp规则仅作次要结果。'
             'Qwen2 的物理 λ 另用原26点与经过 JPEG/payload 核对的旧核算模型；其他接收器不借用 Joule 数值。',
             '- C：101 张验证图像、2646 键、六档 Rician SNR；presence/counting 仅23张图，另外三类101张。'
             'NF4 double quantization、BF16 compute、同 JPEG/prompt/实际 image grid、greedy24 tokens。'
             '每模型精确复用120条，只补推2526条。无7B测试集推理或路由训练。',
             '- 既有 benchmark 已用于开发。图像级划分不代表视频序列完全独立，不称 pristine holdout。', '',
             '## 4. Main Findings', '',
             '| Receiver | Network mean ± SD (%) | GD400 (%) | Converged logistic (%) | Ridge (%) |',
             '|---|---:|---:|---:|---:|']
    for r, methods in ab.items():
        n = methods['enhanced_network']['unpriced_accuracy']
        values = [f'{100*n["mean"]:.3f} ± {100*n["sd"]:.3f}']
        for name in ('logistic_gd400', 'logistic_converged', 'ridge_advantage'):
            values.append(f'{100*methods[name]["unpriced_accuracy"]["mean"]:.3f}' if name in methods else 'N/A')
        lines.append(f'| {r} | ' + ' | '.join(values) + ' |')
    lines += ['', '[A/B 完整表](analysis_ab_v2/analysis-report.md)还列出每种题型、每个固定规则和全部选价结果。'
              'Qwen2.5 的网络在共同验证约束选点后，测试为68.031%、图像使用1.243%；收敛线性为69.409%、3.419%。'
              '这是不同资源/性能结果，不能仅凭使用率更低就宣称支配。', '',
              '| Type | n | 3B NF4 (%) | 7B NF4 (%) | Difference (pp) |', '|---|---:|---:|---:|---:|']
    for qt in ('presence', 'counting', 'comparison', 'co_presence', 'threshold'):
        x, y = c['3b_nf4']['by_type'][qt], c['7b_nf4']['by_type'][qt]
        lines.append(f'| {qt} | {x["n"]} | {100*x["accuracy"]:.3f} | {100*y["accuracy"]:.3f} | {100*(y["accuracy"]-x["accuracy"]):+.3f} |')
    ci = stats['bootstrap_95_percentile_pp']
    lines += ['', '## 5. Statistical Validation', '',
              'A/B 将每个决策上的10种子正确性取平均，再按104张测试图像聚类；8个网络对新线性比较做 Holm 校正。'
              'Qwen2和Qwen2.5区间跨零；Smol 的探索性结果更强，但不据此保证未来测试收益或证明其他方法等效。'
              '所有效应量、区间、原始/调整p值见[统计附录](analysis_ab_v2/stats-appendix.md)。', '',
              f'C 只有一个总体配对对比：101图像聚类、10000次bootstrap/sign flip，差值 '
              f'{stats["effect_pp"]:+.3f}pp，95% percentile区间[{ci[0]:+.3f}, {ci[1]:+.3f}]pp，'
              f'p={stats["two_sided_sign_flip_p"]:.5f}。不对题型/SNR表添加未校正显著性星号。'
              '这些统计依赖聚类独立/可交换假设，邻近视频帧相关以及此前开发暴露会限制确认性解释。', '',
              '## 6. Figure-by-Figure Interpretation', '',
              '1. [同输入准确率](analysis_ab_v2/figures/matched_input_accuracy.pdf)：用于区分输入与非线性。'
              'Qwen2 的点基本重合，Smol 分离较大；支持接收器依赖的增益，决定保留全部线性对照。误差线为网络10种子SD。',
              '2. [验证使用率曲线](analysis_ab_v2/figures/validation_accuracy_image_use.pdf)：检验统一74%目标的选点。'
              '标记来自验证选择，所有26点均展示；支持在共同准则下比较，不支持各自相对目标的优势声明。',
              '3. [测试使用率曲线](analysis_ab_v2/figures/test_accuracy_image_use.pdf)：检验冻结策略的实际泛化。'
              '测试未维持74%且部分网络/线性互有权衡，因此报告完整曲线与实际二元指标，不把验证门限当成测试保证。'
              '网络粗线是十种子均值，不是部署集成。',
              '4. [Qwen2 原核算能耗](analysis_ab_v2/figures/qwen2_original_energy_accounting.pdf)：将匹配payload用于旧能耗模型。'
              '冻结选点下网络70.082%、2.8755J，收敛线性69.801%、5.4154J；只支持该有限网格/核算边界中的观察，'
              '不包含新特征和路由器的机载实测成本，也不推广至其他VLM。',
              '5. [完整VLM逐类型](analysis_c/figures/full_validation_by_type.pdf)：防止 pooled 准确率掩盖题型变化。'
              f'最小差值为{low} {differences[low]:+.3f}pp，最大为{high} {differences[high]:+.3f}pp。'
              '所有正负变化与不同分母一起保留；是否继续7B方向应同时考虑计数、共现以及成本。',
              '6. [完整VLM逐SNR](analysis_c/figures/full_validation_by_snr.pdf)：展示六个冻结SNR上的直接结果。'
              f'3B准确率范围为{snr_range["3b_nf4"][0]:.3f}–{snr_range["3b_nf4"][1]:.3f}%，'
              f'7B为{snr_range["7b_nf4"][0]:.3f}–{snr_range["7b_nf4"][1]:.3f}%。'
              '不平滑、不挑有利SNR，也不据此推断未评估SNR泛化。', '',
              '## 7. Failure Cases / Negative Results / Limitations', '',
              '- Qwen2网络presence为76.330%，低于收敛线性82.372%；Smol网络counting为41.859%，低于收敛线性45.833%。'
              '总体提升不等于逐类型提升。',
              '- 固定GD400未达到梯度收敛阈值，但收敛求解也不自动提高测试准确率，因此两者都保留。',
              '- 共同验证约束未在测试维持相同水平。任何部署可靠性保证仍需独立评估，不由本轮点估计给出。',
              '- 原120条是嵌套子集且问题权重不同，不是独立复现；历史BF16缓存与新NF4同时改变精度和采样flag。',
              '- 原counting判分有max(1,round(.1*GT))容差，不是exact match。原yes/no解析、首个整数抽取等规则均未修复。'
              '原prompt只明示presence/counting格式，另外三类格式提示不足的问题列入待办，不在本轮混改。',
              '- NVML power.draw不可用但Power Samples存在；缺少本轮连续相位记录和新idle baseline，因此不给3B/7B编造J值。'
              'CPU打分是同batch摊销吞吐，非在线单样本或机载能耗。', '',
              '## 8. What Changed Our Belief', '',
              '此前“增强网络提升”需要拆成“增强发送前信息”和“利用这些信息的非线性收益”。Qwen2主要支持前者，'
              'Smol更支持后者，Qwen2.5仍较弱。资源选择还受到验证—测试差异影响；准确率最高与某个有限成本约束下合适并非同一问题。'
              f'7B完整验证的实际差值是{stats["effect_pp"]:+.3f}pp，应替代120条小试的总体印象，同时保留其逐类型不利结果。', '',
              '## 9. Next Actions', '',
              '- 保留18维与增强输入线性对照；不再扩大网络容量搜索来追逐当前测试分数。',
              '- 将“输入改善”和“非线性收益”分开陈述，图表同时保留原始准确率与共同验证约束下的使用率。',
              '- 等待独立detector/decoder子任务的受控结果；它不改变本轮A/B/C标签、协议或统计。',
              '- 7B正式测试、7B路由训练或prompt修复需要另立冻结协议并由主任务决定，本轮不自动启动。', '',
              '## 10. Artifact and Reproducibility Index', '',
              '- [A/B分析与完整逐类型表](analysis_ab_v2/analysis-report.md)、[统计](analysis_ab_v2/stats-appendix.md)、'
              '[图索引](analysis_ab_v2/figure-catalog.md)、phase_ab/validation_selection.json。',
              '- [C分析](analysis_c/analysis-report.md)、analysis_c/results.json、format_audit.json、'
              'exploratory_paired_statistics.json及phase_c逐条答案、source/checkpoint/reuse哈希。',
              '- [CPU成本](cpu_cost/report.md)、[固定分支端点](fixed_endpoints/report.md)。',
              '- code/、code_dependencies/、protocol-plan.md、unit_tests.log、manifest.json/manifest.sha256。'
              '全部完成后核对本地与远端manifest；不把仍运行的任务标记为完成。', '',
              '执行决策：本轮分析与证据保留；停止本轮新增拟合/推理。把上面的边界与待办交回主任务，不自动改论文。']
    (root / NAME).write_text('\n'.join(lines) + '\n')
    print(root / NAME, flush=True)


if __name__ == '__main__':
    main()
