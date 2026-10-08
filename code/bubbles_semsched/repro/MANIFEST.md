# 结果文件 → 生成命令 → 图表（由 tools/build.py 生成，勿手改）

“核对”一栏：checked = 记录的命令与结果文件里存的元数据（策略、信道数、速率、缓冲、流量种子、召回要求 / 训练种子）一致；unchecked = 结果文件不含可比对的元数据；rebuilt = 当时是手敲的命令，按结果文件里的配置重建。

| 结果文件 | res/ 下目录 | 复现脚本 | 当时运行的脚本 | 服务器 | 核对 | 用于 |
|---|---|---|---|---|---|---|
| `sppoX_fast5_s0.json` | pass1 | `exp/10_train_proposed.sh` | `run_old_pass3.sh` | rented 3090 | checked | conv, conv2 |
| `sppoX_fast5_s1.json` | pass1 | `exp/10_train_proposed.sh` | `run_old_pass3.sh` | rented 3090 | checked | conv, conv2 |
| `sppoX_fast5_s2.json` | pass1 | `exp/10_train_proposed.sh` | `run_old_pass3.sh` | rented 3090 | checked | conv, conv2 |
| `sppoX_fast5_s3.json` | pass1 | `exp/10_train_proposed.sh` | `run_old_pass3.sh` | rented 3090 | checked | conv, conv2 |
| `sppoX_fast5_s4.json` | pass1 | `exp/10_train_proposed.sh` | `run_old_pass3.sh` | rented 3090 | checked | conv, conv2 |
| `sppoX_fast_s0.json` | pass1_182 | `exp/10_train_proposed.sh` | `run_182_pass2.sh` | server 182 | checked | 训练产物（检查点被评估用） |
| `sppoX_fast_s1.json` | pass1_182 | `exp/10_train_proposed.sh` | `run_182_pass3.sh` | server 182 | checked | 训练产物（检查点被评估用） |
| `sppoX_fast_s2.json` | pass1_182 | `exp/10_train_proposed.sh` | `run_182_pass3.sh` | server 182 | checked | 训练产物（检查点被评估用） |
| `sppoX_fast_s3.json` | pass1_182 | `exp/10_train_proposed.sh` | `run_182_pass3.sh` | server 182 | checked | 训练产物（检查点被评估用） |
| `sppoX_fast_s4.json` | pass1_182 | `exp/10_train_proposed.sh` | `run_182_pass3.sh` | server 182 | checked | 训练产物（检查点被评估用） |
| `mrl5k_B_M4_s0.json` | gate | `exp/11_train_baselines.sh` | `(by hand; rebuilt from the command of seeds 1-4, which states it is the same)` | — | rebuilt | conv, conv2 |
| `mrl5k_B_M4_s1.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_B_M4_s2.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_B_M4_s3.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_B_M4_s4.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_d3qn_M4_s0.json` | gate | `exp/11_train_baselines.sh` | `run_pass1_train.sh` | rented 3090 | checked | conv, conv2 |
| `mrl5k_d3qn_M4_s1.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_d3qn_M4_s2.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_d3qn_M4_s3.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_d3qn_M4_s4.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_td3_M4_s0.json` | gate | `exp/11_train_baselines.sh` | `run_pass1_train.sh` | rented 3090 | checked | conv, conv2 |
| `mrl5k_td3_M4_s1.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_td3_M4_s2.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_td3_M4_s3.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `mrl5k_td3_M4_s4.json` | gate | `exp/11_train_baselines.sh` | `run_182_base.sh` | server 182 | checked | conv, conv2 |
| `sppoAb_pfix_s0.json` | final | `exp/12_train_variants.sh` | `run_final.sh` | AutoDL | checked | 训练产物（检查点被评估用） |
| `sppoAb_pfix_s1.json` | final | `exp/12_train_variants.sh` | `run_final.sh` | AutoDL | checked | 训练产物（检查点被评估用） |
| `sppoAb_pfix_s2.json` | final | `exp/12_train_variants.sh` | `run_final.sh` | AutoDL | checked | 训练产物（检查点被评估用） |
| `sppoAb_vfix_s0.json` | final | `exp/12_train_variants.sh` | `run_final.sh` | AutoDL | checked | 训练产物（检查点被评估用） |
| `sppoAb_vfix_s1.json` | final | `exp/12_train_variants.sh` | `run_final.sh` | AutoDL | checked | 训练产物（检查点被评估用） |
| `sppoAb_vfix_s2.json` | final | `exp/12_train_variants.sh` | `run_final.sh` | AutoDL | checked | 训练产物（检查点被评估用） |
| `sppoF_p1_s0.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p1_s1.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p1_s2.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p1_s3.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p1_s4.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p2_s0.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p2_s1.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p2_s2.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p2_s3.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoF_p2_s4.json` | gate | `exp/12_train_variants.sh` | `run_sppo_final5.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoL1_s0.json` | final | `exp/12_train_variants.sh` | `run_old_narr.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoL1_s1.json` | final | `exp/12_train_variants.sh` | `run_old_narr.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoL1_s2.json` | final | `exp/12_train_variants.sh` | `run_old_narr.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoL2_s0.json` | final | `exp/12_train_variants.sh` | `run_old_lev2.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoL2_s1.json` | final | `exp/12_train_variants.sh` | `run_old_lev2.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `sppoL2_s2.json` | final | `exp/12_train_variants.sh` | `run_old_lev2.sh` | rented 3090 | checked | 训练产物（检查点被评估用） |
| `val_refs.json` | pass1 | `exp/20_validation_refs.sh` | `run_pass1_eval.sh` | rented 3090 | unchecked | conv, conv2 |
| `val_refs_exh.json` | pass1 | `exp/20_validation_refs.sh` | `run_182_pass2.sh` | server 182 | unchecked | conv, conv2 |
| `landscape_pv.json` | pass1 | `exp/21_grid_search.sh` | `run_pass1_eval.sh` | rented 3090 | checked | land |
| `landscape_pv_M4_48.json` | pass1 | `exp/21_grid_search.sh` | `run_pass2_eval.sh` | rented 3090 | checked | trade |
| `base48_B_M4.json` | pass1 | `exp/30_dev_traffic.sh` | `run_base_eval.sh` | rented 3090 | checked | 表 |
| `base48_d3qn_M4.json` | pass1 | `exp/30_dev_traffic.sh` | `run_base_eval.sh` | rented 3090 | checked | 表 |
| `base48_rules_M4.json` | pass1 | `exp/30_dev_traffic.sh` | `run_base_eval.sh` | rented 3090 | checked | 表 |
| `base48_td3_M4.json` | pass1 | `exp/30_dev_traffic.sh` | `run_base_eval.sh` | rented 3090 | checked | 表 |
| `main_M4_48seeds.json` | main | `exp/30_dev_traffic.sh` | `(by hand; rebuilt from the recorded sibling main_M5_48seeds and the metadata stored in the result)` | — | rebuilt | trade |
| `main_hevc48.json` | pass1 | `exp/30_dev_traffic.sh` | `run_base_more.sh` | rented 3090 | checked | cap |
| `raw48_B.json` | extra | `exp/30_dev_traffic.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `raw48_d3qn.json` | extra | `exp/30_dev_traffic.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `raw48_main.json` | extra | `exp/30_dev_traffic.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `raw48_td3.json` | extra | `exp/30_dev_traffic.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `sysF_M4_48seeds.json` | pass1_182 | `exp/30_dev_traffic.sh` | `run_old_pass3.sh` | rented 3090 | checked | trade |
| `tradeoff2_mg0.004.json` | pass1 | `exp/30_dev_traffic.sh` | `run_new_pass1.sh` | AutoDL | checked | trade |
| `tradeoff2_mg0.006.json` | pass1 | `exp/30_dev_traffic.sh` | `run_new_pass1.sh` | AutoDL | checked | trade |
| `tradeoff2_mg0.010.json` | pass1 | `exp/30_dev_traffic.sh` | `run_new_pass1.sh` | AutoDL | checked | trade |
| `tradeoff2_mg0.012.json` | pass1 | `exp/30_dev_traffic.sh` | `run_new_pass1.sh` | AutoDL | checked | trade |
| `tradeoff3_mg0.004.json` | pass1 | `exp/30_dev_traffic.sh` | `run_new_pass5.sh` | AutoDL | checked | trade |
| `tradeoff3_mg0.006.json` | pass1 | `exp/30_dev_traffic.sh` | `run_new_pass5.sh` | AutoDL | checked | trade |
| `tradeoff3_mg0.010.json` | pass1 | `exp/30_dev_traffic.sh` | `run_new_pass5.sh` | AutoDL | checked | trade |
| `tradeoff3_mg0.012.json` | pass1 | `exp/30_dev_traffic.sh` | `run_new_pass5.sh` | AutoDL | checked | trade |
| `tradeoff_mg0.004.json` | pass1 | `exp/30_dev_traffic.sh` | `run_pass1_eval.sh` | rented 3090 | checked | trade |
| `tradeoff_mg0.006.json` | pass1 | `exp/30_dev_traffic.sh` | `run_pass1_eval.sh` | rented 3090 | checked | trade |
| `tradeoff_mg0.010.json` | pass1 | `exp/30_dev_traffic.sh` | `run_pass1_eval.sh` | rented 3090 | checked | trade |
| `tradeoff_mg0.012.json` | pass1 | `exp/30_dev_traffic.sh` | `run_pass1_eval.sh` | rented 3090 | checked | trade |
| `tradeoff_mg0.014.json` | pass1 | `exp/30_dev_traffic.sh` | `run_pass1_eval.sh` | rented 3090 | checked | trade |
| `final48_M3_B.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `final48_M3_B_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M3_d3qn.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `final48_M3_d3qn_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M3_main.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `final48_M3_prop182.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M3_td3.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `final48_M3_td3_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M4_B.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cdf, sys |
| `final48_M4_B_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final.sh` | server 182 | checked | cap |
| `final48_M4_d3qn.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cdf, sys |
| `final48_M4_d3qn_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final.sh` | server 182 | checked | cap |
| `final48_M4_main.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cdf, sys |
| `final48_M4_prop182.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M4_td3.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cdf, sys |
| `final48_M4_td3_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final.sh` | server 182 | checked | cap |
| `final48_M5_B.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `final48_M5_B_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M5_d3qn.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `final48_M5_d3qn_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M5_main.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `final48_M5_prop182.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M5_td3.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `final48_M5_td3_s1to4.json` | final_182 | `exp/31_test_traffic.sh` | `run_182_final2.sh` | server 182 | checked | cap |
| `final48_M6to10_rules.json` | final | `exp/31_test_traffic.sh` | `run_final.sh` | AutoDL | checked | cap |
| `low48_M3_B.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M3_d3qn.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M3_rules.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M3_td3.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M4_B.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M4_d3qn.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M4_rules.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M4_td3.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M5_B.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M5_d3qn.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M5_rules.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48_M5_td3.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48a_M4_B.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48a_M4_d3qn.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `low48a_M4_td3.json` | final | `exp/31_test_traffic.sh` | `run_old_low.sh` | rented 3090 | checked | cap |
| `ablate48_M4.json` | final | `exp/32_ablation_levels.sh` | `run_final.sh` | AutoDL | checked | 表 |
| `final48_M3_lev1.json` | final | `exp/32_ablation_levels.sh` | `run_old_narr.sh` | rented 3090 | checked | 表 |
| `final48_M3_lev2.json` | final | `exp/32_ablation_levels.sh` | `run_old_lev2.sh` | rented 3090 | checked | 表 |
| `final48_M4_lev1.json` | final | `exp/32_ablation_levels.sh` | `run_old_narr.sh` | rented 3090 | checked | 表 |
| `final48_M4_lev1best.json` | final | `exp/32_ablation_levels.sh` | `run_old_lev2.sh` | rented 3090 | checked | 表 |
| `final48_M4_lev2.json` | final | `exp/32_ablation_levels.sh` | `run_old_lev2.sh` | rented 3090 | checked | 表 |
| `final48_M4_lev4best.json` | final | `exp/32_ablation_levels.sh` | `run_old_lev2.sh` | rented 3090 | checked | 表 |
| `final48_M5_lev1.json` | final | `exp/32_ablation_levels.sh` | `run_old_narr.sh` | rented 3090 | checked | 表 |
| `lev1_rules.json` | extra | `exp/32_ablation_levels.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `lev2_rules.json` | extra | `exp/32_ablation_levels.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `lev4_rules.json` | extra | `exp/32_ablation_levels.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `net_rules.json` | extra | `exp/32_ablation_levels.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `robust_iot17_B.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_iot17_d3qn.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_iot17_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_iot17_td3.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_iot23_B.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_iot23_d3qn.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_iot23_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_iot23_td3.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_port1_B.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_port1_d3qn.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_port1_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `robust_port1_td3.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_robust.sh` | AutoDL | checked | 表 |
| `sensD120_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `sensD30_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `sensG1200_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_final.sh` | AutoDL | checked | 表 |
| `sensG300_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_final.sh` | AutoDL | checked | 表 |
| `sensH240_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `sensH60_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `sensK15_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `sensK5_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_extra.sh` | AutoDL | checked | 表 |
| `sensR150_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_final.sh` | AutoDL | checked | 表 |
| `sensR600_main.json` | extra | `exp/33_robustness_sensitivity.sh` | `run_final.sh` | AutoDL | checked | 表 |
| `qsweep0.42_Basis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.42_Basis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.42_d3qnasis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.42_d3qnasis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.42_fixed.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.42_propasis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.42_rules.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.42_td3asis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.42_td3asis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.43_Basis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.43_Basis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.43_d3qnasis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.43_d3qnasis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.43_fixed.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.43_propasis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.43_rules.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.43_td3asis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.43_td3asis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.44_fixed.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | cap |
| `qsweep0.44_rules.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.45_Basis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.45_Basis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.45_d3qnasis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.45_d3qnasis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.45_fixed.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.45_fixedlow.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_fixlow.sh` | AutoDL | checked | qsweep |
| `qsweep0.45_propasis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.45_rules.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.45_td3asis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.45_td3asis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.46_Basis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.46_Basis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.46_d3qnasis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.46_d3qnasis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `qsweep0.46_fixed.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.46_fixedlow.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_fixlow.sh` | AutoDL | checked | qsweep |
| `qsweep0.46_propasis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.46_rules.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qbar.sh` | server 182 | checked | qsweep |
| `qsweep0.46_td3asis.json` | qsweep | `exp/34_recall_requirement.sh` | `run_new_qasis.sh` | AutoDL | checked | qsweep |
| `qsweep0.46_td3asis_s1to4.json` | qsweep | `exp/34_recall_requirement.sh` | `run_182_qasis.sh` | server 182 | checked | qsweep |
| `diag_dpp.json` | extra | `exp/40_behaviour_timing.sh` | `run_old_narr.sh` | rented 3090 | unchecked | behav |
| `diag_exh.json` | extra | `exp/40_behaviour_timing.sh` | `run_old_narr.sh` | rented 3090 | unchecked | behav |
| `diag_prop.json` | extra | `exp/40_behaviour_timing.sh` | `run_old_narr.sh` | rented 3090 | unchecked | behav |
| `time_B.json` | extra | `exp/40_behaviour_timing.sh` | `run_extra.sh` | AutoDL | unchecked | 表 |
| `time_d3qn.json` | extra | `exp/40_behaviour_timing.sh` | `run_extra.sh` | AutoDL | unchecked | 表 |
| `time_fixed2.json` | extra | `exp/40_behaviour_timing.sh` | `run_extra.sh` | AutoDL | unchecked | 表 |
| `time_lyap0.3.json` | extra | `exp/40_behaviour_timing.sh` | `run_extra.sh` | AutoDL | unchecked | 表 |
| `time_lyapp3v0.1.json` | extra | `exp/40_behaviour_timing.sh` | `run_extra.sh` | AutoDL | unchecked | 表 |
| `time_prop.json` | extra | `exp/40_behaviour_timing.sh` | `run_extra.sh` | AutoDL | unchecked | 表 |
| `time_td3.json` | extra | `exp/40_behaviour_timing.sh` | `run_extra.sh` | AutoDL | unchecked | 表 |

## 图表读取、但没有对应仿真命令的文件

- `a3_per_image.json`（no record）
- `sim_inputs.json`（no record）
