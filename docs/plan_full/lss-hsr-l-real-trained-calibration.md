# LSS-HSR-L Trained Radar Calibration

Overall: **trained-calibration-ready**
Input manifest: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_sample_manifest.csv`
Feature CSV: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_mat_statistics.csv`
Output manifest: `paper/data/lss_hsr_l_real_trained_calibrated_manifest.csv`
Rows: `1455`

## Model Evidence

- train accuracy: `0.7328`
- validation accuracy: `0.736`
- train log loss: `0.8146`
- validation log loss: `0.7856`
- high trajectory risk rows: `449`
- ambiguous drone-bird rows: `416`

## Claim Boundary

This is a lightweight in-repo calibration model trained on MAT-statistic
features. It supports the radar-to-VQA semantic bridge, but final TCCN
claims still require multi-seed communication/control evidence.
