# LSS-HSR-L Trained Radar Calibration

Overall: **trained-calibration-ready**
Input manifest: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_sample_manifest.csv`
Feature CSV: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_windowed_statistics.csv`
Output manifest: `paper/data/lss_hsr_l_real_calibrated_manifest.csv`
Rows: `1455`

## Model Evidence

- train accuracy: `0.7585`
- validation accuracy: `0.756`
- train log loss: `0.7239`
- validation log loss: `0.7399`
- high trajectory risk rows: `467`
- ambiguous drone-bird rows: `407`

## Claim Boundary

This is a lightweight in-repo calibration model trained on MAT-statistic
features. It supports the radar-to-VQA semantic bridge, but final TCCN
claims still require multi-seed communication/control evidence.
