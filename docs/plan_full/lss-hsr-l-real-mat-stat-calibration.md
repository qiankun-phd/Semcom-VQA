# LSS-HSR-L MAT-Statistic Semantic Calibration

Overall: **mat-stat-calibration-ready**
Input manifest: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_sample_manifest.csv`
Dataset path: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/数据集及使用说明`
Output manifest: `paper/data/lss_hsr_l_real_mat_calibrated_manifest.csv`
Feature CSV: `paper/data/lss_hsr_l_real_mat_statistics.csv`
Rows: `1455`
All real calibrated: `True`

## Calibrated Signal Summary

- avg radar confidence: `0.6477`
- avg drone-bird margin: `0.3419`
- avg trajectory risk: `0.5062`
- high trajectory risk rows: `112`
- ambiguous drone-bird rows: `296`

## Feature Basis

- Doppler: rows/bins, mean, standard deviation, 95th percentile, contrast.
- Trajectory: radial speed, range, azimuth, height, and normalized SNR statistics.

## Claim Boundary

These are transparent real-file statistics, not a trained radar classifier.
They are sufficient to move beyond class-only priors, but final TCCN
claims still need learned calibration and multi-seed system evidence.
