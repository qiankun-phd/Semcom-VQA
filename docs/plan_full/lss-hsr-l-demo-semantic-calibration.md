# LSS-HSR-L Semantic Calibration

Input manifest: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_demo_manifest.csv`
Output manifest: `paper/data/lss_hsr_l_demo_calibrated_manifest.csv`
Rows: `4`
All real calibrated: `True`

## Calibration Modes

- `provided`: `4`

## Risk Signals

- high trajectory risk rows: `1`
- ambiguous drone-bird rows: `1`

## Claim Boundary

Rows with `prior-derived-needs-real-calibration` are usable for interface
testing only. They are not classifier confidence, not measured trajectory
risk, and not final TCCN evidence.
