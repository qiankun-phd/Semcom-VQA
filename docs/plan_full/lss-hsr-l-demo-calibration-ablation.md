# LSS-HSR-L Calibration Ablation

Overall: **calibration-ablation-ready-real-data-pending**
Input manifest: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_demo_manifest.csv`
Rows: `4`
Real calibrated rows: `4`
Confirm flip rows: `1`
Average typed price abs delta: `0.6997`
Max typed price abs delta: `2.4736`

| Sample | Target | Prior confirm | Calibrated confirm | Price delta |
|---|---|---|---|---:|
| `lss_000001_track002` | ground fixed rotating target | no | no | 0.2052 |
| `lss_000002_sample001` | DJI Air3 | yes | yes | 0.0729 |
| `lss_000003_sample002` | car | no | no | 0.0473 |
| `lss_000004_track001` | bird flock | no | yes | 2.4736 |

## Claim Boundary

This ablation proves the calibration interface can measure prior-vs-real
semantic-control changes. It is final evidence only when the calibrated
fields come from audited LSS-HSR-L files or a validated radar classifier.
