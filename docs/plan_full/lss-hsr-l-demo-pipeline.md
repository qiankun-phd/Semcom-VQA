# LSS-HSR-L Demo Pipeline

This is a synthetic placeholder demo of the radar-triggered patrol-warning
semantic communication pipeline. It does not contain real LSS-HSR-L samples.

Dataset root: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_demo/lss_hsr_l_demo_dataset`
Manifest CSV: `paper/data/lss_hsr_l_demo_manifest.csv`
Question CSV: `paper/data/lss_hsr_l_demo_questions.csv`
Feature CSV: `paper/data/lss_hsr_l_demo_semantic_control_features.csv`
Simulation CSV: `paper/data/lss_hsr_l_demo_comm_policy_sim.csv`

## Semantic-Control Feature Summary

- question rows: `16`
- risk-gated rows: `8`
- average typed price seed: `3.9582`
- max typed price seed: `10.2789`

## Policy Metrics

| Policy | Cost | Avg cost | Floor rate | Drone miss | False confirm |
|---|---:|---:|---:|---:|---:|
| `no_radar_uniform` | 24 | 6.0 | 1.0 | 0.0 | 1.0 |
| `radar_triggered_greedy` | 16 | 4.0 | 1.0 | 0.0 | 0.3333 |
| `typed_price_qrs` | 19 | 4.75 | 1.0 | 0.0 | 0.3333 |
| `learnrisk_qrs` | 18 | 4.5 | 1.0 | 0.0 | 0.3333 |
