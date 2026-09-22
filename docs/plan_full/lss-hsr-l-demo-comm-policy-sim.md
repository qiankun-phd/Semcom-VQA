# LSS-HSR-L Communication Policy Simulation

Input questions: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_demo_questions.csv`
Samples: `4`

| Policy | Cost | Avg cost | Floor rate | Drone miss | False confirm | Critical floor |
|---|---:|---:|---:|---:|---:|---:|
| `no_radar_uniform` | 24 | 6.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| `radar_triggered_greedy` | 16 | 4.0 | 1.0 | 0.0 | 0.3333 | 1.0 |
| `typed_price_qrs` | 19 | 4.75 | 1.0 | 0.0 | 0.3333 | 1.0 |
| `learnrisk_qrs` | 18 | 4.5 | 1.0 | 0.0 | 0.3333 | 1.0 |
