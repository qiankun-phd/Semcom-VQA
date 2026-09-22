# LSS-HSR-L Communication Policy Simulation

Input questions: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_radar_questions.csv`
Samples: `1455`

| Policy | Cost | Avg cost | Floor rate | Drone miss | False confirm | Critical floor |
|---|---:|---:|---:|---:|---:|---:|
| `no_radar_uniform` | 8730 | 6.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| `radar_triggered_greedy` | 7221 | 4.9629 | 1.0 | 0.0 | 0.4501 | 1.0 |
| `typed_price_qrs` | 8649 | 5.9443 | 1.0 | 0.0 | 0.4501 | 1.0 |
| `learnrisk_qrs` | 10273 | 7.0605 | 1.0 | 0.0 | 0.8435 | 1.0 |
