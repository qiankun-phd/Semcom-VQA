# LSS-HSR-L Three-Tier UAV/VQA Escalation Simulation

Input questions: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_radar_questions.csv`
Samples: `1455`

## Tier Semantics

- `radar_record_only`: radar is confident and operational risk is low; no UAV image is simulated.
- `visual_evidence_capture`: radar is confident enough but risk/accountability requires visual evidence.
- `close_vqa_disambiguation`: confidence or drone-bird margin is insufficient; UAV/VQA is used for semantic disambiguation.

## Summary

- tier counts: `{'radar_record_only': 501, 'visual_evidence_capture': 210, 'close_vqa_disambiguation': 744}`
- UAV dispatch rate: `0.6557`
- average visual-symbol cost: `6.468`
- expected VQA error rate: `0.1929`
- expected rotary-UAV miss rate: `0.0191`
- expected false escalation rate: `0.0692`

## Claim Boundary

This simulation does not create paired radar-image evidence. It tests how a radar-triggered
semantic controller would spend UAV/VQA confirmation resources when visual evidence is modeled
as a noisy, costly escalation sensor.
