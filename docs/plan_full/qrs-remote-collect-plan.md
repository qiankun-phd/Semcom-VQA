# LearnRisk-QRS Remote Result Collection Plan

Mode: `dry-run`
Remote host: `lab-s2`
Remote root: `/home/qiankun/HPPO-VQA`
Artifact count: `15`

## Status Check

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 run/qrs_remote_job_status.py --check-remote --remote-host lab-s2 --remote-root /home/qiankun/HPPO-VQA --out-json paper/data/qrs_remote_job_status_collectcheck.json --out-md plan/qrs-remote-job-status-collectcheck.md
```

## Artifacts

- `hppo-uav/logs/qrs_component_ablation_remote_mseed.log`
- `hppo-uav/logs/qrs_large_ue_sweep_remote_mseed.log`
- `hppo-uav/logs/qrs_moe_learnrisk_remote_mseed.log`
- `hppo-uav/logs/qrs_robustness_sweep_remote_mseed.log`
- `hppo-uav/logs/qrs_tccn_gate_remote_mseed.log`
- `paper/data/qrs_component_ablation_remote_mseed.csv`
- `paper/data/qrs_large_ue_sweep_remote_mseed.csv`
- `paper/data/qrs_moe_learnrisk_remote_mseed.csv`
- `paper/data/qrs_robustness_sweep_remote_mseed.csv`
- `paper/data/qrs_tccn_gate_remote_mseed.csv`
- `paper/tables/qrs_component_ablation_remote_mseed.tex`
- `paper/tables/qrs_large_ue_sweep_remote_mseed.tex`
- `paper/tables/qrs_moe_learnrisk_remote_mseed.tex`
- `paper/tables/qrs_robustness_sweep_remote_mseed.tex`
- `paper/tables/qrs_tccn_gate_remote_mseed.tex`

## Rsync Commands

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./hppo-uav/logs/qrs_component_ablation_remote_mseed.log /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./hppo-uav/logs/qrs_large_ue_sweep_remote_mseed.log /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./hppo-uav/logs/qrs_moe_learnrisk_remote_mseed.log /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./hppo-uav/logs/qrs_robustness_sweep_remote_mseed.log /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./hppo-uav/logs/qrs_tccn_gate_remote_mseed.log /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/data/qrs_component_ablation_remote_mseed.csv /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/data/qrs_large_ue_sweep_remote_mseed.csv /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/data/qrs_moe_learnrisk_remote_mseed.csv /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/data/qrs_robustness_sweep_remote_mseed.csv /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/data/qrs_tccn_gate_remote_mseed.csv /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/tables/qrs_component_ablation_remote_mseed.tex /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/tables/qrs_large_ue_sweep_remote_mseed.tex /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/tables/qrs_moe_learnrisk_remote_mseed.tex /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/tables/qrs_robustness_sweep_remote_mseed.tex /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

```bash
rsync -azR lab-s2:/home/qiankun/HPPO-VQA/./paper/tables/qrs_tccn_gate_remote_mseed.tex /Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

## Post-Collect Evidence Contract

```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 run/qrs_tccn_evidence_contract.py --out-json paper/data/tccn_evidence_contract_after_collect.json --out-md plan/tccn-evidence-contract-after-collect.md
```
