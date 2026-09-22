# LearnRisk-QRS Remote Launch Plan

Mode: `dry-run`
Remote host: `lab-s2`
Remote root: `/home/qiankun/HPPO-VQA`

## Required Pre-Launch Verification

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA
/home/qiankun/.conda/envs/DI-engine/bin/python hppo-uav/run/qrs_remote_sync_manifest.py --verify-json paper/data/qrs_remote_sync_manifest.json'
```

This file is generated in dry-run mode by default. It does not prove
that remote jobs have started; use the remote job-status checker for
that.

## F2 `qrs_tccn_gate_remote_mseed`

Script: `run/qrs_tccn_gate.py`
Log: `hppo-uav/logs/qrs_tccn_gate_remote_mseed.log`

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA/hppo-uav
mkdir -p logs ../paper/data ../paper/tables
export PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid
nohup /home/qiankun/.conda/envs/DI-engine/bin/python run/qrs_tccn_gate.py --n-seeds 3 --eval-episodes 20 --collect-episodes 10 --max-steps 80 --epochs 40 --learnrisk-mixed-collect --learnrisk-safety-mode regime_learned_risk --learnrisk-use-moe --risk-threshold 0.9 --risk-threshold-stress 0.5 --risk-regime-loss-db 5.0 --out-csv paper/data/qrs_tccn_gate_remote_mseed.csv --out-tex paper/tables/qrs_tccn_gate_remote_mseed.tex > logs/qrs_tccn_gate_remote_mseed.log 2>&1 &
echo "JOB:qrs_tccn_gate_remote_mseed:PID:$!"'
```

## F4 `qrs_moe_learnrisk_remote_mseed`

Script: `run/qrs_moe_learnrisk_ablation.py`
Log: `hppo-uav/logs/qrs_moe_learnrisk_remote_mseed.log`

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA/hppo-uav
mkdir -p logs ../paper/data ../paper/tables
export PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid
nohup /home/qiankun/.conda/envs/DI-engine/bin/python run/qrs_moe_learnrisk_ablation.py --n-seeds 3 --collect-episodes 10 --eval-episodes 20 --max-steps 80 --epochs 40 --tag remote_mseed --out-csv paper/data/qrs_moe_learnrisk_remote_mseed.csv --out-tex paper/tables/qrs_moe_learnrisk_remote_mseed.tex > logs/qrs_moe_learnrisk_remote_mseed.log 2>&1 &
echo "JOB:qrs_moe_learnrisk_remote_mseed:PID:$!"'
```

## F4 `qrs_component_ablation_remote_mseed`

Script: `run/qrs_component_ablation.py`
Log: `hppo-uav/logs/qrs_component_ablation_remote_mseed.log`

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA/hppo-uav
mkdir -p logs ../paper/data ../paper/tables
export PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid
nohup /home/qiankun/.conda/envs/DI-engine/bin/python run/qrs_component_ablation.py --n-seeds 3 --n-episodes 20 --max-steps 80 --out-json paper/data/qrs_component_ablation_remote_mseed.json --out-csv paper/data/qrs_component_ablation_remote_mseed.csv --out-tex paper/tables/qrs_component_ablation_remote_mseed.tex > logs/qrs_component_ablation_remote_mseed.log 2>&1 &
echo "JOB:qrs_component_ablation_remote_mseed:PID:$!"'
```

## F3 `qrs_robustness_sweep_remote_mseed`

Script: `run/qrs_robustness_sweep.py`
Log: `hppo-uav/logs/qrs_robustness_sweep_remote_mseed.log`

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA/hppo-uav
mkdir -p logs ../paper/data ../paper/tables
export PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid
nohup /home/qiankun/.conda/envs/DI-engine/bin/python run/qrs_robustness_sweep.py --loss-offsets-db 0,5,10,15 --n-seeds 3 --collect-episodes 8 --eval-episodes 20 --max-steps 80 --epochs 40 --out-csv paper/data/qrs_robustness_sweep_remote_mseed.csv --out-tex paper/tables/qrs_robustness_sweep_remote_mseed.tex > logs/qrs_robustness_sweep_remote_mseed.log 2>&1 &
echo "JOB:qrs_robustness_sweep_remote_mseed:PID:$!"'
```

## F5 `qrs_large_ue_sweep_remote_mseed`

Script: `run/qrs_large_ue_sweep.py`
Log: `hppo-uav/logs/qrs_large_ue_sweep_remote_mseed.log`

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA/hppo-uav
mkdir -p logs ../paper/data ../paper/tables
export PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid
nohup /home/qiankun/.conda/envs/DI-engine/bin/python run/qrs_large_ue_sweep.py --N-values 4,6,8,10 --n-seeds 3 --collect-episodes 8 --eval-episodes 20 --max-steps 80 --epochs 40 --out-csv paper/data/qrs_large_ue_sweep_remote_mseed.csv --out-tex paper/tables/qrs_large_ue_sweep_remote_mseed.tex > logs/qrs_large_ue_sweep_remote_mseed.log 2>&1 &
echo "JOB:qrs_large_ue_sweep_remote_mseed:PID:$!"'
```
