# LearnRisk-QRS Remote Experiment Runbook

Date: 2026-05-30

Target: `lab-s2:/home/qiankun/HPPO-VQA`

Purpose: turn the current local-smoke LearnRisk-QRS evidence into the
remote multi-seed evidence required by the five TCCN readiness gates.
The current remote audit is `needs-remote-sync`; do not treat any local
mini/tiny CSV as submission evidence.

## Execution Order

1. Sync and verify the LearnRisk-QRS code path.
2. Run environment smoke tests and the tiny remote gate.
3. Launch the five remote multi-seed jobs for F2-F5.
4. Monitor logs until each job writes its CSV/TEX outputs.
5. Collect the remote CSV/TEX/log artifacts back into the local
   workspace.
6. Re-run the remote readiness audit; metric sources should switch from
   `local-smoke` to `final` for completed gates.

## Step 1: Sync

Run the local preflight before syncing:

```bash
python3 hppo-uav/run/qrs_remote_preflight.py
```

It checks Python syntax, script `--help` entry points, manifest
generation/verification, local readiness, local evidence contract, and
local job-status reports. With `--check-remote`, it also runs read-only
SSH checks and reports whether the server is `ready`, still
`needs-sync`, or has missing remote job outputs. Generate the manifest
locally before syncing if you need to refresh the rsync command:

```bash
python3 hppo-uav/run/qrs_remote_sync_manifest.py
```

Then generate the guarded sync plan:

```bash
python3 hppo-uav/run/qrs_remote_sync_plan.py
```

This writes `plan/qrs-remote-sync-plan.md` with both the `rsync -azR`
command and the remote manifest verification command. It does not copy
files by default. To execute the sync intentionally, run it with
`--execute --confirm SYNC_REMOTE_QRS_FILES`; the script will run rsync
and then verify the remote manifest before reporting success.

If you sync manually, run the `rsync -azR` command printed in
`plan/qrs-remote-sync-manifest.md`. After syncing, verify on the server:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA &&
  /home/qiankun/.conda/envs/DI-engine/bin/python \
    hppo-uav/run/qrs_remote_sync_manifest.py \
    --verify-json paper/data/qrs_remote_sync_manifest.json
'
```

Pass condition: status is `pass` and every hashed manifest file is `ok`.
Use the file count printed in `plan/qrs-remote-sync-manifest.md` as the
authority.

## Step 2: Smoke

Before running smoke by hand, generate the guarded smoke plan:

```bash
python3 hppo-uav/run/qrs_remote_smoke_plan.py
```

This writes `plan/qrs-remote-smoke-plan.md` and
`paper/data/qrs_remote_smoke_plan.json`. It verifies the remote sync
manifest, runs the environment smoke test, runs a tiny one-seed F2 gate,
and checks that the tiny remote CSV/TEX outputs exist. It does not touch
the server by default. To intentionally run the remote smoke sequence,
use `--execute --confirm RUN_REMOTE_QRS_SMOKE` after sync verification.

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  PY=/home/qiankun/.conda/envs/DI-engine/bin/python &&
  PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
  $PY -m pytest tests/test_uavnet_env.py -q
'
```

Then:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  PY=/home/qiankun/.conda/envs/DI-engine/bin/python &&
  PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
  $PY run/qrs_tccn_gate.py \
    --n-seeds 1 --eval-episodes 1 --collect-episodes 1 \
    --max-steps 10 --epochs 5 \
    --out-csv paper/data/qrs_tccn_gate_remote_smoke.csv \
    --out-tex paper/tables/qrs_tccn_gate_remote_smoke.tex
'
```

Pass condition: pytest passes and `qrs_tccn_gate_remote_smoke.csv`
exists.

## Step 3: Multi-Seed Jobs

Before launching jobs by hand, generate the dry-run launch plan:

```bash
python3 hppo-uav/run/qrs_remote_launch_plan.py
```

This writes `plan/qrs-remote-launch-plan.md` and
`paper/data/qrs_remote_launch_plan.json` with the exact F2-F5 SSH/nohup
commands. The script does not start anything unless it is run with
`--execute --confirm START_REMOTE_QRS_JOBS`; use that only after the
remote sync manifest verifies successfully on the server.

Launch F2:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  mkdir -p logs ../paper/data ../paper/tables &&
  PY=/home/qiankun/.conda/envs/DI-engine/bin/python &&
  nohup env PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
    $PY run/qrs_tccn_gate.py \
      --n-seeds 3 --eval-episodes 20 --collect-episodes 10 \
      --max-steps 80 --epochs 40 \
      --learnrisk-mixed-collect \
      --learnrisk-safety-mode regime_learned_risk \
      --learnrisk-use-moe \
      --risk-threshold 0.9 \
      --risk-threshold-stress 0.5 \
      --risk-regime-loss-db 5.0 \
      --out-csv paper/data/qrs_tccn_gate_remote_mseed.csv \
      --out-tex paper/tables/qrs_tccn_gate_remote_mseed.tex \
    > logs/qrs_tccn_gate_remote_mseed.log 2>&1 &
  echo PID:$!
'
```

Launch F4 MoE:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  mkdir -p logs ../paper/data ../paper/tables &&
  PY=/home/qiankun/.conda/envs/DI-engine/bin/python &&
  nohup env PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
    $PY run/qrs_moe_learnrisk_ablation.py \
      --n-seeds 3 --collect-episodes 10 --eval-episodes 20 \
      --max-steps 80 --epochs 40 \
      --tag remote_mseed \
      --out-csv paper/data/qrs_moe_learnrisk_remote_mseed.csv \
      --out-tex paper/tables/qrs_moe_learnrisk_remote_mseed.tex \
    > logs/qrs_moe_learnrisk_remote_mseed.log 2>&1 &
  echo PID:$!
'
```

Launch F4 component ablation:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  mkdir -p logs ../paper/data ../paper/tables &&
  PY=/home/qiankun/.conda/envs/DI-engine/bin/python &&
  nohup env PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
    $PY run/qrs_component_ablation.py \
      --n-seeds 3 --n-episodes 20 --max-steps 80 \
      --out-json paper/data/qrs_component_ablation_remote_mseed.json \
      --out-csv paper/data/qrs_component_ablation_remote_mseed.csv \
      --out-tex paper/tables/qrs_component_ablation_remote_mseed.tex \
    > logs/qrs_component_ablation_remote_mseed.log 2>&1 &
  echo PID:$!
'
```

Launch F3:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  mkdir -p logs ../paper/data ../paper/tables &&
  PY=/home/qiankun/.conda/envs/DI-engine/bin/python &&
  nohup env PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
    $PY run/qrs_robustness_sweep.py \
      --loss-offsets-db 0,5,10,15 \
      --n-seeds 3 --collect-episodes 8 --eval-episodes 20 \
      --max-steps 80 --epochs 40 \
      --out-csv paper/data/qrs_robustness_sweep_remote_mseed.csv \
      --out-tex paper/tables/qrs_robustness_sweep_remote_mseed.tex \
    > logs/qrs_robustness_sweep_remote_mseed.log 2>&1 &
  echo PID:$!
'
```

Launch F5:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  mkdir -p logs ../paper/data ../paper/tables &&
  PY=/home/qiankun/.conda/envs/DI-engine/bin/python &&
  nohup env PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
    $PY run/qrs_large_ue_sweep.py \
      --N-values 4,6,8,10 \
      --n-seeds 3 --collect-episodes 8 --eval-episodes 20 \
      --max-steps 80 --epochs 40 \
      --out-csv paper/data/qrs_large_ue_sweep_remote_mseed.csv \
      --out-tex paper/tables/qrs_large_ue_sweep_remote_mseed.tex \
    > logs/qrs_large_ue_sweep_remote_mseed.log 2>&1 &
  echo PID:$!
'
```

## Step 4: Monitor

After syncing, the same status helper can check the expected job
artifacts without scanning logs manually:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  /home/qiankun/.conda/envs/DI-engine/bin/python \
    run/qrs_remote_job_status.py \
    --out-json paper/data/qrs_remote_job_status.json \
    --out-md plan/qrs-remote-job-status.md
'
```

It reports `complete` only when every remote/mseed CSV and TEX file for
F2-F5 exists, and it prints the CSV data-row count so an empty artifact
is not mistaken for final evidence.

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  echo active_python:
  pgrep -af "python.*qrs_" || true
  echo logs:
  for f in logs/qrs_*remote_mseed.log; do
    echo === $f
    tail -n 20 "$f" 2>/dev/null || true
  done
'
```

Expected final outputs:

- `paper/data/qrs_tccn_gate_remote_mseed.csv`
- `paper/tables/qrs_tccn_gate_remote_mseed.tex`
- `paper/data/qrs_moe_learnrisk_remote_mseed.csv`
- `paper/tables/qrs_moe_learnrisk_remote_mseed.tex`
- `paper/data/qrs_component_ablation_remote_mseed.csv`
- `paper/tables/qrs_component_ablation_remote_mseed.tex`
- `paper/data/qrs_robustness_sweep_remote_mseed.csv`
- `paper/tables/qrs_robustness_sweep_remote_mseed.tex`
- `paper/data/qrs_large_ue_sweep_remote_mseed.csv`
- `paper/tables/qrs_large_ue_sweep_remote_mseed.tex`

## Step 5: Collect Results

Generate the guarded collection plan locally:

```bash
python3 hppo-uav/run/qrs_remote_collect_plan.py
```

This writes `plan/qrs-remote-collect-plan.md` and
`paper/data/qrs_remote_collect_plan.json`. It does not copy files by
default. To intentionally collect completed remote results, run:

```bash
python3 hppo-uav/run/qrs_remote_collect_plan.py \
  --execute --confirm COLLECT_REMOTE_QRS_RESULTS
```

The collect script first runs the remote job-status checker. It refuses
to copy results unless every expected F2-F5 CSV/TEX/log exists; use
`--allow-partial` only for debugging failed jobs. After collecting, it
reruns the local evidence-contract checker and writes
`plan/tccn-evidence-contract-after-collect.md`.

## Step 6: Audit

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  PY=/home/qiankun/.conda/envs/DI-engine/bin/python &&
  PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid \
  $PY run/qrs_tccn_readiness_audit.py --out-json paper/data/tccn_readiness_audit_remote.json --out-md plan/tccn-readiness-audit-remote.md
'
```

Pass condition: F2-F5 no longer report missing final evidence, and their
metric source strength is `final`. The overall paper still requires
manual interpretation of whether the final metrics support a TCCN
dominance or trade-off claim.

Then run the evidence-contract checker:

```bash
ssh lab-s2 '
  cd /home/qiankun/HPPO-VQA/hppo-uav &&
  /home/qiankun/.conda/envs/DI-engine/bin/python \
    run/qrs_tccn_evidence_contract.py \
    --out-json paper/data/tccn_evidence_contract.json \
    --out-md plan/tccn-evidence-contract-check.md
'
```

Pass condition: every gate is at least `remote-mseed`; only
`claim-ready` or explicitly justified `tradeoff-ready` results should be
used as main TCCN claims.
