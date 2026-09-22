# LearnRisk-QRS Remote Smoke Plan

Mode: `dry-run`
Remote host: `lab-s2`
Remote root: `/home/qiankun/HPPO-VQA`

This file is generated in dry-run mode by default. It verifies the
remote manifest, runs the environment smoke test, runs a tiny QRS gate,
and checks for the tiny gate CSV/TEX outputs.

## `manifest-verify`

Verify that all synced QRS files match the local manifest.

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA
/home/qiankun/.conda/envs/DI-engine/bin/python hppo-uav/run/qrs_remote_sync_manifest.py --verify-json paper/data/qrs_remote_sync_manifest.json'
```

## `env-pytest`

Run the remote UAVNet environment pytest smoke test.

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA/hppo-uav
export PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid
/home/qiankun/.conda/envs/DI-engine/bin/python -m pytest tests/test_uavnet_env.py -q'
```

## `tiny-gate`

Run a one-seed tiny F2 gate to verify the QRS entry point.

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA/hppo-uav
mkdir -p ../paper/data ../paper/tables
export PYTHONPATH=$PWD:$PWD/dizoo/gym_hybrid/envs/gym-hybrid
/home/qiankun/.conda/envs/DI-engine/bin/python run/qrs_tccn_gate.py --n-seeds 1 --eval-episodes 1 --collect-episodes 1 --max-steps 10 --epochs 5 --out-csv paper/data/qrs_tccn_gate_remote_smoke.csv --out-tex paper/tables/qrs_tccn_gate_remote_smoke.tex'
```

## `artifact-check`

Check that the tiny remote gate wrote non-empty CSV/TEX outputs.

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA
test -s paper/data/qrs_tccn_gate_remote_smoke.csv
test -s paper/tables/qrs_tccn_gate_remote_smoke.tex
echo "REMOTE_SMOKE_ARTIFACTS:pass"'
```
