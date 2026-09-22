# LearnRisk-QRS Remote Execution Approval Packet

Date: 2026-05-30

Purpose: make the remaining TCCN evidence work explicit before any
remote side effects occur. The current local state is ready for guarded
remote validation, but the server is not yet synced.

## Current Evidence State

- Local preflight: `local-pass`.
- Local submission gate: `local-smoke-only`.
- Remote root: `/home/qiankun/HPPO-VQA` exists.
- Remote sync state: `needs-sync`; 37 QRS validation files are missing.
- Remote result state: `missing`; 5 logs and 10 final CSV/TEX outputs
  are absent.

## Side Effects Requiring Approval

The next phase will:

1. Upload the 44-file QRS sync manifest to
   `lab-s2:/home/qiankun/HPPO-VQA/`.
2. Write or overwrite those synced files on the remote server.
3. Run a remote smoke sequence that creates
   `paper/data/qrs_tccn_gate_remote_smoke.csv` and
   `paper/tables/qrs_tccn_gate_remote_smoke.tex`.
4. If smoke passes, start five background multi-seed jobs with `nohup`.
5. Later collect remote CSV/TEX/log artifacts back into this workspace.

The guarded scripts refuse to run without their confirmation tokens, and
the launch script verifies the remote manifest before starting jobs.

## Execution Order After Approval

```bash
python3 hppo-uav/run/qrs_remote_sync_plan.py --execute --confirm SYNC_REMOTE_QRS_FILES
python3 hppo-uav/run/qrs_remote_smoke_plan.py --execute --confirm RUN_REMOTE_QRS_SMOKE
python3 hppo-uav/run/qrs_remote_launch_plan.py --execute --confirm START_REMOTE_QRS_JOBS
python3 hppo-uav/run/qrs_remote_job_status.py --check-remote
```

After all five jobs are complete:

```bash
python3 hppo-uav/run/qrs_remote_collect_plan.py --execute --confirm COLLECT_REMOTE_QRS_RESULTS
python3 hppo-uav/run/qrs_tccn_submission_gate.py --strict
```

## Approval Phrase

To authorize the remote side effects, reply with:

`同意执行远程同步、smoke、启动五个 multi-seed jobs`

Without that explicit approval, the safe next action is limited to local
dry-run planning and read-only remote checks.
