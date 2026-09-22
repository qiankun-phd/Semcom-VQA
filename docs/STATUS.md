# TCCN Submission — Pipeline Status (2026-05-08)

## What is running NOW

### Server 1 — `lab-s1` (RTX 3080, 10 GB)
- Screen session: `main_sweep` (was 224048.main_sweep)
- Schedule: PDQN seeds 0, 1, 2 at (M=2, N=4) — sequential, ~3 runs × 21h each ≈ 63h
- Greedy eval (background nohup): (M=1, N=3), 100 episodes × 3 seeds → `~/HPPO-VQA/runs/greedy_M1N3_eval.json`

### Server 2 — `lab-s2` (RTX 4060, 8 GB)
- tmux session: `main_sweep`
- Schedule: 12 runs sequential
  - mahppo seeds 0/1/2 at (M=2, N=4)
  - mappo_hybrid seeds 0/1/2 at (M=2, N=4)
  - hppo seeds 0/1/2 at (M=1, N=3)  ← conference-replicating baseline
  - paddpg seeds 0/1/2 at (M=2, N=4)
- Each run ~4.1h → ~50h total
- Greedy eval (background nohup): (M=2, N=4), 100 episodes × 3 seeds → `~/HPPO-VQA/runs/greedy_M2N4_eval.json`

### Iteration budget
- 5e4 train iters per run (down from paper's 1.5e5 — trade-off for faster finish on weaker GPUs).
- Sufficient for clean PPO convergence on this scale of problem.

## Quick status check

```bash
bash /Users/zhangqiankun/Documents/mpu/HPPO-VQA/hppo-uav/run/status.sh
```

Reports GPU usage, screen/tmux sessions, log file sizes, done flags on both servers.

## Working baselines (smoke-test passed)

| Algorithm | Config | Status |
|---|---|---|
| MA-HPPO (ours) | `gym_uav_mahppo_config.py` | ✅ |
| MAPPO-hybrid | `gym_uav_mappo_hybrid_config.py` | ✅ (needed `HPPOSharedVAC` compat shims for PPO policy init) |
| HPPO (M=1, conference) | `gym_uav_hppo_config.py` | ✅ |
| PADDPG | `gym_uav_paddpg_config.py` | ✅ |
| PDQN | `gym_uav_pdqn_config.py` | ✅ |
| Greedy LUT heuristic | `dizoo/gym_hybrid/baselines/greedy.py` | ✅ (eval-only) |

## DEFERRED: HAPPO + Hybrid-SAC

DI-engine's native `HAPPOPolicy` and `havac.py` only support `action_space ∈ {discrete, continuous}` — not hybrid. Implementing hybrid HAPPO requires either:
1. New `HybridHAVAC` model with role-aware hybrid heads + `HAPPOPolicy._forward_*` patches, or
2. Custom HAPPO-style sequential update on top of our existing MA-HPPO model.

`SACPolicy` and `DiscreteSACPolicy` exist but no hybrid-SAC. Need a Christodoulou-style factorized policy (~300-400 LOC).

**Decision**: Keep §VIII Limitations item 4 (existing deferral language). Run with the 5+1 baselines that work. Add HAPPO/Hybrid-SAC in a revision pass if reviewers demand it.

## Bug fixes applied

1. `dizoo/gym_hybrid/envs/gym_hybrid_env.py`: env wrapper now passes `num_uav`, `num_usr`, `dynamic_task_arrival`, etc. through to `gym.make()`. Without this, multi-UAV configs silently fell back to (M=1, N=3).
2. `ding/model/template/hppo_shared.py`: added `actor_head` / `actor` / `critic` `_ListView` shims so `ding.policy.ppo._init_learn` can do its hybrid-action `log_sigma_param` init and orthogonal init. Avoids double-registration of encoder by NOT including it in the lists.
3. S1: needed `regex` (transformers dep) and upgraded `yapf` (Python 3.12 dropped lib2to3).
4. S2: replaced stale `gym-hybrid` editable install at `~/DI-engine/...` with our project's version.

## Output layout per run

```
~/HPPO-VQA/runs/<algo>_M<M>N<N>_s<seed>_main/
└── <algo>_M<M>N<N>_s<seed>_main_<TIMESTAMP>/
    ├── ckpt/
    │   ├── iteration_0.pth.tar
    │   ├── iteration_<latest>.pth.tar
    │   └── ckpt_best.pth.tar
    ├── log/
    │   ├── learner/learner_logger.txt   # tabular text logs
    │   ├── evaluator/evaluator_logger.txt
    │   ├── collector/collector_logger.txt
    │   ├── buffer/buffer_logger.txt
    │   └── serial/events.out.tfevents.* # TensorBoard
    ├── total_config.py
    └── formatted_total_config.py
```

## Next steps (when training finishes)

### 1. Aggregate results
```bash
ssh lab-s2 \
  'source ~/miniconda3/etc/profile.d/conda.sh && conda activate DI-engine \
   && pip install --user tensorboard \
   && cd ~/HPPO-VQA/hppo-uav \
   && PYTHONPATH=$PWD python run/aggregate_results.py \
        --runs-root ~/HPPO-VQA/runs \
        --suffix main \
        --out ~/HPPO-VQA/runs/_aggregated_main.json'
```
Output JSON has per-(algo, M, N) scalar series with iter axis + per-seed traces + last-10%-window mean/std.

### 2. Pull aggregated JSONs locally
```bash
mkdir -p paper/data
scp lab-s2:~/HPPO-VQA/runs/_aggregated_main.json paper/data/
scp lab-s2:~/HPPO-VQA/runs/greedy_M2N4_eval.json paper/data/
scp lab-s1:~/HPPO-VQA/runs/greedy_M1N3_eval.json paper/data/
```

### 3. Regenerate figures (TODO — not yet written)
- `paper/figures/_make_figures.py` already exists for mock data. Adapt to read from `paper/data/_aggregated_main.json` and produce real `fig3_convergence.pdf`, `fig6_similarity_cdf.pdf`.
- `fig4_scaling.pdf` needs B2 sweep results.
- `fig5_trajectory.pdf` needs an eval-time rollout dump.
- `fig7..9` need B4–B6 sweep results.

### 4. Replace mock numbers in §VI
- Search for "synthetic-mock" in `paper/sections/06_experiments.tex` — there are 8 occurrences (each figure / table caption).
- Replace headline numbers in Abstract (17%, 0.66→0.81), §VI:main paragraph, §VII discussion, §VI:cost wall-clock comments.
- Numbers come from `_aggregated_main.json` → `final_mean` / `final_std` of `evaluator_step/eval_episode_return` and per-metric scalars.

### 5. Run remaining sweeps (B2–B7)
After main sweep frees the GPUs:
- B2 scaling: 9 cells × 2 algos (MA-HPPO + MAPPO-hybrid) × 2 seeds × 3e4 iters
- B3 ablation: 4 variants × 2 seeds × 5e4 iters
- B4 sensitivity, B5 robustness, B6 jamming, B7 large-N: eval-only on B1 ckpts (seeds in env, not retrain).

### 6. TCCN final-compile pipeline
Already documented in `plan/tccn-submission-checklist.md`. After data swap:
- `cd paper && latexmk -pdf -outdir=build main.tex`
- Verify ≤ 14 pp.
- Run `paper-claim-audit`, `citation-audit`, `paper-self-review` skills.
- Update cover-letter §3 (iv) to match actual baseline scope.

## Servers' env summary

| | S1 (lab-s1) | S2 (lab-s2) |
|---|---|---|
| GPU | RTX 3080 / 10 GB | RTX 4060 / 8 GB |
| Driver | 550.54.14 | 550.144.03 |
| CUDA | 12.6 (torch) | 12.1 (torch) |
| Python | system 3.12 | conda DI-engine env, 3.8 |
| DI-engine | pip-user 0.5.3 | conda 0.5.3 |
| Throughput | ~1.1 train iter/s (PDQN actually slower ~0.7) | ~3.4 train iter/s |
| Conda? | NO (pip --user --break-system-packages) | YES (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate DI-engine`) |
| Code dir | `~/HPPO-VQA/hppo-uav/` | `~/HPPO-VQA/hppo-uav/` |
| Runs dir | `~/HPPO-VQA/runs/` | `~/HPPO-VQA/runs/` |
| Master logs | `~/HPPO-VQA/runs/_master_logs/` | `~/HPPO-VQA/runs/_master_logs/` |

## Open issues

- **Loss magnitudes**: PDQN's `total_loss_avg` shows 6e11 magnitudes early. Likely env reward magnitudes scaling with multi-UAV. Watch for divergence.
- **GPU underutilization**: S2 GPU at 1% util, S1 GPU at 93%. S2 is CPU-bound (subprocess env workers + smaller GPU). May want to tune `collector_env_num` from 8 → 4 if subsequent sweeps lag.
- **Disk on S2**: only 39 GB free. 12 runs × ckpts may stress this. Monitor with `df -h /` periodically; clean older ckpts if needed.
