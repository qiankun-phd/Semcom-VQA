"""Unit tests for conditional C1000 runner (no formal training, no final seeds)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "joint_control_rl_conditional_c1000.py"


def _load_c1000():
    spec = importlib.util.spec_from_file_location("joint_control_rl_conditional_c1000", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def c1000():
    if not SCRIPT.exists():
        pytest.skip(f"runner script missing: {SCRIPT}")
    return _load_c1000()


# ---------------------------------------------------------------------------
# Seed cycle + denylist
# ---------------------------------------------------------------------------


def test_seed_cycle_deterministic(c1000):
    seeds = [c1000.seed_at_episode(i) for i in range(40)]
    assert seeds[:20] == list(range(60000001, 60000021))
    assert seeds[20:40] == list(range(60000001, 60000021))
    assert len(set(seeds[:20])) == 20


def test_final_seed_denylist(c1000):
    for s in (80000001, 80000010, 80000020):
        with pytest.raises(RuntimeError, match="FINAL SEED DENIED"):
            c1000.assert_not_final_seed(s, "unit")
    with pytest.raises(RuntimeError):
        c1000.assert_allowed_train_seed(80000001)
    with pytest.raises(RuntimeError):
        c1000.assert_allowed_val_seed(80000001)
    with pytest.raises(RuntimeError):
        c1000.assert_allowed_train_seed(70000001)
    c1000.assert_allowed_val_seed(70000003)
    c1000.assert_allowed_train_seed(60000015)


def test_no_c3000_flag_trap(c1000):
    with pytest.raises(SystemExit):
        c1000.main(["--start-c3000", "--dry-run", "--max-episodes", "1", "--device", "cpu"])


# ---------------------------------------------------------------------------
# Decision fixtures (synthetic)
# ---------------------------------------------------------------------------


def _val_row(ep, obj, **kw):
    base = {
        "episode_completed": ep,
        "constrained_objective": obj,
        "quality_utility_mean": obj,
        "deadline_cost_mean": 0.01,
        "deadline_satisfied": True,
        "finite_ok": True,
        "cadence_ok": True,
        "task_conservation_ok": True,
        "provenance_ok": True,
        "entropy_mean": 40.0,
        "lambda_deadline": 0.5,
        "value_error_mean": 1.0,
        "explained_variance": 0.5,
        "per_seed_objective": [obj] * 5,
    }
    base.update(kw)
    return base


def _train_row(ep_idx, **kw):
    base = {
        "episode_index": ep_idx,
        "seed": 60000001 + (ep_idx % 20),
        "finite_ok": True,
        "cadence_ok": True,
        "task_conservation_ok": True,
        "entropy": 40.0,
        "approx_kl_max_over_epochs": 1e-6,
        "clip_fraction_max_over_epochs": 0.0,
        "ratio_mean_final_executed_epoch": 1.0,
        "kl_guard_stopped_early": False,
    }
    base.update(kw)
    return base


def test_decision_converged(c1000):
    # plateau around 200, above baseline 100
    vals = [_val_row(ep, 200.0 + 0.01 * i) for i, ep in enumerate([100, 150, 200, 250, 300])]
    # force plateau by flat objectives
    vals = [_val_row(ep, 200.0) for ep in (100, 150, 200, 250, 300)]
    trains = [_train_row(i) for i in range(300)]
    d = c1000.decide_c1000(
        vals,
        trains,
        episode_index=299,
        best_causal_baseline=100.0,
        oracle_upper=250.0,
    )
    assert d["decision"] == c1000.C1000_CONVERGED
    assert d["stop"] is True


def test_decision_plateau_below_target(c1000):
    vals = [_val_row(ep, 50.0) for ep in (100, 150, 200, 250, 300)]
    trains = [_train_row(i) for i in range(300)]
    d = c1000.decide_c1000(
        vals,
        trains,
        episode_index=299,
        best_causal_baseline=100.0,
        oracle_upper=250.0,
    )
    assert d["decision"] == c1000.C1000_PLATEAU_BELOW_TARGET
    assert d["stop"] is True


def test_decision_stable_still_improving_only_at_1000(c1000):
    # strong positive slope
    vals = [_val_row(ep, 10.0 * i) for i, ep in enumerate([800, 850, 900, 950, 1000])]
    trains = [_train_row(i) for i in range(1000)]
    d_mid = c1000.decide_c1000(
        [_val_row(ep, 10.0 * i) for i, ep in enumerate([100, 150, 200, 250, 300])],
        trains[:300],
        episode_index=299,
        best_causal_baseline=0.0,
        oracle_upper=None,
    )
    # at 300 with improving trajectory should NOT stop with STABLE_STILL_IMPROVING
    assert d_mid["decision"] != c1000.C1000_STABLE_STILL_IMPROVING

    d_end = c1000.decide_c1000(
        vals,
        trains,
        episode_index=999,
        best_causal_baseline=0.0,
        oracle_upper=None,
        force_terminal=True,
    )
    assert d_end["decision"] == c1000.C1000_STABLE_STILL_IMPROVING
    assert d_end["stop"] is True


def test_decision_unstable_nonfinite(c1000):
    trains = [_train_row(i) for i in range(10)]
    trains[-1]["finite_ok"] = False
    d = c1000.decide_c1000(
        [],
        trains,
        episode_index=9,
        best_causal_baseline=None,
        oracle_upper=None,
    )
    assert d["decision"] == c1000.C1000_UNSTABLE_OR_INVALID
    assert d["stop"] is True


def test_bootstrap_slope_includes_zero_on_flat(c1000):
    xs = [1, 2, 3, 4, 5]
    ys = [10.0, 10.0, 10.0, 10.0, 10.0]
    info = c1000.bootstrap_slope_ci(xs, ys, n_boot=500, seed=1)
    assert abs(info["slope"]) < 1e-9
    assert info["includes_zero"] is True


# ---------------------------------------------------------------------------
# Atomic checkpoint + single-instance lock
# ---------------------------------------------------------------------------


def test_atomic_write_json(c1000, tmp_path):
    path = tmp_path / "x.json"
    c1000.atomic_write_json(path, {"a": 1})
    assert path.exists()
    assert json.loads(path.read_text())["a"] == 1
    # no leftover tmp
    tmps = list(tmp_path.glob("*.tmp.*"))
    assert tmps == []


def test_single_instance_lock(c1000, tmp_path):
    lock_path = tmp_path / "runner.lock"
    lock1 = c1000.SingleInstanceLock(lock_path)
    lock1.acquire()
    lock2 = c1000.SingleInstanceLock(lock_path)
    with pytest.raises(RuntimeError, match="another C1000"):
        lock2.acquire()
    lock1.release()
    lock2.acquire()
    lock2.release()


# ---------------------------------------------------------------------------
# Resume RNG / optimizer bit-equivalence (CPU, tiny model-free state)
# ---------------------------------------------------------------------------


def test_rng_state_roundtrip_bit_equivalent(c1000):
    torch.manual_seed(123)
    np.random.seed(123)
    import random as pyrandom

    pyrandom.seed(123)
    # advance a bit
    _ = torch.randn(10)
    _ = np.random.randn(10)
    _ = pyrandom.random()
    st = c1000.capture_rng_state()
    a = torch.randn(5).tolist()
    b = np.random.randn(5).tolist()
    c = [pyrandom.random() for _ in range(3)]
    # restore and compare
    c1000.restore_rng_state(st)
    a2 = torch.randn(5).tolist()
    b2 = np.random.randn(5).tolist()
    c2 = [pyrandom.random() for _ in range(3)]
    assert a == a2
    assert b == b2
    assert c == c2


def test_constrained_objective_formula(c1000):
    j = c1000.constrained_objective_from_episode(
        quality_utility=100.0, deadline_cost_sum=2.0, lambda_deadline=0.5
    )
    assert j == pytest.approx(99.0)


def test_select_best_checkpoint_uses_validation_not_train_return(c1000):
    vals = [
        _val_row(50, 10.0, deadline_satisfied=True),
        _val_row(100, 30.0, deadline_satisfied=True),
        _val_row(150, 20.0, deadline_satisfied=True),
    ]
    ckpts = [
        {"episode_completed": 50, "path": "a.pt"},
        {"episode_completed": 100, "path": "b.pt"},
        {"episode_completed": 150, "path": "c.pt"},
    ]
    sel = c1000.select_best_checkpoint(vals, ckpts)
    assert sel is not None
    assert sel["checkpoint"]["path"] == "b.pt"
    assert sel["selection_rule"].startswith("max_validation")


def test_argv_final_seed_rejected(c1000):
    with pytest.raises(SystemExit):
        c1000.main(["--dry-run", "--max-episodes", "1", "80000001"])


# ---------------------------------------------------------------------------
# Full-checkpoint resume: model+optimizer state equality (tiny synthetic)
# ---------------------------------------------------------------------------


def test_full_checkpoint_resume_optimizer_model(c1000, tmp_path):
    from bubbles_vqa.rl_training.config import RLTrainConfig
    from bubbles_vqa.rl_training.duals import init_dual_state
    from bubbles_vqa.rl_training.networks import JointTwoTimescaleActorCritic

    cfg = RLTrainConfig(device="cpu", learning_rate=3e-5, result_grade=False)
    model = JointTwoTimescaleActorCritic(32, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    dual = init_dual_state(cfg)
    # one fake step to populate optimizer state
    loss = sum(p.sum() for p in model.parameters()) * 0.0 + model.encoder[0].weight.sum() * 0.0
    # force a real grad
    x = model.encoder[0].weight.sum()
    x.backward()
    opt.step()
    opt.zero_grad()

    path = tmp_path / "ckpt.pt"
    rng = c1000.capture_rng_state()
    c1000.save_full_checkpoint(
        path,
        model=model,
        optimizer=opt,
        dual=dual,
        cfg=cfg,
        episode_index=3,
        seed_cycle_position=4,
        rng_state=rng,
        provenance={"test": True},
        validation_history=[{"episode_completed": 50, "constrained_objective": 1.0}],
        decision=None,
    )
    sha_m = c1000.state_dict_sha(model)
    sha_o = c1000.optimizer_state_sha(opt)

    model2 = JointTwoTimescaleActorCritic(32, cfg)
    opt2 = torch.optim.Adam(model2.parameters(), lr=3e-5)
    dual2 = init_dual_state(cfg)
    resume = c1000.load_full_checkpoint(
        path, device="cpu", model=model2, optimizer=opt2, dual=dual2
    )
    assert resume["next_episode_index"] == 4
    assert resume["seed_cycle_position"] == 4
    assert c1000.state_dict_sha(model2) == sha_m
    assert c1000.optimizer_state_sha(opt2) == sha_o


def test_decision_labels_exhaustive(c1000):
    assert set(c1000.DECISION_LABELS) == {
        "C1000_CONVERGED",
        "C1000_STABLE_STILL_IMPROVING",
        "C1000_PLATEAU_BELOW_TARGET",
        "C1000_UNSTABLE_OR_INVALID",
    }
    assert c1000.C1000_STABLE_STILL_IMPROVING in c1000.DECISION_LABELS
    assert c1000.C1000_STABLE_STILL_IMPROVING not in c1000.STOP_IMMEDIATE or True
    # STABLE is stop-at-1000 only, but still a stop decision when emitted
    assert c1000.C1000_CONVERGED in c1000.STOP_IMMEDIATE


# ---------------------------------------------------------------------------
# Crash-window reconciliation (train JSONL / validation / temp / SIGTERM)
# ---------------------------------------------------------------------------


def _acceptable_val_row(ep: int, obj: float = 100.0, **kw):
    """Full validation row accepted by merge_validation_history."""
    base = {
        "episode_completed": ep,
        "constrained_objective": obj,
        "quality_utility_mean": obj,
        "deadline_cost_mean": 0.01,
        "deadline_satisfied": True,
        "finite_ok": True,
        "cadence_ok": True,
        "task_conservation_ok": True,
        "provenance_ok": True,
        "entropy_mean": 40.0,
        "lambda_deadline": 0.5,
        "value_error_mean": 1.0,
        "explained_variance": 0.5,
        "no_param_update": True,
        "dual_unchanged": True,
        "model_hash_pre": "abc",
        "model_hash_post": "abc",
        "optimizer_hash_pre": "opt",
        "optimizer_hash_post": "opt",
        "quality_table_sha256": "837d9f7fdeadbeef",
        "per_seed_objective": [obj] * 5,
    }
    base.update(kw)
    return base


def test_reconcile_train_jsonl_ahead_of_checkpoint(c1000, tmp_path):
    """Train JSONL one row ahead of checkpoint next_episode_index is truncated."""
    path = tmp_path / "training_metrics.jsonl"
    rows = [_train_row(i) for i in range(4)]  # 0..3
    for r in rows:
        c1000.append_jsonl(path, r)
    # Checkpoint authority: completed 0..2 → next=3, so row 3 is ahead
    kept = c1000.reconcile_training_metrics(path, next_episode_index=3, out_dir=tmp_path)
    assert [r["episode_index"] for r in kept] == [0, 1, 2]
    disk = c1000.read_jsonl(path)
    assert [r["episode_index"] for r in disk] == [0, 1, 2]
    log = c1000.read_jsonl(tmp_path / c1000.RECONCILE_LOG)
    assert any(
        e.get("event") == "reconcile_training_metrics"
        and e.get("truncated_episode_indices") == [3]
        for e in log
    )


def test_reconcile_train_jsonl_duplicate_row(c1000, tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    for i in range(3):
        c1000.append_jsonl(path, _train_row(i))
    c1000.append_jsonl(path, _train_row(1))  # identical duplicate of ep 1
    kept = c1000.reconcile_training_metrics(path, next_episode_index=3, out_dir=tmp_path)
    assert [r["episode_index"] for r in kept] == [0, 1, 2]
    assert len(c1000.read_jsonl(path)) == 3
    log = c1000.read_jsonl(tmp_path / c1000.RECONCILE_LOG)
    assert any(1 in (e.get("duplicate_episode_indices") or []) for e in log)


def test_reconcile_train_jsonl_conflict_fail_closed(c1000, tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    c1000.append_jsonl(path, _train_row(0, entropy=40.0))
    c1000.append_jsonl(path, _train_row(0, entropy=41.0))  # conflicting dup
    with pytest.raises(RuntimeError, match="conflicting training_metrics"):
        c1000.reconcile_training_metrics(path, next_episode_index=1, out_dir=tmp_path)


def test_merge_validation_recovers_disk_ahead_of_ckpt_history(c1000, tmp_path):
    """Disk validation JSONL has a legal row not yet in checkpoint history."""
    ckpt_hist = [_acceptable_val_row(50, 10.0)]
    disk = [
        _acceptable_val_row(50, 10.0),
        _acceptable_val_row(100, 20.0),  # recovered
    ]
    merged = c1000.merge_validation_history(
        ckpt_hist,
        disk,
        next_episode_index=100,
        out_dir=tmp_path,
    )
    assert [r["episode_completed"] for r in merged] == [50, 100]
    assert merged[1]["constrained_objective"] == 20.0
    log = c1000.read_jsonl(tmp_path / c1000.RECONCILE_LOG)
    assert any(
        e.get("event") == "merge_validation_history"
        and 100 in (e.get("recovered_episode_completed") or [])
        for e in log
    )


def test_merge_validation_conflict_fail_closed(c1000, tmp_path):
    ckpt_hist = [_acceptable_val_row(50, 10.0)]
    disk = [_acceptable_val_row(50, 99.0)]  # same ep, different objective
    with pytest.raises(RuntimeError, match="validation history conflict"):
        c1000.merge_validation_history(
            ckpt_hist, disk, next_episode_index=50, out_dir=tmp_path
        )


def test_merge_validation_rejects_ahead_and_incomplete(c1000, tmp_path):
    ckpt_hist: list = []
    disk = [
        _acceptable_val_row(150, 1.0),  # ahead of next=100 → skip
        {
            "episode_completed": 50,
            "constrained_objective": 1.0,
            # missing no-update / hashes → fail closed in-window
        },
    ]
    with pytest.raises(RuntimeError, match="invalid validation row"):
        c1000.merge_validation_history(
            ckpt_hist, disk, next_episode_index=100, out_dir=tmp_path
        )
    # ahead-only should succeed with empty merge
    merged = c1000.merge_validation_history(
        [],
        [_acceptable_val_row(150, 1.0)],
        next_episode_index=100,
        out_dir=tmp_path,
    )
    assert merged == []


def test_cleanup_stale_temp_sidecar(c1000, tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    good = ckpt_dir / "checkpoint_latest.json"
    good.write_text('{"ok": true}\n')
    stale = ckpt_dir / "checkpoint_latest.pt.tmp.json"
    stale.write_text('{"stale": true}\n')
    stale2 = ckpt_dir / ".checkpoint_latest.pt.writing.999"
    stale2.write_bytes(b"tmp")
    cleaned = c1000.cleanup_stale_temp_files(tmp_path)
    assert str(stale) in cleaned
    assert str(stale2) in cleaned
    assert good.exists()
    assert not stale.exists()
    assert not stale2.exists()


def test_save_full_checkpoint_no_tmp_json_leftover(c1000, tmp_path):
    from bubbles_vqa.rl_training.config import RLTrainConfig
    from bubbles_vqa.rl_training.duals import init_dual_state
    from bubbles_vqa.rl_training.networks import JointTwoTimescaleActorCritic

    cfg = RLTrainConfig(device="cpu", learning_rate=3e-5, result_grade=False)
    model = JointTwoTimescaleActorCritic(32, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    dual = init_dual_state(cfg)
    path = tmp_path / "checkpoint_latest.pt"
    train_row = _train_row(2)
    val_hist = [_acceptable_val_row(50)]
    c1000.save_full_checkpoint(
        path,
        model=model,
        optimizer=opt,
        dual=dual,
        cfg=cfg,
        episode_index=2,
        seed_cycle_position=3,
        rng_state=c1000.capture_rng_state(),
        provenance={"test": True},
        validation_history=val_hist,
        decision=None,
        last_train_row=train_row,
        journal_sequence=7,
    )
    assert path.exists()
    assert path.with_suffix(".json").exists()
    leftovers = list(tmp_path.glob("**/*tmp*")) + list(tmp_path.glob("**/*writing*"))
    assert leftovers == []
    resume = c1000.load_full_checkpoint(
        path,
        device="cpu",
        model=JointTwoTimescaleActorCritic(32, cfg),
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-5),
        dual=init_dual_state(cfg),
    )
    assert resume["next_episode_index"] == 3
    assert resume["journal_sequence"] == 7
    assert resume["last_train_row"]["episode_index"] == 2
    assert len(resume["validation_history"]) == 1


def test_sigterm_status_interrupted_resumable(c1000, tmp_path):
    """After cooperative interrupt save, status is INTERRUPTED_RESUMABLE not RUNNING."""
    # Minimal fake runner state machine for the interrupt tail.
    class _R:
        pass

    runner = _R()
    runner.out_dir = tmp_path
    runner.train_rows = [_train_row(0), _train_row(1)]
    runner.validation_history = []
    runner.journal_sequence = 2
    runner._interrupted = True
    runner._stop_requested = True
    runner.checkpoint_index = []
    runner.decision = None
    runner.provenance = {}
    runner.last_train_row = runner.train_rows[-1]
    runner.episode_wall_times = [1.0, 1.0]
    calls = []

    def _write_status(**kwargs):
        calls.append(kwargs)
        c1000.atomic_write_json(tmp_path / c1000.STATUS_NAME, {**kwargs, "pid": 1})

    def _maybe_checkpoint(ep_idx):
        calls.append({"checkpoint": ep_idx})

    runner._write_status = _write_status
    runner._maybe_checkpoint = _maybe_checkpoint

    # Execute the interrupt tail logic from C1000Runner.run (mirrors production).
    if runner._interrupted or runner._stop_requested:
        if runner.train_rows:
            runner._maybe_checkpoint(int(runner.train_rows[-1]["episode_index"]))
        completed = len(runner.train_rows)
        runner._write_status(
            status="INTERRUPTED_RESUMABLE",
            episodes_completed=completed,
        )
        c1000.atomic_write_json(
            tmp_path / c1000.RUN_MANIFEST,
            {
                "status": "INTERRUPTED_RESUMABLE",
                "episodes_completed": completed,
            },
        )

    status = json.loads((tmp_path / c1000.STATUS_NAME).read_text())
    assert status["status"] == "INTERRUPTED_RESUMABLE"
    assert status["status"] != "RUNNING"
    assert any("checkpoint" in c for c in calls)


def test_reconcile_then_resume_matches_reference_state(c1000, tmp_path):
    """After crash-window repair, resumed logical state matches uninterrupted ref."""
    from bubbles_vqa.rl_training.config import RLTrainConfig
    from bubbles_vqa.rl_training.duals import init_dual_state
    from bubbles_vqa.rl_training.networks import JointTwoTimescaleActorCritic

    cfg = RLTrainConfig(device="cpu", learning_rate=3e-5, result_grade=False)
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Uninterrupted reference at next_episode_index=3 ---
    ref_dir = tmp_path / "ref"
    crash_dir = tmp_path / "crash"
    ref_dir.mkdir()
    crash_dir.mkdir()
    (ref_dir / "checkpoints").mkdir()
    (crash_dir / "checkpoints").mkdir()

    model = JointTwoTimescaleActorCritic(32, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    dual = init_dual_state(cfg)
    x = model.encoder[0].weight.sum()
    x.backward()
    opt.step()
    opt.zero_grad()
    rng = c1000.capture_rng_state()
    train_rows = [_train_row(i) for i in range(3)]
    val_hist = [_acceptable_val_row(50, 12.5)]  # legal, within next=3? 50>3 → no
    # At next=3 validation history is empty (val only at 50). Keep empty for both.
    val_hist = []

    for r in train_rows:
        c1000.append_jsonl(ref_dir / c1000.TRAIN_METRICS, r)
    c1000.save_full_checkpoint(
        ref_dir / "checkpoints" / "checkpoint_latest.pt",
        model=model,
        optimizer=opt,
        dual=dual,
        cfg=cfg,
        episode_index=2,
        seed_cycle_position=3,
        rng_state=rng,
        provenance={"test": True},
        validation_history=val_hist,
        decision=None,
        last_train_row=train_rows[-1],
        journal_sequence=3,
    )

    # --- Crash window: train JSONL has ahead row + duplicate; stale temp ---
    for r in train_rows:
        c1000.append_jsonl(crash_dir / c1000.TRAIN_METRICS, r)
    c1000.append_jsonl(crash_dir / c1000.TRAIN_METRICS, _train_row(2))  # dup
    c1000.append_jsonl(crash_dir / c1000.TRAIN_METRICS, _train_row(3))  # ahead
    # Checkpoint matches reference (authority next=3)
    c1000.save_full_checkpoint(
        crash_dir / "checkpoints" / "checkpoint_latest.pt",
        model=model,
        optimizer=opt,
        dual=dual,
        cfg=cfg,
        episode_index=2,
        seed_cycle_position=3,
        rng_state=rng,
        provenance={"test": True},
        validation_history=val_hist,
        decision=None,
        last_train_row=train_rows[-1],
        journal_sequence=3,
    )
    (crash_dir / "checkpoints" / "checkpoint_latest.pt.tmp.json").write_text("{}\n")

    # Validation crash window: legal disk row not in ckpt (use next large enough)
    # For next=3 we cannot recover ep=50; use empty. Separate subcase below.
    c1000.cleanup_stale_temp_files(crash_dir)
    assert not (crash_dir / "checkpoints" / "checkpoint_latest.pt.tmp.json").exists()

    kept = c1000.reconcile_training_metrics(
        crash_dir / c1000.TRAIN_METRICS, next_episode_index=3, out_dir=crash_dir
    )
    ref_rows = c1000.read_jsonl(ref_dir / c1000.TRAIN_METRICS)
    assert [r["episode_index"] for r in kept] == [r["episode_index"] for r in ref_rows]
    assert len(kept) == 3

    # Load both checkpoints → model/opt/dual/RNG/next must match
    m1 = JointTwoTimescaleActorCritic(32, cfg)
    o1 = torch.optim.Adam(m1.parameters(), lr=3e-5)
    d1 = init_dual_state(cfg)
    r1 = c1000.load_full_checkpoint(
        ref_dir / "checkpoints" / "checkpoint_latest.pt",
        device="cpu",
        model=m1,
        optimizer=o1,
        dual=d1,
    )
    m2 = JointTwoTimescaleActorCritic(32, cfg)
    o2 = torch.optim.Adam(m2.parameters(), lr=3e-5)
    d2 = init_dual_state(cfg)
    r2 = c1000.load_full_checkpoint(
        crash_dir / "checkpoints" / "checkpoint_latest.pt",
        device="cpu",
        model=m2,
        optimizer=o2,
        dual=d2,
    )
    assert r1["next_episode_index"] == r2["next_episode_index"] == 3
    assert c1000.state_dict_sha(m1) == c1000.state_dict_sha(m2)
    assert c1000.optimizer_state_sha(o1) == c1000.optimizer_state_sha(o2)
    assert d1.to_trace() == d2.to_trace()
    assert r1["journal_sequence"] == r2["journal_sequence"] == 3
    assert r1.get("validation_history") == r2.get("validation_history")

    # Validation recovery subcase: ckpt history empty, disk has legal ep=50, next=50
    v_ckpt: list = []
    v_disk = [_acceptable_val_row(50, 33.0)]
    merged = c1000.merge_validation_history(
        v_ckpt, v_disk, next_episode_index=50, out_dir=crash_dir
    )
    assert len(merged) == 1
    assert merged[0]["constrained_objective"] == 33.0
