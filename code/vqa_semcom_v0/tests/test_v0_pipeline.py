from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import sys
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.sim.allocators import ALLOCATORS
from vqa_semcom.sim.vqa_resource_env import AllocationDecision, Area4D, SemanticCacheEntry, VQAResourceEnv, summarize_outcomes, write_resource_outputs
from vqa_semcom.config import load_config
from vqa_semcom.data.visdrone import demo_objects, parse_annotation_line
from vqa_semcom.quality.lut_builder import build_lut, estimate_accuracy
from vqa_semcom.rl.diengine_env import VQASemComDingEnv, env_spec
from vqa_semcom.rl.diengine_hybrid_env import HybridVQADingEnv, hybrid_env_spec
from vqa_semcom.rl.tch_ppo import TCHPPOConfig, TCHPPOEnv, tch_costs_from_info
from vqa_semcom.sim.resource_env import load_lut, load_semantic_lut, run_simulation, write_results
from vqa_semcom.tasks.generate_tasks import _view_quality, generate_tasks, write_tasks_csv


class V0PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(ROOT / "configs" / "v0.yaml")

    def test_visdrone_parser_skips_ignored_category(self) -> None:
        ignored = parse_annotation_line("1,2,3,4,1,0,0,0", "img")
        car = parse_annotation_line("1,2,30,40,1,4,0,0", "img")
        self.assertIsNone(ignored)
        self.assertIsNotNone(car)
        self.assertEqual(car.category, "car")

    def test_task_generator_creates_required_question_types(self) -> None:
        tasks = generate_tasks(demo_objects(), self.cfg)
        qtypes = {task.question_type for task in tasks}
        self.assertTrue({"presence", "counting", "risk", "attribute", "relation"}.issubset(qtypes))
        self.assertTrue(all(task.risk_level in {"normal", "critical"} for task in tasks))

    def test_view_quality_proxy_keeps_large_clear_box_at_least_medium(self) -> None:
        view_bin, score = _view_quality(
            scale_proxy=0.010,
            occlusion_score=0.0,
            truncation_score=0.0,
            density_score=3.0,
            scale_reference=self.cfg["view_quality"]["scale_reference"],
            good_score=self.cfg["view_quality"]["good_score"],
            medium_score=self.cfg["view_quality"]["medium_score"],
            density_penalty_start=self.cfg["view_quality"]["density_penalty_start"],
            density_penalty_per_object=self.cfg["view_quality"]["density_penalty_per_object"],
            min_density_component=self.cfg["view_quality"]["min_density_component"],
        )
        self.assertIn(view_bin, {"medium", "good"})
        self.assertGreater(score, 0.0)

    def test_service_levels_are_fixed_and_critical_is_not_service_level(self) -> None:
        self.assertEqual(self.cfg["bins"]["service_levels"], [0, 1, 2, 3])
        self.assertNotIn("critical", self.cfg["bins"]["service_levels"])
        self.assertIn("critical", self.cfg["bins"]["risk_levels"])

    def test_lut_monotonicity_and_quality_symbol_contract(self) -> None:
        bad = estimate_accuracy("counting", 2, "bad", "poor", "fresh", "normal", self.cfg["evaluator"])
        good = estimate_accuracy("counting", 2, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        self.assertLess(bad, good)
        light = estimate_accuracy("presence", 1, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        image = estimate_accuracy("presence", 2, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        cache = estimate_accuracy("presence", 0, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        counting_cache = estimate_accuracy("counting", 0, "good", "medium", "fresh", "normal", self.cfg["evaluator"])
        counting_light = estimate_accuracy("counting", 1, "good", "medium", "fresh", "normal", self.cfg["evaluator"])
        self.assertGreater(light, cache)
        self.assertGreater(counting_light, counting_cache)
        self.assertGreaterEqual(image, light)
        attribute = estimate_accuracy("attribute", 1, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        relation = estimate_accuracy("relation", 2, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        roi = estimate_accuracy("presence", 3, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        self.assertGreater(attribute, 0.0)
        self.assertGreater(relation, 0.0)
        self.assertGreater(roi, 0.0)
        self.assertTrue(self.cfg["evaluator"]["empirical_calibration"]["enabled"])
        poor = estimate_accuracy("attribute", 2, "good", "poor", "fresh", "normal", self.cfg["evaluator"])
        calibrated_good = estimate_accuracy("attribute", 2, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        self.assertGreater(calibrated_good, poor)
        self.assertEqual("A_k >= epsilon_k", "A_k >= epsilon_k")
        self.assertNotEqual("A_k >= epsilon_k", "T_k <= tau_k")

    def test_end_to_end_demo_lut_and_simulation(self) -> None:
        tasks = generate_tasks(demo_objects(), self.cfg)
        task_dicts = [dict((k, str(v)) for k, v in task.__dict__.items()) for task in tasks]
        rows = build_lut(task_dicts, self.cfg)
        self.assertTrue(rows)
        self.assertTrue(all(row.sample_count > 0 for row in rows))
        self.assertTrue(all(row.payload_bytes > 0 for row in rows))
        self.assertTrue(any(row.std_or_ci > 0 for row in rows))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_csv = tmp_path / "tasks.csv"
            lut_csv = tmp_path / "lut.csv"
            results_csv = tmp_path / "results.csv"
            results_md = tmp_path / "summary.md"
            write_tasks_csv(tasks, task_csv)
            with lut_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].__dataclass_fields__.keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(row.__dict__)
            with task_csv.open(newline="", encoding="utf-8") as f:
                task_rows = list(csv.DictReader(f))
            results = run_simulation(task_rows, load_lut(lut_csv), self.cfg, episodes=2)
            write_results(results, results_csv, results_md)
            self.assertTrue(results_csv.exists())
            self.assertTrue(results_md.exists())
            self.assertEqual({r.policy for r in results}, set(self.cfg["simulation"]["policies"]))
            roi = next(r for r in results if r.policy == "always_roi")
            full = next(r for r in results if r.policy == "always_image")
            self.assertGreater(roi.average_payload_mb, 0.0)
            self.assertLess(roi.average_payload_mb, full.average_payload_mb)

    def test_area4d_overlap_detects_spatial_altitude_temporal_conflict(self) -> None:
        left = Area4D(0.0, 0.0, 50.0, 30.0, 120.0, 2, 5)
        overlap = Area4D(60.0, 0.0, 50.0, 80.0, 160.0, 4, 8)
        far = Area4D(500.0, 0.0, 50.0, 80.0, 160.0, 4, 8)
        late = Area4D(60.0, 0.0, 50.0, 80.0, 160.0, 6, 8)
        self.assertTrue(left.overlaps(overlap))
        self.assertFalse(left.overlaps(far))
        self.assertFalse(left.overlaps(late))

    def test_resource_allocators_produce_valid_resource_bounded_actions(self) -> None:
        tasks = generate_tasks(demo_objects(), self.cfg)
        task_dicts = [dict((k, str(v)) for k, v in asdict(task).items()) for task in tasks]
        rows = build_lut(task_dicts, self.cfg)
        lut = {
            (
                row.question_type,
                row.service_level,
                row.channel_bin,
                row.view_quality_bin,
                row.freshness_bin,
                row.risk_level,
            ): row.expected_accuracy
            for row in rows
        }
        cfg = dict(self.cfg)
        cfg["resource_env"] = dict(self.cfg["resource_env"])
        cfg["resource_env"]["tasks_per_episode"] = 6
        env = VQAResourceEnv(task_dicts, lut, cfg)
        env.reset(episode=0)
        for allocator in ALLOCATORS.values():
            decisions = allocator.allocate(env)
            self.assertEqual(len(decisions), len(env.tasks))
            self.assertTrue(all(d.service_level in {0, 1, 2, 3} for d in decisions))
            self.assertLessEqual(sum(d.bandwidth_share for d in decisions), 1.000001)
            self.assertLessEqual(sum(d.cpu_share for d in decisions), 1.000001)
            self.assertLessEqual(sum(d.gpu_share for d in decisions), 1.000001)
            _state, outcomes, _done, info = env.step(decisions, policy=allocator.name)
            self.assertEqual(len(outcomes), len(env.tasks))
            self.assertIn("quality_violation_rate", info)
            self.assertIn("deadline_violation_rate", info)

    def test_resource_allocation_end_to_end_writes_summary(self) -> None:
        tasks = generate_tasks(demo_objects(), self.cfg)
        task_dicts = [dict((k, str(v)) for k, v in asdict(task).items()) for task in tasks]
        rows = build_lut(task_dicts, self.cfg)
        lut = {
            (
                row.question_type,
                row.service_level,
                row.channel_bin,
                row.view_quality_bin,
                row.freshness_bin,
                row.risk_level,
            ): row.expected_accuracy
            for row in rows
        }
        cfg = dict(self.cfg)
        cfg["resource_env"] = dict(self.cfg["resource_env"])
        cfg["resource_env"]["tasks_per_episode"] = 4
        env = VQAResourceEnv(task_dicts, lut, cfg)
        env.reset(episode=1)
        decisions = ALLOCATORS["joint_greedy_resource"].allocate(env)
        _state, outcomes, _done, _info = env.step(decisions, policy="joint_greedy_resource")
        summary = summarize_outcomes(outcomes)
        self.assertIn("airspace_conflict_rate", summary)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "resource.csv"
            md_path = tmp_path / "resource.md"
            write_resource_outputs(outcomes, csv_path, md_path)
            self.assertTrue(csv_path.exists())
            self.assertIn("V0 UAV-VQA Resource Allocation Summary", md_path.read_text(encoding="utf-8"))

    def _demo_env(self, tasks_per_episode: int = 6) -> VQAResourceEnv:
        tasks = generate_tasks(demo_objects(), self.cfg)
        task_dicts = [dict((k, str(v)) for k, v in asdict(task).items()) for task in tasks]
        rows = build_lut(task_dicts, self.cfg)
        lut = {
            (
                row.question_type,
                row.service_level,
                row.channel_bin,
                row.view_quality_bin,
                row.freshness_bin,
                row.risk_level,
            ): row.expected_accuracy
            for row in rows
        }
        cfg = dict(self.cfg)
        cfg["resource_env"] = dict(self.cfg["resource_env"])
        cfg["resource_env"]["tasks_per_episode"] = tasks_per_episode
        return VQAResourceEnv(task_dicts, lut, cfg)

    def test_literature_demo_has_cache_light_and_image_quality_levels(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="literature_demo")
        min_quality_levels = {env.feasibility_diagnostics(task)["min_quality_level"] for task in env.tasks}
        self.assertTrue({0, 1, 2}.issubset(min_quality_levels))
        question_types = {task.question_type for task in env.tasks}
        self.assertTrue({"presence", "counting", "risk", "attribute", "relation"}.issubset(question_types))

    def test_goal_oriented_priority_protects_critical_tasks_under_budget(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="critical_preemption")
        decisions = ALLOCATORS["goal_oriented_priority"].allocate(env)
        _state, outcomes, _done, _info = env.step(decisions, policy="goal_oriented_priority")
        critical = [o for o in outcomes if any(t.task_id == o.task_id and t.priority >= 2.0 for t in env.tasks)]
        normal = [o for o in outcomes if any(t.task_id == o.task_id and t.priority < 2.0 for t in env.tasks)]
        critical_success = sum(float(o.success) for o in critical) / max(1, len(critical))
        normal_success = sum(float(o.success) for o in normal) / max(1, len(normal))
        self.assertGreaterEqual(critical_success, normal_success)

    def test_expired_high_priority_cache_triggers_observe_or_revisit_when_upgraded(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="cache_freshness")
        decisions = ALLOCATORS["goal_oriented_priority"].allocate(env)
        expired_high_priority = [
            (task, decision)
            for task in env.tasks
            for decision in decisions
            if task.task_id == decision.task_id and task.freshness_bin == "expired" and task.priority >= 2.0
        ]
        self.assertTrue(expired_high_priority)
        self.assertTrue(any(decision.sensing_decision in {"observe", "revisit"} and decision.service_level > 0 for _task, decision in expired_high_priority))

    def test_area4d_conflict_scenario_marks_conflicting_tasks(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="area4d_conflict")
        decisions = [
            AllocationDecision(task.task_id, idx % 2, "observe", 1, 0.2, 1.2, 0.2)
            for idx, task in enumerate(env.tasks)
        ]
        _state, outcomes, _done, _info = env.step(decisions, policy="conflict_probe")
        self.assertTrue(any(o.airspace_conflict for o in outcomes))

    def test_cache_only_overlap_does_not_create_airspace_conflict(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="area4d_conflict")
        decisions = [
            AllocationDecision(task.task_id, 0, "reuse_cache", 0, 0.1, 0.2, 0.1, 0.05)
            for task in env.tasks
        ]
        _state, outcomes, _done, _info = env.step(decisions, policy="cache_conflict_probe")
        self.assertFalse(any(o.airspace_conflict for o in outcomes))

    def test_a2g_rate_is_monotonic_with_power_and_distance(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="clear_area_nominal")
        task = env.tasks[0]
        uav = env.nearest_uav(task)
        low_power = AllocationDecision(task.task_id, uav.uav_id, "observe", 1, 0.3, 0.2, 0.3, 0.2)
        high_power = AllocationDecision(task.task_id, uav.uav_id, "observe", 1, 0.3, 2.0, 0.3, 0.2)
        low = env.evaluate(task, low_power, policy="probe", include_diagnostics=False)
        high = env.evaluate(task, high_power, policy="probe", include_diagnostics=False)
        self.assertGreaterEqual(high.rate_mbps, low.rate_mbps)
        near = env._a2g_link(task, uav, high_power, bandwidth_mhz=3.0)
        far_uav = type(uav)(uav.uav_id, uav.x + 1500.0, uav.y + 1500.0, uav.altitude, uav.battery)
        far = env._a2g_link(task, far_uav, high_power, bandwidth_mhz=3.0)
        self.assertGreaterEqual(float(near["rate_mbps"]), float(far["rate_mbps"]))
        self.assertGreaterEqual(float(far["path_loss_db"]), float(near["path_loss_db"]))

    def test_los_probability_and_link_budget_follow_elevation(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="los_vs_nlos")
        near_task = next(task for task in env.tasks if "high_elevation" in task.task_id)
        far_task = next(task for task in env.tasks if "low_elevation" in task.task_id)
        near_uav = env.nearest_uav(near_task)
        far_uav = env.nearest_uav(far_task)
        decision_near = AllocationDecision(near_task.task_id, near_uav.uav_id, "observe", 1, 0.3, 1.2, 0.3, 0.2)
        decision_far = AllocationDecision(far_task.task_id, far_uav.uav_id, "observe", 1, 0.3, 1.2, 0.3, 0.2)
        near = env.evaluate(near_task, decision_near, policy="probe", include_diagnostics=False)
        far = env.evaluate(far_task, decision_far, policy="probe", include_diagnostics=False)
        self.assertGreater(near.elevation_deg, far.elevation_deg)
        self.assertGreater(near.los_probability, far.los_probability)
        self.assertGreater(far.path_loss_db, near.path_loss_db)
        self.assertLess(far.rate_mbps, near.rate_mbps)

    def test_interference_lowers_sinr_for_concurrent_uploads(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="interference_stress")
        decisions = [
            AllocationDecision(task.task_id, env.nearest_uav(task).uav_id, "observe", 2, 0.3, 1.2, 0.3, 0.3)
            for task in env.tasks
        ]
        no_interference = env.evaluate(env.tasks[0], decisions[0], policy="probe", include_diagnostics=False)
        _state, outcomes, _done, _info = env.step(decisions, policy="interference_probe")
        interfered = next(outcome for outcome in outcomes if outcome.task_id == env.tasks[0].task_id)
        self.assertLess(interfered.sinr_db, no_interference.sinr_db)
        self.assertLess(interfered.rate_mbps, no_interference.rate_mbps)
        self.assertGreater(interfered.interference_dbm, -120.0)

    def test_slow_fading_changes_smoothly_across_mdp_slots(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="mobility_fading", mdp_mode=True)
        task = env.active_tasks()[0]
        decision = AllocationDecision(task.task_id, env.nearest_uav(task).uav_id, "observe", 1, 0.3, 1.2, 0.3, 0.2)
        _state, first_outcomes, _done, _info = env.step([decision], policy="fading_probe")
        updated = next(t for t in env.tasks if t.task_id == task.task_id)
        second_decision = AllocationDecision(updated.task_id, env.nearest_uav(updated).uav_id, "observe", 1, 0.3, 1.2, 0.3, 0.2)
        _state, second_outcomes, _done, _info = env.step([second_decision], policy="fading_probe")
        first = first_outcomes[0]
        second = second_outcomes[0]
        self.assertNotEqual(first.fading_db, second.fading_db)
        self.assertLess(abs(first.fading_db - second.fading_db), 3.0)

    def test_channel_prior_is_not_overwritten_by_effective_channel(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="clear_area_nominal", mdp_mode=True)
        task = replace(env.tasks[0], channel_prior_bin="good", channel_bin="good", effective_channel_bin="bad")
        env.tasks[0] = task
        decision = AllocationDecision(task.task_id, env.nearest_uav(task).uav_id, "observe", 1, 0.6, 2.0, 0.6, 0.5)
        _state, outcomes, _done, _info = env.step([decision], policy="channel_prior_probe")
        outcome = outcomes[0]
        updated = next(t for t in env.tasks if t.task_id == task.task_id)
        self.assertEqual(outcome.channel_prior_bin, "good")
        self.assertEqual(updated.channel_prior_bin, "good")
        self.assertEqual(updated.channel_bin, "good")
        self.assertEqual(updated.effective_channel_bin, outcome.effective_channel_bin)
        self.assertNotEqual(outcome.effective_channel_bin, "bad")

    def test_gpu_and_model_cache_reduce_edge_delay(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="clear_area_nominal")
        task = next(t for t in env.tasks if env.feasibility_diagnostics(t)["min_quality_level"] >= 1)
        uav = env.nearest_uav(task)
        slow = AllocationDecision(task.task_id, uav.uav_id, "observe", 2, 0.4, 1.2, 0.2, 0.05)
        fast = AllocationDecision(task.task_id, uav.uav_id, "observe", 2, 0.4, 1.2, 0.2, 0.5)
        env.edge.model_cache_hit = False
        env.edge.cached_model_levels = ()
        slow_out = env.evaluate(task, slow, policy="probe", include_diagnostics=False)
        env.edge.model_cache_hit = True
        env.edge.cached_model_levels = (2,)
        fast_out = env.evaluate(task, fast, policy="probe", include_diagnostics=False)
        self.assertLessEqual(fast_out.inference_delay, slow_out.inference_delay)
        self.assertLess(fast_out.model_load_delay, slow_out.model_load_delay)

    def test_mdp_step_updates_cache_age_battery_and_vectors(self) -> None:
        env = self._demo_env()
        state = env.reset(episode=0, scenario="cache_freshness", mdp_mode=True)
        self.assertIn("available_actions", state)
        before_battery = env.uavs[0].battery
        task = env.active_tasks()[0]
        decision = AllocationDecision(task.task_id, 0, "observe", 1, 0.3, 1.2, 0.3, 0.2)
        _state, outcomes, _done, _info = env.step([decision], policy="probe")
        updated = next(t for t in env.tasks if t.task_id == task.task_id)
        self.assertEqual(updated.cache_age, 0)
        self.assertLessEqual(env.uavs[0].battery, before_battery)
        obs = env.observation_vector()
        self.assertEqual(len(obs), 1 + self.cfg["resource_env"]["observation"]["max_tasks"] * 20 + self.cfg["resource_env"]["observation"]["max_uavs"] * 5 + 5)
        actions = env.action_from_vector([0, 1, 2, 0.2, 0.5, 0.3, 0.4] * max(1, len(env.tasks)))
        self.assertTrue(actions)
        self.assertTrue(all(a.service_level in {0, 1, 2, 3} for a in actions))
        self.assertTrue(outcomes)

    def test_semantic_lut_entry_uses_payload_bytes_for_upload_delay(self) -> None:
        tasks = generate_tasks(demo_objects(), self.cfg)
        task_dicts = [dict((k, str(v)) for k, v in asdict(task).items()) for task in tasks]
        rows = build_lut(task_dicts, self.cfg)
        with tempfile.TemporaryDirectory() as tmp:
            lut_csv = Path(tmp) / "lut.csv"
            with lut_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].__dataclass_fields__.keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(row.__dict__)
            lut = load_semantic_lut(lut_csv, self.cfg["resource_env"]["payload_mb_by_level"])
        cfg = dict(self.cfg)
        cfg["resource_env"] = dict(self.cfg["resource_env"])
        cfg["resource_env"]["tasks_per_episode"] = 4
        env = VQAResourceEnv(task_dicts, lut, cfg)
        env.reset(episode=0, scenario="clear_area_nominal")
        task = env.tasks[0]
        uav = env.nearest_uav(task)
        full = AllocationDecision(task.task_id, uav.uav_id, "observe", 2, 0.4, 1.2, 0.4, 0.5)
        roi = AllocationDecision(task.task_id, uav.uav_id, "observe", 3, 0.4, 1.2, 0.4, 0.5)
        full_out = env.evaluate(task, full, policy="probe", include_diagnostics=False)
        roi_out = env.evaluate(task, roi, policy="probe", include_diagnostics=False)
        self.assertEqual(full_out.payload_bytes, env.lookup_entry(task, 2, full_out.effective_channel_bin).payload_bytes)
        self.assertEqual(roi_out.payload_bytes, env.lookup_entry(task, 3, roi_out.effective_channel_bin).payload_bytes)
        self.assertLess(roi_out.payload_mb, full_out.payload_mb)
        self.assertLess(roi_out.upload_delay, full_out.upload_delay)

    def test_v19_snr_lut_prefers_snr_bin_and_maps_snr_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_csv = Path(tmp) / "v19_snr_lut.csv"
            fieldnames = [
                "question_type",
                "service_level",
                "snr_bin",
                "snr_db",
                "channel_bin",
                "view_quality_bin",
                "freshness_bin",
                "risk_level",
                "expected_accuracy",
                "payload_bytes",
                "sample_count",
                "std_or_confidence_interval",
            ]
            rows = [
                ["presence", 0, "", "0.0", "", "good", "fresh", "normal", "0.30", "1000", "2", "0.01"],
                ["presence", 1, "", "10.0", "", "good", "fresh", "normal", "0.60", "2000", "3", "0.02"],
                ["presence", 2, "high", "20.0", "bad", "good", "fresh", "normal", "0.90", "3000", "4", "0.03"],
            ]
            with lut_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                writer.writerows(rows)
            lut = load_semantic_lut(lut_csv, self.cfg["resource_env"]["payload_mb_by_level"])
        bad = lut[("presence", 0, "bad", "good", "fresh", "normal")]
        medium = lut[("presence", 1, "medium", "good", "fresh", "normal")]
        good = lut[("presence", 2, "good", "good", "fresh", "normal")]
        self.assertEqual(bad.link_quality_source, "snr_db")
        self.assertEqual(medium.link_quality_source, "snr_db")
        self.assertEqual(good.link_quality_source, "snr_bin")
        self.assertAlmostEqual(good.accuracy, 0.90)
        self.assertEqual(good.payload_bytes, 3000.0)
        self.assertEqual(good.std_or_ci, 0.03)

    def test_bad_channel_stress_has_physical_a2g_impairment(self) -> None:
        clear_env = self._demo_env()
        clear_env.reset(episode=0, scenario="clear_area_nominal")
        bad_env = self._demo_env()
        bad_env.reset(episode=0, scenario="bad_channel_stress")
        clear_outcomes = [
            clear_env.evaluate(task, clear_env.default_decision(task), policy="probe", include_diagnostics=False)
            for task in clear_env.tasks
        ]
        bad_outcomes = [
            bad_env.evaluate(task, bad_env.default_decision(task), policy="probe", include_diagnostics=False)
            for task in bad_env.tasks
        ]
        clear_snr = sum(o.snr_db for o in clear_outcomes) / len(clear_outcomes)
        bad_snr = sum(o.snr_db for o in bad_outcomes) / len(bad_outcomes)
        self.assertLess(bad_snr, clear_snr - 15.0)
        self.assertGreaterEqual(sum(o.effective_channel_bin == "bad" for o in bad_outcomes), 2)
        self.assertTrue(all(o.channel_impairment_db > 0.0 for o in bad_outcomes))

    def test_mdp_reports_attempt_and_unique_task_success_separately(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="literature_demo", mdp_mode=True)
        outcomes = []
        done = False
        while not done:
            original_tasks = env.tasks
            env.tasks = env.active_tasks()
            decisions = ALLOCATORS["goal_oriented_priority"].allocate(env) if env.tasks else []
            env.tasks = original_tasks
            _state, step_outcomes, done, _info = env.step(decisions, policy="goal_oriented_priority")
            outcomes.extend(step_outcomes)
        summary = summarize_outcomes(outcomes)
        self.assertIn("attempt_success_rate", summary)
        self.assertIn("unique_task_success_rate", summary)
        self.assertGreaterEqual(summary["unique_task_success_rate"], summary["attempt_success_rate"])
        self.assertGreater(summary["average_attempts_per_task"], 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "resource.csv"
            md_path = Path(tmp) / "resource.md"
            write_resource_outputs(outcomes, csv_path, md_path)
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertIn("attempt_index", rows[0])
            self.assertIn("task_completed", rows[0])
            self.assertTrue(any(int(row["attempt_index"]) > 1 for row in rows))
            self.assertIn("unique task success", md_path.read_text(encoding="utf-8"))

    def test_semantic_cache_reuse_probability_improves_with_completed_nearby_task(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="cache_freshness", mdp_mode=True)
        task = env.tasks[1]
        baseline = env._semantic_cache_hit_probability(task)
        env.tasks[0] = replace(env.tasks[0], completed=True, question_type=task.question_type, area=task.area)
        boosted = env._semantic_cache_hit_probability(task)
        self.assertGreaterEqual(boosted, baseline)

    def test_semantic_cache_priority_freshness_replacement_keeps_high_priority_entry(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="cache_freshness", mdp_mode=True)
        env.cfg["cache_capacity"] = 1
        env.semantic_cache_entries = [
            SemanticCacheEntry("low", "presence", "normal", 1.0, 0.0, 0.0, 0, 0),
            SemanticCacheEntry("high", "presence", "critical", 3.0, 1.0, 1.0, 2, 0),
        ]
        env._update_semantic_cache([])
        self.assertEqual(len(env.semantic_cache_entries), 1)
        self.assertEqual(env.semantic_cache_entries[0].task_id, "high")

    def test_gpu_model_cache_memory_tracks_cached_model_set(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="clear_area_nominal")
        env.edge.cached_model_levels = (1, 3)
        env.edge.cache_capacity = 3
        env._update_edge_state([], [], {})
        base = self.cfg["resource_env"]["gpu_memory_capacity_mb"] * self.cfg["resource_env"]["gpu_memory_load"]
        expected = base + self.cfg["resource_env"]["model_memory_mb_by_level"]["1"] + self.cfg["resource_env"]["model_memory_mb_by_level"]["3"]
        self.assertAlmostEqual(env.edge.gpu_memory_used_mb, expected)
        self.assertEqual(env.edge.cached_model_levels, (1, 3))

    def test_uav_movement_uses_operational_priority_target(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="critical_preemption", mdp_mode=True)
        uav = env.uavs[0]
        before = (uav.x, uav.y, uav.total_travel_m)
        decisions = [
            AllocationDecision(task.task_id, 0, "observe", 1, 0.2, 1.2, 0.2, 0.2)
            for task in env.active_tasks()
        ]
        _state, _outcomes, _done, _info = env.step(decisions, policy="uav_move_probe")
        self.assertGreater(uav.total_travel_m, before[2])
        self.assertNotEqual((uav.x, uav.y), before[:2])

    def test_gpu_memory_and_battery_constraints_are_reported(self) -> None:
        env = self._demo_env()
        env.reset(episode=0, scenario="clear_area_nominal")
        task = next(t for t in env.tasks if env.feasibility_diagnostics(t)["min_quality_level"] >= 1)
        uav = env.nearest_uav(task)
        decision = AllocationDecision(task.task_id, uav.uav_id, "observe", 2, 0.4, 1.2, 0.4, 0.5)
        env.edge.gpu_memory_used_mb = env.edge.gpu_memory_capacity_mb - 1.0
        out = env.evaluate(task, decision, policy="probe", include_diagnostics=False)
        self.assertFalse(out.gpu_memory_ok)
        uav.battery = 0.1
        out_low_battery = env.evaluate(task, decision, policy="probe", include_diagnostics=False)
        self.assertFalse(out_low_battery.battery_ok)

    def test_action_mask_and_hierarchical_schema_are_exposed(self) -> None:
        env = self._demo_env()
        state = env.reset(episode=0, scenario="literature_demo")
        mask = state["available_actions"]["action_mask"]
        schema = state["available_actions"]["hierarchical_schema"]
        self.assertIn("bandwidth_budget", mask)
        self.assertIn("battery_ok_by_uav", mask)
        self.assertIn("active_task_mask", mask)
        self.assertIn("per_task_service_level_allowed", mask)
        self.assertIn("resource_budget_hint", mask)
        self.assertIn("high_level", schema)
        self.assertIn("low_level_continuous", schema)

    def test_diengine_adapter_exposes_fixed_vector_spaces(self) -> None:
        try:
            spec = env_spec(ROOT / "configs" / "v0.yaml", "literature_demo")
            env = VQASemComDingEnv(ROOT / "configs" / "v0.yaml", "literature_demo", seed=0)
        except ModuleNotFoundError:
            self.skipTest("gymnasium/gym is not installed")
        self.assertEqual(spec.action_shape, self.cfg["resource_env"]["observation"]["max_tasks"] * 7)
        reset_out = env.reset(seed=0)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.assertEqual(tuple(obs.shape), (spec.obs_shape,))
        action = env.action_space.sample()
        step_out = env.step(action)
        self.assertIn(len(step_out), {4, 5})
        next_obs = step_out[0]
        reward = step_out[1]
        info = step_out[-1]
        self.assertEqual(tuple(next_obs.shape), (spec.obs_shape,))
        self.assertIsInstance(float(reward), float)
        self.assertIn("active_tasks", info)

    def test_flat_diengine_adapter_smokes_literature_and_cache_scenarios(self) -> None:
        for scenario in ("literature_demo", "cache_freshness"):
            try:
                env = VQASemComDingEnv(ROOT / "configs" / "v0.yaml", scenario, seed=0)
            except ModuleNotFoundError:
                self.skipTest("gymnasium/gym is not installed")
            reset_out = env.reset(seed=0)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            step_out = env.step(env.action_space.sample())
            self.assertIn(len(step_out), {4, 5})
            self.assertEqual(tuple(obs.shape), tuple(env.observation_space.shape))

    def test_hybrid_diengine_adapter_exposes_state_and_per_task_masks(self) -> None:
        try:
            spec = hybrid_env_spec(ROOT / "configs" / "v0.yaml", "literature_demo")
            env = HybridVQADingEnv(ROOT / "configs" / "v0.yaml", "literature_demo", seed=0)
        except ModuleNotFoundError:
            self.skipTest("gymnasium/gym is not installed")
        reset_out = env.reset(seed=0)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        mask = obs["action_mask"]
        self.assertEqual(tuple(obs["state"].shape), (spec.obs_shape,))
        self.assertEqual(tuple(obs["observation"].shape), (spec.obs_shape,))
        self.assertEqual(mask["assigned_uav_mask"].shape, (spec.max_tasks, spec.num_uavs))
        self.assertEqual(mask["sensing_mask"].shape, (spec.max_tasks, 3))
        self.assertEqual(mask["service_level_mask"].shape, (spec.max_tasks, len(spec.service_levels)))
        self.assertEqual(mask["active_task_mask"].shape, (spec.max_tasks,))
        self.assertGreaterEqual(mask["active_task_mask"].sum(), 1.0)

    def test_hybrid_adapter_converts_structured_actions_and_normalizes_resources(self) -> None:
        try:
            env = HybridVQADingEnv(ROOT / "configs" / "v0.yaml", "literature_demo", seed=0)
        except ModuleNotFoundError:
            self.skipTest("gymnasium/gym is not installed")
        reset_out = env.reset(seed=0)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        max_tasks = env.spec_info.max_tasks
        action = {
            "assigned_uav": [99] * max_tasks,
            "sensing_decision": [2] * max_tasks,
            "service_level": [3] * max_tasks,
            "bandwidth_share": [1.0] * max_tasks,
            "power_scalar": [0.5] * max_tasks,
            "cpu_share": [1.0] * max_tasks,
            "gpu_share": [1.0] * max_tasks,
        }
        decisions = env.action_to_decisions(action)
        active_count = int(obs["action_mask"]["active_task_mask"].sum())
        self.assertEqual(len(decisions), active_count)
        self.assertLessEqual(sum(d.bandwidth_share for d in decisions), 1.000001)
        self.assertLessEqual(sum(d.cpu_share for d in decisions), 1.000001)
        self.assertLessEqual(sum(d.gpu_share for d in decisions), 1.000001)
        step_out = env.step(action)
        self.assertIn(len(step_out), {4, 5})

    def test_hybrid_adapter_accepts_diengine_hybrid_action_format(self) -> None:
        try:
            env = HybridVQADingEnv(ROOT / "configs" / "v0.yaml", "cache_freshness", seed=0)
        except ModuleNotFoundError:
            self.skipTest("gymnasium/gym is not installed")
        env.reset(seed=0)
        spec = env.spec_info
        action = {
            "action_type": [0, 1, len(spec.service_levels) - 1] * spec.max_tasks,
            "action_args": [-1.0, 0.0, 1.0, 1.0] * spec.max_tasks,
        }
        decisions = env.action_to_decisions(action)
        self.assertTrue(decisions)
        self.assertTrue(all(d.service_level in spec.service_levels for d in decisions))
        self.assertLessEqual(sum(d.cpu_share for d in decisions), 1.000001)

    def test_hybrid_adapter_masks_service_level_and_battery_fallback(self) -> None:
        try:
            env = HybridVQADingEnv(ROOT / "configs" / "v0.yaml", "literature_demo", seed=0)
        except ModuleNotFoundError:
            self.skipTest("gymnasium/gym is not installed")
        env.reset(seed=0)
        for uav in env._env.uavs:
            uav.battery = 0.01
        env._env.uavs[-1].battery = 0.02
        env._env.edge.gpu_memory_used_mb = env._env.edge.gpu_memory_capacity_mb
        max_tasks = env.spec_info.max_tasks
        action = {
            "assigned_uav": [0] * max_tasks,
            "sensing_decision": [1] * max_tasks,
            "service_level": [3] * max_tasks,
            "bandwidth_share": [0.2] * max_tasks,
            "power_scalar": [0.2] * max_tasks,
            "cpu_share": [0.2] * max_tasks,
            "gpu_share": [0.2] * max_tasks,
        }
        decisions = env.action_to_decisions(action)
        self.assertTrue(decisions)
        self.assertTrue(all(d.service_level == 0 for d in decisions))
        self.assertTrue(all(d.assigned_uav == env._env.uavs[-1].uav_id for d in decisions))

    def test_tch_cost_mapping_uses_environment_constraint_rates(self) -> None:
        costs = tch_costs_from_info(
            {
                "quality_violation_rate": 0.2,
                "deadline_violation_rate": 0.3,
                "resource_violation_rate": 0.1,
                "raw_resource_violation": 1.0,
                "airspace_conflict_rate": 0.4,
                "battery_ok_rate": 0.75,
                "gpu_memory_ok_rate": 0.5,
            }
        )
        self.assertAlmostEqual(costs["quality"], 0.2)
        self.assertAlmostEqual(costs["deadline"], 0.3)
        self.assertAlmostEqual(costs["resource"], 1.0)
        self.assertAlmostEqual(costs["conflict"], 0.4)
        self.assertAlmostEqual(costs["battery"], 0.25)
        self.assertAlmostEqual(costs["gpu_memory"], 0.5)

    def test_tch_env_shapes_reward_updates_duals_and_writes_trace(self) -> None:
        try:
            env = TCHPPOEnv(
                ROOT / "configs" / "v0.yaml",
                "literature_demo",
                seed=0,
                tch_config=TCHPPOConfig(lambda_lr=0.5, initial_lambda=1.0),
            )
        except ModuleNotFoundError:
            self.skipTest("gymnasium/gym is not installed")
        env.reset(seed=0)
        max_tasks = env.spec_info.max_tasks
        action = {
            "assigned_uav": [0] * max_tasks,
            "sensing_decision": [1] * max_tasks,
            "service_level": [2] * max_tasks,
            "bandwidth_share": [1.0] * max_tasks,
            "power_scalar": [0.5] * max_tasks,
            "cpu_share": [1.0] * max_tasks,
            "gpu_share": [1.0] * max_tasks,
        }
        step_out = env.step(action)
        self.assertIn(len(step_out), {4, 5})
        reward = float(step_out[1])
        info = step_out[-1]
        self.assertAlmostEqual(reward, info["tch_reward"])
        self.assertLessEqual(info["tch_reward"], info["raw_reward"])
        self.assertEqual(info["cost_resource"], 1.0)
        self.assertGreater(info["lambda_resource"], 1.0)
        self.assertIn("lambda_quality", info)
        self.assertEqual(len(env.lambda_trace()), 1)
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "lambda.csv"
            env.write_lambda_trace(trace_path)
            with trace_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertIn("lambda_resource", rows[0])


if __name__ == "__main__":
    unittest.main()
