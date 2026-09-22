from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from vqa_semcom.config import ensure_parent
from vqa_semcom.sim.resource_env import SemanticServiceEntry, load_semantic_lut, read_csv


LUTKey = tuple[str, int, str, str, str, str]
SemanticLUT = dict[LUTKey, SemanticServiceEntry]


@dataclass(frozen=True)
class Area4D:
    center_x: float
    center_y: float
    radius: float
    altitude_min: float
    altitude_max: float
    start_slot: int
    end_slot: int

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.center_x - x, self.center_y - y)

    def overlaps(self, other: "Area4D") -> bool:
        spatial = self.distance_to(other.center_x, other.center_y) <= self.radius + other.radius
        altitude = self.altitude_min <= other.altitude_max and other.altitude_min <= self.altitude_max
        temporal = self.start_slot <= other.end_slot and other.start_slot <= self.end_slot
        return spatial and altitude and temporal


@dataclass(frozen=True)
class VQATaskState:
    task_id: str
    image_id: str
    question_type: str
    question: str
    risk_level: str
    epsilon_k: float
    tau_k: float
    priority: float
    generation_slot: int
    cache_age: int
    freshness_bin: str
    view_quality_bin: str
    channel_bin: str
    area: Area4D
    channel_prior_bin: str = ""
    effective_channel_bin: str = ""
    arrival_slot: int = 0
    completed: bool = False
    channel_impairment_db: float = 0.0
    channel_gain_db: float = 0.0
    snr_db: float = 0.0
    rate_mbps: float = 0.0
    fading_db: float = 0.0


@dataclass
class UAVState:
    uav_id: int
    x: float
    y: float
    altitude: float
    battery: float
    assigned_load: int = 0
    total_travel_m: float = 0.0
    utilization: float = 0.0


@dataclass
class EdgeState:
    cpu_capacity: float
    gpu_capacity: float
    load: float
    model_cache_hit: bool
    gpu_load: float = 0.0
    cached_models: int = 0
    cache_capacity: int = 0
    gpu_memory_capacity_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    queued_batches: int = 0
    cached_model_levels: tuple[int, ...] = ()


@dataclass(frozen=True)
class AllocationDecision:
    task_id: str
    assigned_uav: int
    sensing_decision: str
    service_level: int
    bandwidth_share: float
    power_w: float
    cpu_share: float
    gpu_share: float = 0.0


@dataclass(frozen=True)
class AllocationOutcome:
    policy: str
    episode: int
    slot: int
    task_id: str
    assigned_uav: int
    sensing_decision: str
    service_level: int
    bandwidth_share: float
    power_w: float
    cpu_share: float
    expected_accuracy: float
    delay: float
    energy: float
    quality_ok: bool
    deadline_ok: bool
    resource_ok: bool
    airspace_conflict: bool
    success: bool
    reward: float
    semantic_utility: float
    min_quality_level: int
    min_delay_level: int
    feasible_service_levels: str
    feasibility_class: str
    question_type: str = ""
    question: str = ""
    risk_level: str = ""
    priority: float = 1.0
    tau_k: float = 0.0
    epsilon_k: float = 0.0
    gpu_share: float = 0.0
    channel_gain_db: float = 0.0
    snr_db: float = 0.0
    rate_mbps: float = 0.0
    travel_delay: float = 0.0
    sensing_delay: float = 0.0
    upload_delay: float = 0.0
    queue_delay: float = 0.0
    model_load_delay: float = 0.0
    inference_delay: float = 0.0
    travel_energy: float = 0.0
    tx_energy: float = 0.0
    sensing_energy: float = 0.0
    compute_energy: float = 0.0
    battery_cost: float = 0.0
    effective_channel_bin: str = ""
    channel_prior_bin: str = ""
    channel_impairment_db: float = 0.0
    task_completed: bool = False
    attempt_index: int = 1
    distance_3d_m: float = 0.0
    elevation_deg: float = 0.0
    los_probability: float = 0.0
    path_loss_db: float = 0.0
    interference_dbm: float = -120.0
    sinr_db: float = 0.0
    fading_db: float = 0.0
    cache_hit_probability: float = 0.0
    semantic_cache_hit: bool = False
    accuracy_gain: float = 0.0
    payload_mb: float = 0.0
    payload_bytes: float = 0.0
    lut_missing: bool = False
    semantic_efficiency: float = 0.0
    utility_per_latency: float = 0.0
    gpu_memory_required_mb: float = 0.0
    gpu_memory_ok: bool = True
    edge_batch_size: int = 1
    battery_ok: bool = True
    battery_remaining: float = 0.0
    uav_utilization: float = 0.0
    conflict_resolution: str = "none"


@dataclass(frozen=True)
class SemanticCacheEntry:
    task_id: str
    question_type: str
    risk_level: str
    priority: float
    center_x: float
    center_y: float
    cache_age: int
    updated_slot: int


DEFAULT_RESOURCE_ENV: dict[str, Any] = {
    "seed": 17,
    "num_uavs": 2,
    "tasks_per_episode": 24,
    "episode_slots": 6,
    "scenario": "literature_demo",
    "cell_radius_m": 1000.0,
    "uav_altitude_m": 80.0,
    "initial_battery": 100.0,
    "uav_speed_mps": 18.0,
    "observe_travel_fraction": 0.05,
    "total_bandwidth_mhz": 12.0,
    "edge_cpu_capacity": 10.0,
    "edge_gpu_capacity": 4.0,
    "cache_capacity": 64,
    "freshness_slots": {"fresh": 1, "stale": 3},
    "payload_mb_by_level": {"0": 0.01, "1": 0.25, "2": 2.0, "3": 0.75},
    "sensing_time_by_level": {"0": 0.0, "1": 0.35, "2": 0.9, "3": 0.65},
    "workload_by_level": {"0": 0.15, "1": 1.2, "2": 3.2, "3": 2.2},
    "gpu_workload_by_level": {"0": 0.02, "1": 0.35, "2": 1.1, "3": 0.85},
    "channel_rate_factor": {"bad": 0.7, "medium": 1.2, "good": 2.0},
    "a2g": {
        "enabled": True,
        "carrier_mhz": 2400.0,
        "reference_distance_m": 1.0,
        "reference_gain_db": -40.0,
        "path_loss_exponent": 2.2,
        "noise_figure_db": 7.0,
        "excess_loss_db": 4.0,
        "los_excess_loss_db": 1.0,
        "nlos_excess_loss_db": 10.0,
        "los_a": 9.61,
        "los_b": 0.16,
        "fading_mode": "slow_fading",
        "slow_fading_std_db": 2.0,
        "fast_fading_std_db": 5.0,
        "fading_correlation": 0.85,
        "interference_enabled": True,
        "interference_floor_dbm": -120.0,
        "interference_overlap_scale": 0.02,
        "bad_snr_db": 5.0,
        "good_snr_db": 15.0,
    },
    "power_levels_w": [0.2, 0.6, 1.2, 2.0],
    "bandwidth_shares": [0.15, 0.25, 0.4, 0.6],
    "cpu_shares": [0.15, 0.25, 0.4, 0.6],
    "gpu_shares": [0.05, 0.15, 0.3, 0.5],
    "queue_delay_scale": 0.6,
    "gpu_queue_delay_scale": 0.25,
    "model_load_delay": 0.4,
    "model_cache_capacity": 3,
    "model_cache_hit_delay_scale": 0.15,
    "gpu_memory_capacity_mb": 8192.0,
    "gpu_memory_load": 0.25,
    "model_memory_mb_by_level": {"0": 256.0, "1": 1536.0, "2": 4096.0, "3": 3072.0},
    "batching_delay_scale": 0.08,
    "semantic_cache": {
        "fresh_hit_probability": 0.96,
        "stale_hit_probability": 0.68,
        "expired_hit_probability": 0.28,
        "spatial_reuse_radius_m": 140.0,
        "reuse_boost": 0.18,
        "policy": "priority_freshness",
    },
    "propulsion_energy_per_m": 0.003,
    "hover_energy_per_slot": 0.03,
    "return_energy_reserve": 5.0,
    "battery_failure_threshold": 1.0,
    "sensing_energy_by_level": {"0": 0.0, "1": 0.2, "2": 0.55, "3": 0.4},
    "compute_energy_per_workload": 0.08,
    "gpu_energy_per_workload": 0.05,
    "battery_cost_scale": 0.02,
    "observation": {"max_tasks": 12, "max_uavs": 4},
    "reward_weights": {
        "success": 4.0,
        "delay": 0.25,
        "energy": 0.15,
        "payload": 0.05,
        "quality_violation": 2.0,
        "deadline_violation": 1.5,
        "airspace_conflict": 1.0,
    },
    "scenario_task_count": 6,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resource_env_config(cfg: dict[str, Any]) -> dict[str, Any]:
    env_cfg = _deep_merge(DEFAULT_RESOURCE_ENV, cfg.get("resource_env", {}))
    env_cfg["service_levels"] = list(cfg.get("bins", {}).get("service_levels", env_cfg.get("service_levels", [0, 1, 2, 3])))
    return env_cfg


def freshness_from_age(age: int, env_cfg: dict[str, Any]) -> str:
    thresholds = env_cfg["freshness_slots"]
    if age <= int(thresholds["fresh"]):
        return "fresh"
    if age <= int(thresholds["stale"]):
        return "stale"
    return "expired"


class VQAResourceEnv:
    def __init__(
        self,
        tasks: list[dict[str, str]],
        lut: dict[LUTKey, float | SemanticServiceEntry],
        cfg: dict[str, Any],
        seed: int | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("VQAResourceEnv needs at least one task")
        self.raw_tasks = tasks
        self.cfg = resource_env_config(cfg)
        self.lut = self._normalize_lut(lut)
        self.rng = random.Random(int(self.cfg["seed"] if seed is None else seed))
        self.episode = -1
        self.slot = 0
        self.uavs: list[UAVState] = []
        self.edge = EdgeState(0.0, 0.0, 0.0, False)
        self.tasks: list[VQATaskState] = []
        self.semantic_cache_entries: list[SemanticCacheEntry] = []
        self.mdp_mode = False

    def _normalize_lut(self, lut: dict[LUTKey, float | SemanticServiceEntry]) -> SemanticLUT:
        normalized: SemanticLUT = {}
        for key, value in lut.items():
            if isinstance(value, SemanticServiceEntry):
                normalized[key] = value
                continue
            fallback_mb = float(self.cfg["payload_mb_by_level"].get(str(key[1]), 0.0))
            normalized[key] = SemanticServiceEntry(
                accuracy=float(value),
                payload_bytes=fallback_mb * 1_000_000.0,
            )
        return normalized

    def service_levels(self) -> list[int]:
        return [int(level) for level in self.cfg.get("service_levels", [0, 1, 2, 3])]

    def reset(self, episode: int = 0, scenario: str | None = None, mdp_mode: bool = False) -> dict[str, Any]:
        self.episode = episode
        self.slot = 0
        self.mdp_mode = mdp_mode
        self.rng.seed(int(self.cfg["seed"]) + episode)
        self.uavs = self._init_uavs()
        self.edge = self._sample_edge()
        self.semantic_cache_entries = []
        self.tasks = self._sample_tasks(scenario or str(self.cfg.get("scenario", "sampled")))
        self.tasks = [self._with_initial_link(task) for task in self.tasks]
        return self.state_dict()

    def available_actions(self) -> dict[str, Any]:
        return {
            "uav_ids": [u.uav_id for u in self.uavs],
            "sensing_decisions": ["reuse_cache", "observe", "revisit"],
            "service_levels": self.service_levels(),
            "bandwidth_shares": list(self.cfg["bandwidth_shares"]),
            "power_levels_w": list(self.cfg["power_levels_w"]),
            "cpu_shares": list(self.cfg["cpu_shares"]),
            "gpu_shares": list(self.cfg["gpu_shares"]),
            "action_mask": self.action_mask(),
            "hierarchical_schema": self.hierarchical_action_schema(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "slot": self.slot,
            "uavs": [asdict(u) for u in self.uavs],
            "edge": asdict(self.edge),
            "tasks": [self._task_to_dict(t) for t in self.tasks],
            "available_actions": self.available_actions(),
        }

    def feasibility_diagnostics(self, task: VQATaskState) -> dict[str, Any]:
        levels = self.service_levels()
        quality_levels = [level for level in levels if self.lookup_accuracy(task, level) >= task.epsilon_k]
        delay_levels: list[int] = []
        feasible_levels: list[int] = []
        for level in levels:
            decision = self.default_decision(task)
            decision = AllocationDecision(
                task_id=task.task_id,
                assigned_uav=decision.assigned_uav,
                sensing_decision="reuse_cache" if level == 0 else "observe",
                service_level=level,
                bandwidth_share=max(float(x) for x in self.cfg["bandwidth_shares"]),
                power_w=max(float(x) for x in self.cfg["power_levels_w"]),
                cpu_share=max(float(x) for x in self.cfg["cpu_shares"]),
                gpu_share=max(float(x) for x in self.cfg["gpu_shares"]),
            )
            outcome = self.evaluate(task, decision, policy="feasibility", resource_ok=True, airspace_conflict=False, include_diagnostics=False)
            if outcome.delay <= task.tau_k:
                delay_levels.append(level)
            if outcome.quality_ok and outcome.deadline_ok:
                feasible_levels.append(level)
        if feasible_levels:
            feasibility_class = "feasible"
        elif quality_levels:
            feasibility_class = "deadline_limited"
        else:
            feasibility_class = "quality_limited"
        return {
            "min_quality_level": min(quality_levels) if quality_levels else -1,
            "min_delay_level": min(delay_levels) if delay_levels else -1,
            "feasible_service_levels": feasible_levels,
            "feasibility_class": feasibility_class,
        }

    def step(self, decisions: list[AllocationDecision], policy: str = "external") -> tuple[dict[str, Any], list[AllocationOutcome], bool, dict[str, Any]]:
        by_task = {d.task_id: d for d in decisions}
        outcomes: list[AllocationOutcome] = []
        resource_ok = self._resource_ok(decisions)
        conflict_task_ids = self._conflict_task_ids(decisions)
        active = self.active_tasks()
        eval_decisions = [by_task.get(task.task_id, self.default_decision(task)) for task in active]
        interference_by_task = self._interference_by_task(active, eval_decisions)
        batch_size_by_level: dict[int, int] = {}
        utilization_by_uav: dict[int, float] = {}
        for decision in eval_decisions:
            utilization_by_uav[decision.assigned_uav] = utilization_by_uav.get(decision.assigned_uav, 0.0) + 1.0 / max(1, len(active))
            if int(decision.service_level) > 0:
                batch_size_by_level[int(decision.service_level)] = batch_size_by_level.get(int(decision.service_level), 0) + 1

        for task in active:
            decision = by_task.get(task.task_id)
            if decision is None:
                decision = self.default_decision(task)
            outcome = self.evaluate(
                task,
                decision,
                policy=policy,
                resource_ok=resource_ok,
                airspace_conflict=task.task_id in conflict_task_ids,
                interference_dbm=interference_by_task.get(task.task_id, float(self.cfg["a2g"]["interference_floor_dbm"])),
                edge_batch_size=max(1, batch_size_by_level.get(int(decision.service_level), 1)),
                uav_utilization=utilization_by_uav.get(decision.assigned_uav, 0.0),
            )
            outcomes.append(outcome)

        for uav in self.uavs:
            assigned = [o for o in outcomes if o.assigned_uav == uav.uav_id]
            uav.assigned_load = len(assigned)
            uav.battery = max(0.0, uav.battery - sum(o.battery_cost for o in assigned))
            uav.utilization = len(assigned) / max(1, len(active))
            self._advance_uav_position(uav, assigned)

        self._update_edge_state(decisions, outcomes, batch_size_by_level)
        self._update_semantic_cache(outcomes)
        self.tasks = self._advance_tasks(outcomes)
        self.slot += 1
        done = self.slot >= int(self.cfg["episode_slots"]) or (self.mdp_mode and all(t.completed for t in self.tasks))
        info = summarize_outcomes(outcomes)
        return self.state_dict(), outcomes, done, info

    def active_tasks(self) -> list[VQATaskState]:
        if not self.mdp_mode:
            return list(self.tasks)
        return [task for task in self.tasks if not task.completed and task.arrival_slot <= self.slot]

    def default_decision(self, task: VQATaskState) -> AllocationDecision:
        nearest = self.nearest_uav(task)
        return AllocationDecision(
            task_id=task.task_id,
            assigned_uav=nearest.uav_id,
            sensing_decision="reuse_cache",
            service_level=0,
            bandwidth_share=float(self.cfg["bandwidth_shares"][0]),
            power_w=float(self.cfg["power_levels_w"][0]),
            cpu_share=float(self.cfg["cpu_shares"][0]),
            gpu_share=float(self.cfg["gpu_shares"][0]),
        )

    def evaluate(
        self,
        task: VQATaskState,
        decision: AllocationDecision,
        policy: str,
        resource_ok: bool = True,
        airspace_conflict: bool = False,
        include_diagnostics: bool = True,
        interference_dbm: float | None = None,
        edge_batch_size: int = 1,
        uav_utilization: float | None = None,
    ) -> AllocationOutcome:
        if int(decision.service_level) not in set(self.service_levels()):
            raise ValueError(f"invalid service level: {decision.service_level}")
        uav = self.uavs[max(0, min(decision.assigned_uav, len(self.uavs) - 1))]
        level = int(decision.service_level)
        freshness = task.freshness_bin
        cache_hit_probability = self._semantic_cache_hit_probability(task)
        probe_components = self._service_components(task, uav, decision, interference_dbm=interference_dbm, edge_batch_size=edge_batch_size)
        entry = self.lookup_entry(task, level, str(probe_components["channel_bin"]), freshness, cache_hit_probability)
        components = self._service_components(
            task,
            uav,
            decision,
            interference_dbm=interference_dbm,
            edge_batch_size=edge_batch_size,
            payload_bytes=entry.payload_bytes,
        )
        accuracy = entry.accuracy
        cache_entry = self.lookup_entry(task, 0, str(components["channel_bin"]), freshness, cache_hit_probability)
        cache_accuracy = cache_entry.accuracy
        delay = float(components["delay"])
        energy = float(components["energy"])
        quality_ok = accuracy >= task.epsilon_k
        deadline_ok = delay <= task.tau_k
        battery_ok = self._battery_ok(uav, float(components["battery_cost"]))
        gpu_memory_ok = bool(components["gpu_memory_ok"])
        success = quality_ok and deadline_ok and resource_ok and not airspace_conflict and battery_ok and gpu_memory_ok
        reward = self._reward(task, delay, energy, float(components["payload_mb"]), quality_ok, deadline_ok, success, airspace_conflict)
        diagnostics = (
            self.feasibility_diagnostics(task)
            if include_diagnostics
            else {
                "min_quality_level": -1,
                "min_delay_level": -1,
                "feasible_service_levels": [],
                "feasibility_class": "not_evaluated",
            }
        )
        semantic_utility = task.priority * accuracy * float(quality_ok)
        return AllocationOutcome(
            policy=policy,
            episode=self.episode,
            slot=self.slot,
            task_id=task.task_id,
            assigned_uav=uav.uav_id,
            sensing_decision=decision.sensing_decision,
            service_level=level,
            bandwidth_share=round(float(decision.bandwidth_share), 6),
            power_w=round(float(decision.power_w), 6),
            cpu_share=round(float(decision.cpu_share), 6),
            expected_accuracy=round(accuracy, 6),
            delay=round(delay, 6),
            energy=round(energy, 6),
            quality_ok=quality_ok,
            deadline_ok=deadline_ok,
            resource_ok=resource_ok,
            airspace_conflict=airspace_conflict,
            success=success,
            reward=round(reward, 6),
            semantic_utility=round(semantic_utility, 6),
            min_quality_level=int(diagnostics["min_quality_level"]),
            min_delay_level=int(diagnostics["min_delay_level"]),
            feasible_service_levels=";".join(str(x) for x in diagnostics["feasible_service_levels"]),
            feasibility_class=str(diagnostics["feasibility_class"]),
            question_type=task.question_type,
            question=task.question,
            risk_level=task.risk_level,
            priority=round(float(task.priority), 6),
            tau_k=round(float(task.tau_k), 6),
            epsilon_k=round(float(task.epsilon_k), 6),
            gpu_share=round(float(decision.gpu_share), 6),
            channel_gain_db=round(float(components["channel_gain_db"]), 6),
            snr_db=round(float(components["snr_db"]), 6),
            rate_mbps=round(float(components["rate_mbps"]), 6),
            travel_delay=round(float(components["travel_delay"]), 6),
            sensing_delay=round(float(components["sensing_delay"]), 6),
            upload_delay=round(float(components["upload_delay"]), 6),
            queue_delay=round(float(components["queue_delay"]), 6),
            model_load_delay=round(float(components["model_load_delay"]), 6),
            inference_delay=round(float(components["inference_delay"]), 6),
            travel_energy=round(float(components["travel_energy"]), 6),
            tx_energy=round(float(components["tx_energy"]), 6),
            sensing_energy=round(float(components["sensing_energy"]), 6),
            compute_energy=round(float(components["compute_energy"]), 6),
            battery_cost=round(float(components["battery_cost"]), 6),
            effective_channel_bin=str(components["channel_bin"]),
            channel_prior_bin=task.channel_prior_bin or task.channel_bin,
            channel_impairment_db=round(float(task.channel_impairment_db), 6),
            task_completed=bool(task.completed or success),
            distance_3d_m=round(float(components["distance_3d_m"]), 6),
            elevation_deg=round(float(components["elevation_deg"]), 6),
            los_probability=round(float(components["los_probability"]), 6),
            path_loss_db=round(float(components["path_loss_db"]), 6),
            interference_dbm=round(float(components["interference_dbm"]), 6),
            sinr_db=round(float(components["sinr_db"]), 6),
            fading_db=round(float(components["fading_db"]), 6),
            cache_hit_probability=round(float(cache_hit_probability), 6),
            semantic_cache_hit=bool(cache_hit_probability >= 0.5 and level == 0),
            accuracy_gain=round(max(0.0, accuracy - cache_accuracy), 6),
            payload_mb=round(float(components["payload_mb"]), 6),
            payload_bytes=round(float(components["payload_bytes"]), 6),
            lut_missing=bool(entry.missing),
            semantic_efficiency=round(max(0.0, accuracy - cache_accuracy) / max(1e-9, float(components["payload_mb"])), 6),
            utility_per_latency=round(semantic_utility / max(1e-9, delay), 6),
            gpu_memory_required_mb=round(float(components["gpu_memory_required_mb"]), 6),
            gpu_memory_ok=gpu_memory_ok,
            edge_batch_size=int(edge_batch_size),
            battery_ok=battery_ok,
            battery_remaining=round(max(0.0, uav.battery - float(components["battery_cost"])), 6),
            uav_utilization=round(float(uav_utilization if uav_utilization is not None else uav.utilization), 6),
            conflict_resolution="none" if not airspace_conflict else self._conflict_resolution(task, decision),
        )

    def lookup_accuracy(self, task: VQATaskState, service_level: int, freshness: str | None = None) -> float:
        return self.lookup_entry(
            task,
            service_level,
            task.effective_channel_bin or task.channel_bin,
            freshness or task.freshness_bin,
            self._semantic_cache_hit_probability(task),
        ).accuracy

    def lookup_entry(
        self,
        task: VQATaskState,
        service_level: int,
        channel_bin: str | None = None,
        freshness: str | None = None,
        cache_hit_probability: float = 1.0,
    ) -> SemanticServiceEntry:
        channel_bin = channel_bin or task.effective_channel_bin or task.channel_bin
        freshness = freshness or task.freshness_bin
        key = (
            task.question_type,
            int(service_level),
            channel_bin,
            task.view_quality_bin,
            freshness,
            task.risk_level,
        )
        fallback_payload = float(self.cfg["payload_mb_by_level"].get(str(service_level), 0.0)) * 1_000_000.0
        entry = self.lut.get(key, SemanticServiceEntry(0.0, fallback_payload, missing=True))
        acc = float(entry.accuracy)
        if int(service_level) == 0:
            acc *= 0.85 + 0.15 * max(0.0, min(1.0, cache_hit_probability))
        return SemanticServiceEntry(
            accuracy=acc,
            payload_bytes=float(entry.payload_bytes if entry.payload_bytes > 0 else fallback_payload),
            sample_count=entry.sample_count,
            std_or_ci=entry.std_or_ci,
            missing=entry.missing,
        )

    def nearest_uav(self, task: VQATaskState) -> UAVState:
        return min(self.uavs, key=lambda u: task.area.distance_to(u.x, u.y))

    @staticmethod
    def _requires_operational_intent(decision: AllocationDecision) -> bool:
        return int(decision.service_level) > 0 and decision.sensing_decision in {"observe", "revisit"}

    def _advance_uav_position(self, uav: UAVState, assigned: list[AllocationOutcome]) -> None:
        operational = [outcome for outcome in assigned if outcome.service_level > 0 and outcome.sensing_decision in {"observe", "revisit"}]
        if not operational:
            return
        task_by_id = {task.task_id: task for task in self.tasks}
        candidates = [task_by_id[outcome.task_id] for outcome in operational if outcome.task_id in task_by_id]
        if not candidates:
            return
        target = min(candidates, key=lambda task: (-task.priority, task.area.distance_to(uav.x, uav.y)))
        dist = target.area.distance_to(uav.x, uav.y)
        step_dist = min(dist, float(self.cfg["uav_speed_mps"]))
        uav.total_travel_m += step_dist
        if dist > 1e-9:
            uav.x += (target.area.center_x - uav.x) * step_dist / dist
            uav.y += (target.area.center_y - uav.y) * step_dist / dist

    def _update_edge_state(
        self,
        decisions: list[AllocationDecision],
        outcomes: list[AllocationOutcome],
        batch_size_by_level: dict[int, int],
    ) -> None:
        self.edge.load = min(1.0, 0.55 * self.edge.load + 0.45 * sum(max(0.0, d.cpu_share) for d in decisions))
        self.edge.gpu_load = min(1.0, 0.55 * self.edge.gpu_load + 0.45 * sum(max(0.0, d.gpu_share) for d in decisions))
        self.edge.queued_batches = sum(1 for _level, size in batch_size_by_level.items() if size > 0)
        current_levels = list(self.edge.cached_model_levels)
        successful = sorted(
            {o.service_level for o in outcomes if o.service_level > 0 and o.success},
            key=lambda level: (-max((o.priority for o in outcomes if o.service_level == level), default=0.0), level),
        )
        requested: list[int] = []
        for level in [*successful, *current_levels]:
            if level not in requested:
                requested.append(level)
        base_memory = float(self.cfg["gpu_memory_capacity_mb"]) * float(self.cfg["gpu_memory_load"])
        selected: list[int] = []
        used = base_memory
        for level in requested:
            if len(selected) >= int(self.edge.cache_capacity):
                continue
            memory = float(self.cfg["model_memory_mb_by_level"][str(level)])
            if used + memory <= self.edge.gpu_memory_capacity_mb + 1e-9:
                selected.append(level)
                used += memory
        self.edge.cached_model_levels = tuple(sorted(selected))
        self.edge.cached_models = len(selected)
        self.edge.model_cache_hit = bool(selected)
        self.edge.gpu_memory_used_mb = min(self.edge.gpu_memory_capacity_mb, used)

    def _update_semantic_cache(self, outcomes: list[AllocationOutcome]) -> None:
        aged = [replace(entry, cache_age=entry.cache_age + 1) for entry in self.semantic_cache_entries]
        task_by_id = {task.task_id: task for task in self.tasks}
        updated: list[SemanticCacheEntry] = []
        for outcome in outcomes:
            task = task_by_id.get(outcome.task_id)
            if task is None or not outcome.success or not self._requires_operational_intent(
                AllocationDecision(
                    task_id=outcome.task_id,
                    assigned_uav=outcome.assigned_uav,
                    sensing_decision=outcome.sensing_decision,
                    service_level=outcome.service_level,
                    bandwidth_share=outcome.bandwidth_share,
                    power_w=outcome.power_w,
                    cpu_share=outcome.cpu_share,
                    gpu_share=outcome.gpu_share,
                )
            ):
                continue
            updated.append(
                SemanticCacheEntry(
                    task_id=task.task_id,
                    question_type=task.question_type,
                    risk_level=task.risk_level,
                    priority=task.priority,
                    center_x=task.area.center_x,
                    center_y=task.area.center_y,
                    cache_age=0,
                    updated_slot=self.slot,
                )
            )
        dedup: dict[tuple[str, str], SemanticCacheEntry] = {}
        for entry in [*aged, *updated]:
            key = (entry.task_id, entry.question_type)
            prev = dedup.get(key)
            if prev is None or entry.updated_slot >= prev.updated_slot:
                dedup[key] = entry
        entries = list(dedup.values())
        policy = str(self.cfg["semantic_cache"].get("policy", "priority_freshness"))
        if policy == "priority_freshness":
            entries.sort(key=lambda entry: (-entry.priority, entry.cache_age, -entry.updated_slot))
        else:
            entries.sort(key=lambda entry: (entry.cache_age, -entry.updated_slot))
        self.semantic_cache_entries = entries[: int(self.cfg["cache_capacity"])]

    def _service_components(
        self,
        task: VQATaskState,
        uav: UAVState,
        decision: AllocationDecision,
        interference_dbm: float | None = None,
        edge_batch_size: int = 1,
        payload_bytes: float | None = None,
    ) -> dict[str, float | str]:
        level = int(decision.service_level)
        distance = task.area.distance_to(uav.x, uav.y)
        travel_needed = level > 0 and decision.sensing_decision in {"observe", "revisit"}
        travel_fraction = 1.0 if decision.sensing_decision == "revisit" else float(self.cfg["observe_travel_fraction"])
        travel_delay = distance * travel_fraction / max(1e-9, float(self.cfg["uav_speed_mps"])) if travel_needed else 0.0
        sensing_delay = float(self.cfg["sensing_time_by_level"][str(level)])
        if payload_bytes is None:
            payload_bytes = float(self.cfg["payload_mb_by_level"][str(level)]) * 1_000_000.0
        payload = float(payload_bytes) / 1_000_000.0
        bandwidth = max(1e-6, float(decision.bandwidth_share) * float(self.cfg["total_bandwidth_mhz"]))
        link = self._a2g_link(task, uav, decision, bandwidth, interference_dbm=interference_dbm)
        rate_mbps = max(1e-6, float(link["rate_mbps"]))
        tx_delay = 0.0 if level == 0 else (payload * 8.0) / rate_mbps
        cpu_workload = float(self.cfg["workload_by_level"][str(level)])
        gpu_workload = float(self.cfg["gpu_workload_by_level"][str(level)])
        gpu_memory_required = float(self.cfg["model_memory_mb_by_level"][str(level)])
        additional_memory = 0.0 if level in self.edge.cached_model_levels else gpu_memory_required
        gpu_memory_ok = self.edge.gpu_memory_used_mb + additional_memory <= self.edge.gpu_memory_capacity_mb + 1e-9
        cpu = max(1e-6, float(decision.cpu_share) * float(self.cfg["edge_cpu_capacity"]))
        gpu = max(1e-6, float(decision.gpu_share) * float(self.cfg["edge_gpu_capacity"]))
        cpu_delay = 0.03 if level == 0 else cpu_workload / cpu
        gpu_delay = 0.02 if level == 0 else gpu_workload / gpu
        batch_gain = 1.0 + float(self.cfg["batching_delay_scale"]) * max(0, int(edge_batch_size) - 1)
        compute_delay = max(cpu_delay, gpu_delay) / batch_gain
        queue_delay = float(self.edge.load) * float(self.cfg["queue_delay_scale"])
        queue_delay += float(self.edge.gpu_load) * float(self.cfg["gpu_queue_delay_scale"])
        model_hit = level > 0 and (level in self.edge.cached_model_levels or (self.edge.model_cache_hit and not self.edge.cached_model_levels))
        if model_hit:
            load_delay = float(self.cfg["model_load_delay"]) * float(self.cfg["model_cache_hit_delay_scale"])
        else:
            load_delay = float(self.cfg["model_load_delay"])
        delay = travel_delay + sensing_delay + tx_delay + compute_delay + queue_delay + load_delay
        travel_energy = distance * travel_fraction * float(self.cfg["propulsion_energy_per_m"]) if travel_needed else 0.0
        tx_energy = max(0.0, float(decision.power_w)) * tx_delay
        sensing_energy = float(self.cfg["sensing_energy_by_level"][str(level)])
        compute_energy = cpu_workload * float(self.cfg["compute_energy_per_workload"])
        compute_energy += gpu_workload * float(self.cfg["gpu_energy_per_workload"])
        energy = travel_energy + tx_energy + sensing_energy + compute_energy
        battery_cost = energy * float(self.cfg["battery_cost_scale"]) + float(self.cfg["hover_energy_per_slot"]) * (1.0 if level > 0 else 0.25)
        return {
            "delay": delay,
            "energy": energy,
            "battery_cost": battery_cost,
            "payload_mb": payload,
            "payload_bytes": float(payload_bytes),
            "channel_gain_db": float(link["channel_gain_db"]),
            "snr_db": float(link["snr_db"]),
            "rate_mbps": rate_mbps,
            "channel_bin": str(link["channel_bin"]),
            "distance_3d_m": float(link["distance_3d_m"]),
            "elevation_deg": float(link["elevation_deg"]),
            "los_probability": float(link["los_probability"]),
            "path_loss_db": float(link["path_loss_db"]),
            "interference_dbm": float(link["interference_dbm"]),
            "sinr_db": float(link["sinr_db"]),
            "fading_db": float(link["fading_db"]),
            "travel_delay": travel_delay,
            "sensing_delay": sensing_delay,
            "upload_delay": tx_delay,
            "queue_delay": queue_delay,
            "model_load_delay": load_delay,
            "inference_delay": compute_delay,
            "travel_energy": travel_energy,
            "tx_energy": tx_energy,
            "sensing_energy": sensing_energy,
            "compute_energy": compute_energy,
            "gpu_memory_required_mb": gpu_memory_required,
            "gpu_memory_ok": gpu_memory_ok,
        }

    def _a2g_link(
        self,
        task: VQATaskState,
        uav: UAVState,
        decision: AllocationDecision,
        bandwidth_mhz: float,
        interference_dbm: float | None = None,
    ) -> dict[str, float | str]:
        if not bool(self.cfg.get("a2g", {}).get("enabled", True)):
            prior_bin = task.channel_prior_bin or task.channel_bin
            rate_factor = float(self.cfg["channel_rate_factor"][prior_bin])
            return {
                "channel_gain_db": task.channel_gain_db,
                "snr_db": task.snr_db,
                "sinr_db": task.snr_db,
                "rate_mbps": bandwidth_mhz * rate_factor,
                "channel_bin": prior_bin,
                "distance_3d_m": 0.0,
                "elevation_deg": 0.0,
                "los_probability": 1.0,
                "path_loss_db": -task.channel_gain_db,
                "interference_dbm": float(self.cfg["a2g"]["interference_floor_dbm"]),
                "fading_db": 0.0,
            }
        a2g = self.cfg["a2g"]
        horizontal = task.area.distance_to(uav.x, uav.y)
        target_altitude = max(0.0, task.area.altitude_min)
        altitude_gap = max(1.0, abs(uav.altitude - target_altitude))
        distance_3d = max(float(a2g["reference_distance_m"]), math.hypot(horizontal, altitude_gap))
        elevation_deg = math.degrees(math.atan2(altitude_gap, max(1e-9, horizontal)))
        los_probability = self._los_probability(elevation_deg)
        expected_excess_loss = (
            los_probability * float(a2g["los_excess_loss_db"])
            + (1.0 - los_probability) * float(a2g["nlos_excess_loss_db"])
        )
        fading_db = self._fading_db(task)
        path_loss_db = (
            -float(a2g["reference_gain_db"])
            + 10.0 * float(a2g["path_loss_exponent"]) * math.log10(distance_3d / float(a2g["reference_distance_m"]))
            + expected_excess_loss
            + float(a2g["excess_loss_db"])
            + max(0.0, float(task.channel_impairment_db))
            - fading_db
        )
        gain_db = -path_loss_db
        tx_dbm = 30.0 + 10.0 * math.log10(max(1e-9, float(decision.power_w)))
        noise_dbm = -174.0 + 10.0 * math.log10(max(1.0, bandwidth_mhz * 1_000_000.0)) + float(a2g["noise_figure_db"])
        snr_db = tx_dbm + gain_db - noise_dbm
        interference_dbm = float(interference_dbm if interference_dbm is not None else a2g["interference_floor_dbm"])
        noise_mw = 10.0 ** (noise_dbm / 10.0)
        interference_mw = 0.0
        if bool(a2g.get("interference_enabled", True)):
            interference_mw = 10.0 ** (interference_dbm / 10.0)
        signal_mw = 10.0 ** ((tx_dbm + gain_db) / 10.0)
        sinr_linear = signal_mw / max(1e-18, noise_mw + interference_mw)
        sinr_db = 10.0 * math.log10(max(1e-18, sinr_linear))
        rate_mbps = bandwidth_mhz * math.log2(1.0 + max(0.0, sinr_linear))
        channel_bin = self._combine_channel_bin(self._snr_to_channel_bin(sinr_db), task.channel_prior_bin or task.channel_bin)
        return {
            "channel_gain_db": gain_db,
            "snr_db": snr_db,
            "sinr_db": sinr_db,
            "rate_mbps": rate_mbps,
            "channel_bin": channel_bin,
            "distance_3d_m": distance_3d,
            "elevation_deg": elevation_deg,
            "los_probability": los_probability,
            "path_loss_db": path_loss_db,
            "interference_dbm": interference_dbm,
            "fading_db": fading_db,
        }

    def _los_probability(self, elevation_deg: float) -> float:
        a2g = self.cfg["a2g"]
        a = float(a2g["los_a"])
        b = float(a2g["los_b"])
        return 1.0 / (1.0 + a * math.exp(-b * (elevation_deg - a)))

    def _fading_db(self, task: VQATaskState) -> float:
        a2g = self.cfg["a2g"]
        mode = str(a2g.get("fading_mode", "static"))
        if mode == "static":
            return float(task.fading_db)
        base = self._deterministic_normal(f"{self.episode}:{task.task_id}:base")
        if mode == "fast_fading":
            slot_component = self._deterministic_normal(f"{self.episode}:{task.task_id}:{self.slot}:fast")
            return float(a2g["fast_fading_std_db"]) * slot_component
        slot_component = self._deterministic_normal(f"{self.episode}:{task.task_id}:{self.slot}:slow")
        rho = float(a2g.get("fading_correlation", 0.85))
        return float(a2g["slow_fading_std_db"]) * (rho * base + (1.0 - rho) * slot_component)

    @staticmethod
    def _deterministic_normal(key: str) -> float:
        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(key))
        rng = random.Random(seed)
        u1 = max(1e-12, rng.random())
        u2 = rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def _snr_to_channel_bin(self, snr_db: float) -> str:
        a2g = self.cfg["a2g"]
        if snr_db < float(a2g["bad_snr_db"]):
            return "bad"
        if snr_db < float(a2g["good_snr_db"]):
            return "medium"
        return "good"

    @staticmethod
    def _combine_channel_bin(dynamic_bin: str, scenario_bin: str) -> str:
        order = {"bad": 0, "medium": 1, "good": 2}
        reverse = {0: "bad", 1: "medium", 2: "good"}
        return reverse[min(order.get(dynamic_bin, 1), order.get(scenario_bin, 1))]

    def _reward(
        self,
        task: VQATaskState,
        delay: float,
        energy: float,
        payload_mb: float,
        quality_ok: bool,
        deadline_ok: bool,
        success: bool,
        conflict: bool,
    ) -> float:
        w = self.cfg["reward_weights"]
        reward = float(w["success"]) * task.priority * float(success)
        reward -= float(w["delay"]) * delay
        reward -= float(w["energy"]) * energy
        reward -= float(w.get("payload", 0.0)) * payload_mb
        reward -= float(w["quality_violation"]) * float(not quality_ok)
        reward -= float(w["deadline_violation"]) * float(not deadline_ok)
        reward -= float(w["airspace_conflict"]) * float(conflict)
        return reward

    def _resource_ok(self, decisions: list[AllocationDecision]) -> bool:
        return (
            sum(max(0.0, d.bandwidth_share) for d in decisions) <= 1.0 + 1e-9
            and sum(max(0.0, d.cpu_share) for d in decisions) <= 1.0 + 1e-9
            and sum(max(0.0, d.gpu_share) for d in decisions) <= 1.0 + 1e-9
        )

    def _interference_by_task(self, tasks: list[VQATaskState], decisions: list[AllocationDecision]) -> dict[str, float]:
        a2g = self.cfg["a2g"]
        floor_dbm = float(a2g["interference_floor_dbm"])
        if not bool(a2g.get("interference_enabled", True)):
            return {task.task_id: floor_dbm for task in tasks}
        task_by_id = {task.task_id: task for task in tasks}
        rx_power: dict[str, float] = {}
        for decision in decisions:
            task = task_by_id.get(decision.task_id)
            if task is None or int(decision.service_level) == 0:
                continue
            uav = self.uavs[max(0, min(decision.assigned_uav, len(self.uavs) - 1))]
            bandwidth = max(1e-6, float(decision.bandwidth_share) * float(self.cfg["total_bandwidth_mhz"]))
            link = self._a2g_link(task, uav, decision, bandwidth, interference_dbm=floor_dbm)
            tx_dbm = 30.0 + 10.0 * math.log10(max(1e-9, float(decision.power_w)))
            rx_power[decision.task_id] = tx_dbm + float(link["channel_gain_db"])
        out: dict[str, float] = {}
        for decision in decisions:
            task = task_by_id.get(decision.task_id)
            if task is None:
                continue
            if int(decision.service_level) == 0:
                out[task.task_id] = floor_dbm
                continue
            total_mw = 10.0 ** (floor_dbm / 10.0)
            for other in decisions:
                if other.task_id == decision.task_id or int(other.service_level) == 0:
                    continue
                overlap = min(max(0.0, decision.bandwidth_share), max(0.0, other.bandwidth_share))
                overlap *= float(a2g.get("interference_overlap_scale", 0.6))
                if overlap <= 0:
                    continue
                total_mw += overlap * 10.0 ** (rx_power.get(other.task_id, floor_dbm) / 10.0)
            out[task.task_id] = 10.0 * math.log10(max(1e-18, total_mw))
        return out

    def _conflict_task_ids(self, decisions: list[AllocationDecision]) -> set[str]:
        task_by_id = {t.task_id: t for t in self.tasks}
        active_ids = {t.task_id for t in self.active_tasks()}
        operational = [decision for decision in decisions if decision.task_id in active_ids and self._requires_operational_intent(decision)]
        conflicts: set[str] = set()
        for i, left in enumerate(operational):
            for right in operational[i + 1 :]:
                lt = task_by_id.get(left.task_id)
                rt = task_by_id.get(right.task_id)
                if lt and rt and lt.area.overlaps(rt.area):
                    conflicts.add(lt.task_id)
                    conflicts.add(rt.task_id)
        return conflicts

    def _semantic_cache_hit_probability(self, task: VQATaskState) -> float:
        cache_cfg = self.cfg["semantic_cache"]
        base = float(cache_cfg[f"{task.freshness_bin}_hit_probability"])
        radius = float(cache_cfg["spatial_reuse_radius_m"])
        reusable = any(
            other.task_id != task.task_id
            and other.completed
            and other.question_type == task.question_type
            and other.area.distance_to(task.area.center_x, task.area.center_y) <= radius
            for other in self.tasks
        ) or any(
            entry.task_id != task.task_id
            and entry.question_type == task.question_type
            and math.hypot(entry.center_x - task.area.center_x, entry.center_y - task.area.center_y) <= radius
            for entry in self.semantic_cache_entries
        )
        boost = float(cache_cfg["reuse_boost"]) if reusable else 0.0
        if str(cache_cfg.get("policy", "")) == "priority_freshness" and task.priority >= 2.0:
            boost *= 0.5
        return max(0.0, min(1.0, base + boost))

    def _battery_ok(self, uav: UAVState, battery_cost: float) -> bool:
        remaining = uav.battery - battery_cost
        reserve = float(self.cfg["return_energy_reserve"])
        return remaining >= max(float(self.cfg["battery_failure_threshold"]), reserve)

    @staticmethod
    def _conflict_resolution(task: VQATaskState, decision: AllocationDecision) -> str:
        if decision.sensing_decision == "revisit":
            return "time_separation_revisit"
        if task.area.altitude_max - task.area.altitude_min >= 60.0:
            return "altitude_separation_possible"
        return "unresolved_overlap"

    def action_mask(self) -> dict[str, Any]:
        remaining_bw = 1.0
        remaining_cpu = 1.0
        remaining_gpu = 1.0
        memory_free = max(0.0, self.edge.gpu_memory_capacity_mb - self.edge.gpu_memory_used_mb)
        max_tasks = int(self.cfg["observation"]["max_tasks"])
        active_ids = {task.task_id for task in self.active_tasks()}
        tasks = self.tasks[:max_tasks]
        active_mask = [1.0 if task.task_id in active_ids else 0.0 for task in tasks]
        active_mask.extend([0.0] * max(0, max_tasks - len(active_mask)))
        service_allowed = {
            level: bool(level == 0 or level in self.edge.cached_model_levels or float(self.cfg["model_memory_mb_by_level"][str(level)]) <= memory_free + 1e-9)
            for level in self.service_levels()
        }
        per_task_service = {
            task.task_id: {level: bool(level == 0 or service_allowed[level]) for level in self.service_levels()}
            for task in tasks
        }
        return {
            "bandwidth_budget": remaining_bw,
            "cpu_budget": remaining_cpu,
            "gpu_budget": remaining_gpu,
            "gpu_memory_free_mb": memory_free,
            "battery_ok_by_uav": {u.uav_id: u.battery >= float(self.cfg["return_energy_reserve"]) for u in self.uavs},
            "service_level_allowed": service_allowed,
            "active_task_ids": sorted(active_ids),
            "active_task_mask": active_mask,
            "per_task_service_level_allowed": per_task_service,
            "resource_budget_hint": {
                "active_tasks": float(len(active_ids)),
                "fair_bandwidth_share": remaining_bw / max(1, len(active_ids)),
                "fair_cpu_share": remaining_cpu / max(1, len(active_ids)),
                "fair_gpu_share": remaining_gpu / max(1, len(active_ids)),
            },
        }

    def hierarchical_action_schema(self) -> dict[str, Any]:
        return {
            "high_level": ["assigned_uav", "sensing_decision"],
            "low_level_discrete": ["service_level"],
            "low_level_continuous": ["bandwidth_share", "power_w", "cpu_share", "gpu_share"],
            "constraint_mask": "available_actions.action_mask",
        }

    def _advance_tasks(self, outcomes: list[AllocationOutcome]) -> list[VQATaskState]:
        outcome_by_id = {o.task_id: o for o in outcomes}
        advanced: list[VQATaskState] = []
        for task in self.tasks:
            outcome = outcome_by_id.get(task.task_id)
            completed = task.completed or bool(outcome and outcome.success)
            if outcome and outcome.sensing_decision in {"observe", "revisit"} and outcome.service_level > 0:
                cache_age = 0
            else:
                cache_age = task.cache_age + 1
            advanced.append(
                replace(
                    task,
                    cache_age=cache_age,
                    freshness_bin=freshness_from_age(cache_age, self.cfg),
                    completed=completed,
                    snr_db=float(outcome.snr_db) if outcome else task.snr_db,
                    rate_mbps=float(outcome.rate_mbps) if outcome else task.rate_mbps,
                    channel_gain_db=float(outcome.channel_gain_db) if outcome else task.channel_gain_db,
                    effective_channel_bin=str(outcome.effective_channel_bin) if outcome else task.effective_channel_bin,
                    fading_db=float(outcome.fading_db) if outcome else task.fading_db,
                )
            )
        return advanced

    def action_spec(self) -> dict[str, Any]:
        return {
            "type": "hybrid",
            "max_tasks": int(self.cfg["observation"]["max_tasks"]),
            "max_uavs": int(self.cfg["observation"]["max_uavs"]),
            "per_task_discrete": {
                "assigned_uav": int(self.cfg["num_uavs"]),
                "sensing_decision": ["reuse_cache", "observe", "revisit"],
                "service_level": self.service_levels(),
            },
            "per_task_continuous": {
                "bandwidth_share": [0.0, 1.0],
                "power_w": [min(self.cfg["power_levels_w"]), max(self.cfg["power_levels_w"])],
                "cpu_share": [0.0, 1.0],
                "gpu_share": [0.0, 1.0],
            },
        }

    def observation_vector(self) -> list[float]:
        max_tasks = int(self.cfg["observation"]["max_tasks"])
        max_uavs = int(self.cfg["observation"]["max_uavs"])
        qtypes = ["presence", "counting", "risk", "attribute", "relation"]
        freshness_order = {"fresh": 0.0, "stale": 0.5, "expired": 1.0}
        view_order = {"poor": 0.0, "medium": 0.5, "good": 1.0}
        channel_order = {"bad": 0.0, "medium": 0.5, "good": 1.0}
        vec: list[float] = [self.slot / max(1.0, float(self.cfg["episode_slots"]))]
        for task in self.tasks[:max_tasks]:
            q_one_hot = [1.0 if task.question_type == q else 0.0 for q in qtypes]
            nearest = self.nearest_uav(task)
            distance_norm = task.area.distance_to(nearest.x, nearest.y) / max(1.0, float(self.cfg["cell_radius_m"]))
            link = self._a2g_link(task, nearest, self.default_decision(task), float(self.cfg["total_bandwidth_mhz"]) * float(self.cfg["bandwidth_shares"][0]))
            vec.extend(
                [
                    *q_one_hot,
                    min(1.0, task.tau_k / 10.0),
                    task.epsilon_k,
                    min(1.0, task.priority / 3.0),
                    1.0 if task.risk_level == "critical" else 0.0,
                    freshness_order.get(task.freshness_bin, 1.0),
                    view_order.get(task.view_quality_bin, 0.0),
                    channel_order.get(task.effective_channel_bin or task.channel_bin, 0.5),
                    min(1.0, task.cache_age / 10.0),
                    min(1.0, distance_norm),
                    1.0 if task.completed else 0.0,
                    min(1.0, max(0.0, (float(link["sinr_db"]) + 20.0) / 60.0)),
                    float(link["los_probability"]),
                    min(1.0, max(0.0, float(link["path_loss_db"]) / 160.0)),
                    min(1.0, max(0.0, (float(link["interference_dbm"]) + 140.0) / 80.0)),
                    self._semantic_cache_hit_probability(task),
                ]
            )
        per_task_width = 20
        vec.extend([0.0] * max(0, (max_tasks - min(len(self.tasks), max_tasks)) * per_task_width))
        for uav in self.uavs[:max_uavs]:
            vec.extend(
                [
                    uav.x / max(1.0, float(self.cfg["cell_radius_m"])),
                    uav.y / max(1.0, float(self.cfg["cell_radius_m"])),
                    uav.altitude / 200.0,
                    uav.battery / max(1e-9, float(self.cfg["initial_battery"])),
                    min(1.0, uav.assigned_load / max(1, len(self.tasks))),
                ]
            )
        vec.extend([0.0] * max(0, (max_uavs - min(len(self.uavs), max_uavs)) * 5))
        vec.extend(
            [
                self.edge.load,
                self.edge.gpu_load,
                1.0 if self.edge.model_cache_hit else 0.0,
                self.edge.cpu_capacity / max(1.0, float(self.cfg["edge_cpu_capacity"])),
                self.edge.gpu_capacity / max(1.0, float(self.cfg["edge_gpu_capacity"])),
            ]
        )
        return [round(float(x), 6) for x in vec]

    def action_from_vector(self, values: list[float]) -> list[AllocationDecision]:
        if not values:
            return [self.default_decision(task) for task in self.tasks]
        decisions: list[AllocationDecision] = []
        width = 7
        sensing = ["reuse_cache", "observe", "revisit"]
        for idx, task in enumerate(self.tasks):
            chunk = values[idx * width : (idx + 1) * width]
            if len(chunk) < width:
                decisions.append(self.default_decision(task))
                continue
            uav_id = int(abs(round(chunk[0]))) % max(1, len(self.uavs))
            sensing_decision = sensing[int(abs(round(chunk[1]))) % len(sensing)]
            levels = self.service_levels()
            service_level = levels[int(abs(round(chunk[2]))) % len(levels)]
            power_min = min(float(x) for x in self.cfg["power_levels_w"])
            power_max = max(float(x) for x in self.cfg["power_levels_w"])
            decisions.append(
                AllocationDecision(
                    task_id=task.task_id,
                    assigned_uav=uav_id,
                    sensing_decision=sensing_decision,
                    service_level=service_level,
                    bandwidth_share=min(1.0, max(0.0, float(chunk[3]))),
                    power_w=power_min + min(1.0, max(0.0, float(chunk[4]))) * (power_max - power_min),
                    cpu_share=min(1.0, max(0.0, float(chunk[5]))),
                    gpu_share=min(1.0, max(0.0, float(chunk[6]))),
                )
            )
        return decisions

    def _init_uavs(self) -> list[UAVState]:
        radius = float(self.cfg["cell_radius_m"]) * 0.35
        num_uavs = int(self.cfg["num_uavs"])
        uavs = []
        for uid in range(num_uavs):
            angle = 2.0 * math.pi * uid / max(1, num_uavs)
            uavs.append(
                UAVState(
                    uav_id=uid,
                    x=radius * math.cos(angle),
                    y=radius * math.sin(angle),
                    altitude=float(self.cfg["uav_altitude_m"]),
                    battery=float(self.cfg["initial_battery"]),
                )
            )
        return uavs

    def _sample_edge(self) -> EdgeState:
        has_cached_model = self.rng.random() < 0.72
        base_memory = float(self.cfg["gpu_memory_capacity_mb"]) * float(self.cfg["gpu_memory_load"])
        cached_levels = (1,) if has_cached_model else ()
        used_memory = base_memory + sum(float(self.cfg["model_memory_mb_by_level"][str(level)]) for level in cached_levels)
        return EdgeState(
            cpu_capacity=float(self.cfg["edge_cpu_capacity"]),
            gpu_capacity=float(self.cfg["edge_gpu_capacity"]),
            load=round(self.rng.uniform(0.05, 0.85), 6),
            model_cache_hit=has_cached_model,
            gpu_load=round(self.rng.uniform(0.03, 0.55), 6),
            cached_models=1 if has_cached_model else 0,
            cache_capacity=int(self.cfg["model_cache_capacity"]),
            gpu_memory_capacity_mb=float(self.cfg["gpu_memory_capacity_mb"]),
            gpu_memory_used_mb=round(min(float(self.cfg["gpu_memory_capacity_mb"]), used_memory), 6),
            queued_batches=0,
            cached_model_levels=cached_levels,
        )

    def _sample_tasks(self, scenario: str) -> list[VQATaskState]:
        if scenario != "sampled":
            return self._scenario_tasks(scenario)
        n = int(self.cfg["tasks_per_episode"])
        selected = [self.raw_tasks[self.rng.randrange(len(self.raw_tasks))] for _ in range(n)]
        return [self._task_from_row(row, idx) for idx, row in enumerate(selected)]

    def _scenario_tasks(self, scenario: str) -> list[VQATaskState]:
        builders = {
            "literature_demo": self._literature_demo_specs,
            "clear_area_nominal": self._clear_area_specs,
            "cache_freshness": self._cache_freshness_specs,
            "critical_preemption": self._critical_preemption_specs,
            "area4d_conflict": self._area4d_conflict_specs,
            "bad_channel_stress": self._bad_channel_stress_specs,
            "los_vs_nlos": self._los_vs_nlos_specs,
            "interference_stress": self._interference_stress_specs,
            "mobility_fading": self._mobility_fading_specs,
        }
        if scenario not in builders:
            raise ValueError(f"unknown resource scenario: {scenario}")
        specs = builders[scenario]()
        return [self._task_from_spec(spec, idx) for idx, spec in enumerate(specs)]

    def _task_from_spec(self, spec: dict[str, Any], idx: int) -> VQATaskState:
        row = self._pick_row(str(spec.get("question_type", "presence")), str(spec.get("risk_level", "")))
        cache_age = int(spec.get("cache_age", 0))
        freshness = str(spec.get("freshness_bin", freshness_from_age(cache_age, self.cfg)))
        area_spec = spec.get("area", {})
        area = Area4D(
            center_x=float(area_spec.get("center_x", 100.0 * idx)),
            center_y=float(area_spec.get("center_y", 80.0 * idx)),
            radius=float(area_spec.get("radius", 45.0)),
            altitude_min=float(area_spec.get("altitude_min", 30.0)),
            altitude_max=float(area_spec.get("altitude_max", 120.0)),
            start_slot=int(area_spec.get("start_slot", idx)),
            end_slot=int(area_spec.get("end_slot", idx + 3)),
        )
        return VQATaskState(
            task_id=f"ep{self.episode}_{spec.get('name', 'scenario')}_{idx}",
            image_id=row["image_id"],
            question_type=str(spec.get("question_type", row["question_type"])),
            question=str(spec.get("question", row.get("question", ""))),
            risk_level=str(spec.get("risk_level", row["risk_level"])),
            epsilon_k=float(spec.get("epsilon_k", row["epsilon_k"])),
            tau_k=float(spec.get("tau_k", row["tau_k"])),
            priority=float(spec.get("priority", 2.0 if spec.get("risk_level", row["risk_level"]) == "critical" else 1.0)),
            generation_slot=int(spec.get("generation_slot", 0)),
            cache_age=cache_age,
            freshness_bin=freshness,
            view_quality_bin=str(spec.get("view_quality_bin", row["view_quality_bin"])),
            channel_bin=str(spec.get("channel_bin", "good")),
            area=area,
            channel_prior_bin=str(spec.get("channel_bin", "good")),
            effective_channel_bin=str(spec.get("channel_bin", "good")),
            arrival_slot=int(spec.get("arrival_slot", spec.get("generation_slot", 0))),
            channel_impairment_db=float(spec.get("channel_impairment_db", 0.0)),
        )

    def _pick_row(self, question_type: str, risk_level: str = "") -> dict[str, str]:
        for row in self.raw_tasks:
            if row.get("question_type") == question_type and (not risk_level or row.get("risk_level") == risk_level):
                return row
        for row in self.raw_tasks:
            if row.get("question_type") == question_type:
                return row
        return self.raw_tasks[0]

    @staticmethod
    def _area(
        x: float,
        y: float,
        start: int,
        end: int,
        radius: float = 45.0,
        altitude_min: float = 0.0,
        altitude_max: float = 120.0,
    ) -> dict[str, float | int]:
        return {
            "center_x": x,
            "center_y": y,
            "radius": radius,
            "altitude_min": altitude_min,
            "altitude_max": altitude_max,
            "start_slot": start,
            "end_slot": end,
        }

    def _literature_demo_specs(self) -> list[dict[str, Any]]:
        return [
            self._cache_task("cache_feasible", 0.58, "good", "good", self._area(-240, -120, 0, 3)),
            self._light_task("light_feasible", 0.55, "good", "medium", self._area(-70, -70, 1, 4)),
            self._image_task("image_required", 0.74, "good", "good", self._area(120, 90, 1, 5)),
            self._infeasible_task("infeasible_bad_channel", self._area(260, 160, 2, 5)),
            self._critical_task("critical_preempt", self._area(-90, 220, 0, 3), tau=2.8),
            self._light_task("normal_competing", 0.52, "medium", "medium", self._area(300, -240, 2, 6)),
            self._attribute_task("attribute_damage", 0.62, "medium", "good", self._area(-330, 260, 1, 5)),
            self._relation_task("relation_person_vehicle", 0.60, "medium", "medium", self._area(330, 250, 1, 5)),
        ]

    def _clear_area_specs(self) -> list[dict[str, Any]]:
        return [
            self._cache_task("clear_cache", 0.58, "good", "good", self._area(-260, -180, 0, 2)),
            self._light_task("clear_light", 0.55, "good", "medium", self._area(0, 0, 1, 3)),
            self._image_task("clear_image", 0.74, "good", "good", self._area(260, 180, 2, 5)),
        ]

    def _cache_freshness_specs(self) -> list[dict[str, Any]]:
        return [
            self._cache_task("fresh_cache", 0.58, "good", "good", self._area(-160, 0, 0, 3), cache_age=0),
            self._light_task("stale_cache", 0.55, "good", "medium", self._area(0, 0, 0, 3), cache_age=2),
            self._image_task("expired_cache", 0.74, "good", "good", self._area(160, 0, 0, 3), cache_age=5),
        ]

    def _critical_preemption_specs(self) -> list[dict[str, Any]]:
        critical = self._critical_task("critical_urgent", self._area(0, 0, 0, 2), tau=4.8, cache_age=3)
        critical["epsilon_k"] = 0.68
        return [
            critical,
            self._light_task("normal_1", 0.52, "medium", "medium", self._area(260, 160, 0, 4)),
            self._light_task("normal_2", 0.52, "medium", "medium", self._area(-280, -160, 0, 4)),
            self._cache_task("normal_cache", 0.58, "good", "good", self._area(220, 120, 1, 5)),
        ]

    def _area4d_conflict_specs(self) -> list[dict[str, Any]]:
        return [
            self._light_task("conflict_left", 0.55, "good", "medium", self._area(0, 0, 0, 4, radius=90)),
            self._light_task("conflict_right", 0.55, "good", "medium", self._area(75, 0, 1, 5, radius=90)),
            self._cache_task("clear_cache", 0.58, "good", "good", self._area(400, 280, 0, 3)),
        ]

    def _bad_channel_stress_specs(self) -> list[dict[str, Any]]:
        specs = [
            self._light_task("bad_presence", 0.62, "bad", "good", self._area(-120, 40, 0, 3)),
            self._image_task("bad_risk", 0.70, "bad", "good", self._area(80, 40, 0, 3)),
            self._infeasible_task("bad_infeasible", self._area(250, 80, 0, 3)),
        ]
        for spec in specs:
            spec["channel_impairment_db"] = 36.0
        return specs

    def _los_vs_nlos_specs(self) -> list[dict[str, Any]]:
        return [
            self._light_task("high_elevation_los", 0.55, "good", "medium", self._area(340, 0, 0, 4, altitude_min=0.0)),
            self._light_task("low_elevation_nlos", 0.55, "good", "medium", self._area(-950, 0, 0, 4, altitude_min=0.0)),
            self._cache_task("reference_cache", 0.58, "good", "good", self._area(0, 300, 0, 3, altitude_min=0.0)),
        ]

    def _interference_stress_specs(self) -> list[dict[str, Any]]:
        return [
            self._image_task("interference_a", 0.70, "good", "good", self._area(-80, 0, 0, 5, altitude_min=0.0)),
            self._image_task("interference_b", 0.70, "good", "good", self._area(80, 0, 0, 5, altitude_min=0.0)),
            self._light_task("interference_c", 0.55, "good", "medium", self._area(0, 110, 0, 5, altitude_min=0.0)),
        ]

    def _mobility_fading_specs(self) -> list[dict[str, Any]]:
        return [
            self._light_task("fading_near", 0.55, "good", "medium", self._area(-160, 0, 0, 5, altitude_min=0.0)),
            self._image_task("fading_mid", 0.70, "good", "good", self._area(160, 120, 1, 5, altitude_min=0.0)),
            self._attribute_task("fading_attribute", 0.62, "medium", "good", self._area(420, -120, 2, 6, altitude_min=0.0)),
        ]

    def _cache_task(self, name: str, epsilon: float, channel: str, view: str, area: dict[str, Any], cache_age: int = 0) -> dict[str, Any]:
        return {
            "name": name,
            "question_type": "presence",
            "risk_level": "normal",
            "epsilon_k": epsilon,
            "tau_k": 3.0,
            "priority": 1.0,
            "cache_age": cache_age,
            "freshness_bin": freshness_from_age(cache_age, self.cfg),
            "channel_bin": channel,
            "view_quality_bin": view,
            "area": area,
        }

    def _light_task(self, name: str, epsilon: float, channel: str, view: str, area: dict[str, Any], cache_age: int = 2) -> dict[str, Any]:
        return {
            "name": name,
            "question_type": "counting",
            "risk_level": "normal",
            "epsilon_k": epsilon,
            "tau_k": 3.5,
            "priority": 1.0,
            "cache_age": cache_age,
            "freshness_bin": freshness_from_age(cache_age, self.cfg),
            "channel_bin": channel,
            "view_quality_bin": view,
            "area": area,
        }

    def _image_task(self, name: str, epsilon: float, channel: str, view: str, area: dict[str, Any], cache_age: int = 5) -> dict[str, Any]:
        return {
            "name": name,
            "question_type": "risk",
            "risk_level": "critical",
            "epsilon_k": epsilon,
            "tau_k": 3.2,
            "priority": 2.0,
            "cache_age": cache_age,
            "freshness_bin": freshness_from_age(cache_age, self.cfg),
            "channel_bin": channel,
            "view_quality_bin": view,
            "area": area,
        }

    def _attribute_task(self, name: str, epsilon: float, channel: str, view: str, area: dict[str, Any], cache_age: int = 3) -> dict[str, Any]:
        return {
            "name": name,
            "question_type": "attribute",
            "risk_level": "normal",
            "epsilon_k": epsilon,
            "tau_k": 4.0,
            "priority": 1.2,
            "cache_age": cache_age,
            "freshness_bin": freshness_from_age(cache_age, self.cfg),
            "channel_bin": channel,
            "view_quality_bin": view,
            "area": area,
        }

    def _relation_task(self, name: str, epsilon: float, channel: str, view: str, area: dict[str, Any], cache_age: int = 4) -> dict[str, Any]:
        return {
            "name": name,
            "question_type": "relation",
            "risk_level": "normal",
            "epsilon_k": epsilon,
            "tau_k": 4.2,
            "priority": 1.3,
            "cache_age": cache_age,
            "freshness_bin": freshness_from_age(cache_age, self.cfg),
            "channel_bin": channel,
            "view_quality_bin": view,
            "area": area,
        }

    def _critical_task(self, name: str, area: dict[str, Any], tau: float = 2.8, cache_age: int = 4) -> dict[str, Any]:
        task = self._image_task(name, 0.72, "good", "good", area, cache_age=cache_age)
        task["tau_k"] = tau
        task["priority"] = 3.0
        return task

    def _infeasible_task(self, name: str, area: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "question_type": "risk",
            "risk_level": "critical",
            "epsilon_k": 0.90,
            "tau_k": 0.8,
            "priority": 2.0,
            "cache_age": 5,
            "freshness_bin": "expired",
            "channel_bin": "bad",
            "view_quality_bin": "poor",
            "area": area,
        }

    def _task_from_row(self, row: dict[str, str], idx: int) -> VQATaskState:
        density = float(row.get("density_score", "1") or 1.0)
        angle = self.rng.random() * 2.0 * math.pi
        radius = float(self.cfg["cell_radius_m"]) * math.sqrt(self.rng.random())
        cache_age = self.rng.randrange(0, 6)
        freshness = freshness_from_age(cache_age, self.cfg)
        tau = float(row["tau_k"])
        priority = 2.0 if row["risk_level"] == "critical" else 1.0
        slot = idx % max(1, int(self.cfg["episode_slots"]))
        area = Area4D(
            center_x=radius * math.cos(angle),
            center_y=radius * math.sin(angle),
            radius=max(25.0, min(120.0, 20.0 + density * 5.0)),
            altitude_min=30.0,
            altitude_max=120.0,
            start_slot=slot,
            end_slot=slot + max(1, int(math.ceil(tau))),
        )
        channel_bin = self.rng.choice(["bad", "medium", "good"])
        return VQATaskState(
            task_id=f"ep{self.episode}_task{idx}",
            image_id=row["image_id"],
            question_type=row["question_type"],
            question=row.get("question", ""),
            risk_level=row["risk_level"],
            epsilon_k=float(row["epsilon_k"]),
            tau_k=tau,
            priority=priority,
            generation_slot=slot,
            cache_age=cache_age,
            freshness_bin=freshness,
            view_quality_bin=row["view_quality_bin"],
            channel_bin=channel_bin,
            area=area,
            channel_prior_bin=channel_bin,
            effective_channel_bin=channel_bin,
            arrival_slot=slot,
            channel_impairment_db=0.0,
        )

    def _with_initial_link(self, task: VQATaskState) -> VQATaskState:
        if not self.uavs:
            return task
        decision = self.default_decision(task)
        uav = self.nearest_uav(task)
        bandwidth = max(1e-6, float(decision.bandwidth_share) * float(self.cfg["total_bandwidth_mhz"]))
        link = self._a2g_link(task, uav, decision, bandwidth)
        return replace(
            task,
            channel_gain_db=round(float(link["channel_gain_db"]), 6),
            snr_db=round(float(link["snr_db"]), 6),
            rate_mbps=round(float(link["rate_mbps"]), 6),
            effective_channel_bin=str(link["channel_bin"]),
            fading_db=round(float(link["fading_db"]), 6),
        )

    @staticmethod
    def _task_to_dict(task: VQATaskState) -> dict[str, Any]:
        data = asdict(task)
        data["area"] = asdict(task.area)
        return data


def summarize_outcomes(outcomes: list[AllocationOutcome]) -> dict[str, float]:
    denom = max(1, len(outcomes))
    service_levels = sorted({0, 1, 2, 3, *(o.service_level for o in outcomes)})
    service_counts = {level: sum(1 for o in outcomes if o.service_level == level) for level in service_levels}
    feasibility_counts = {
        name: sum(1 for o in outcomes if o.feasibility_class == name)
        for name in ("feasible", "deadline_limited", "quality_limited")
    }
    task_summary = summarize_task_level(outcomes)
    return {
        "tasks": float(len(outcomes)),
        "task_success_rate": sum(float(o.success) for o in outcomes) / denom,
        "attempt_success_rate": sum(float(o.success) for o in outcomes) / denom,
        "unique_task_success_rate": task_summary["unique_task_success_rate"],
        "completion_rate": task_summary["completion_rate"],
        "average_attempts_per_task": task_summary["average_attempts_per_task"],
        "average_accuracy": sum(o.expected_accuracy for o in outcomes) / denom,
        "average_delay": sum(o.delay for o in outcomes) / denom,
        "average_energy": sum(o.energy for o in outcomes) / denom,
        "average_rate_mbps": sum(o.rate_mbps for o in outcomes) / denom,
        "average_snr_db": sum(o.snr_db for o in outcomes) / denom,
        "average_sinr_db": sum(o.sinr_db for o in outcomes) / denom,
        "average_distance_3d_m": sum(o.distance_3d_m for o in outcomes) / denom,
        "average_elevation_deg": sum(o.elevation_deg for o in outcomes) / denom,
        "average_los_probability": sum(o.los_probability for o in outcomes) / denom,
        "average_path_loss_db": sum(o.path_loss_db for o in outcomes) / denom,
        "average_interference_dbm": sum(o.interference_dbm for o in outcomes) / denom,
        "average_fading_db": sum(o.fading_db for o in outcomes) / denom,
        "average_cache_hit_probability": sum(o.cache_hit_probability for o in outcomes) / denom,
        "semantic_cache_hit_rate": sum(float(o.semantic_cache_hit) for o in outcomes) / denom,
        "average_accuracy_gain": sum(o.accuracy_gain for o in outcomes) / denom,
        "average_semantic_efficiency": sum(o.semantic_efficiency for o in outcomes) / denom,
        "average_payload_mb": sum(o.payload_mb for o in outcomes) / denom,
        "lut_missing_rate": sum(float(o.lut_missing) for o in outcomes) / denom,
        "average_utility_per_latency": sum(o.utility_per_latency for o in outcomes) / denom,
        "gpu_memory_ok_rate": sum(float(o.gpu_memory_ok) for o in outcomes) / denom,
        "battery_ok_rate": sum(float(o.battery_ok) for o in outcomes) / denom,
        "average_battery_remaining": sum(o.battery_remaining for o in outcomes) / denom,
        "average_uav_utilization": sum(o.uav_utilization for o in outcomes) / denom,
        "average_upload_delay": sum(o.upload_delay for o in outcomes) / denom,
        "average_inference_delay": sum(o.inference_delay for o in outcomes) / denom,
        "average_model_load_delay": sum(o.model_load_delay for o in outcomes) / denom,
        "average_travel_delay": sum(o.travel_delay for o in outcomes) / denom,
        "average_semantic_utility": sum(o.semantic_utility for o in outcomes) / denom,
        "quality_satisfaction_rate": sum(float(o.quality_ok) for o in outcomes) / denom,
        "deadline_satisfaction_rate": sum(float(o.deadline_ok) for o in outcomes) / denom,
        "quality_violation_rate": sum(float(not o.quality_ok) for o in outcomes) / denom,
        "deadline_violation_rate": sum(float(not o.deadline_ok) for o in outcomes) / denom,
        "resource_violation_rate": sum(float(not o.resource_ok) for o in outcomes) / denom,
        "airspace_conflict_rate": sum(float(o.airspace_conflict) for o in outcomes) / denom,
        "cache_hit_rate": service_counts[0] / denom,
        "service_level_0_rate": service_counts[0] / denom,
        "service_level_1_rate": service_counts[1] / denom,
        "service_level_2_rate": service_counts[2] / denom,
        "service_level_3_rate": service_counts[3] / denom,
        "average_reward": sum(o.reward for o in outcomes) / denom,
        "feasible_task_rate": feasibility_counts["feasible"] / denom,
        "deadline_limited_rate": feasibility_counts["deadline_limited"] / denom,
        "quality_limited_rate": feasibility_counts["quality_limited"] / denom,
    }


def summarize_task_level(outcomes: list[AllocationOutcome]) -> dict[str, float]:
    grouped: dict[tuple[str, int, str], list[AllocationOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault((outcome.policy, outcome.episode, outcome.task_id), []).append(outcome)
    denom = max(1, len(grouped))
    unique_success = sum(float(any(o.success for o in rows)) for rows in grouped.values()) / denom
    completion = sum(float(any(o.task_completed or o.success for o in rows)) for rows in grouped.values()) / denom
    attempts = sum(len(rows) for rows in grouped.values()) / denom
    return {
        "unique_tasks": float(len(grouped)),
        "unique_task_success_rate": unique_success,
        "completion_rate": completion,
        "average_attempts_per_task": attempts,
    }


def _with_attempt_indices(outcomes: list[AllocationOutcome]) -> list[AllocationOutcome]:
    counters: dict[tuple[str, int, str], int] = {}
    annotated: list[AllocationOutcome] = []
    completed: set[tuple[str, int, str]] = set()
    for outcome in outcomes:
        key = (outcome.policy, outcome.episode, outcome.task_id)
        counters[key] = counters.get(key, 0) + 1
        if outcome.success or outcome.task_completed:
            completed.add(key)
        annotated.append(
            replace(
                outcome,
                attempt_index=counters[key],
                task_completed=key in completed,
            )
        )
    return annotated


def write_resource_outputs(outcomes: list[AllocationOutcome], csv_path: Path, md_path: Path) -> None:
    ensure_parent(csv_path)
    ensure_parent(md_path)
    outcomes = _with_attempt_indices(outcomes)
    fields = list(AllocationOutcome.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for outcome in outcomes:
            writer.writerow(asdict(outcome))

    by_policy: dict[str, list[AllocationOutcome]] = {}
    for outcome in outcomes:
        by_policy.setdefault(outcome.policy, []).append(outcome)

    lines = ["# V0 UAV-VQA Resource Allocation Summary", ""]
    lines.append("Batch mode is one-shot allocation. MDP mode is multi-slot rollout, so attempt-level and unique-task-level success are reported separately.")
    lines.append("")
    lines.append("| policy | attempt success | unique task success | completion | attempts/task | semantic utility | accuracy | delay | energy | quality sat. | deadline sat. | feasible | conflict | cache | s0 | s1 | s2 full | s3 ROI | reward |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy, rows in sorted(by_policy.items()):
        summary = summarize_outcomes(rows)
        lines.append(
            f"| {policy} | {summary['attempt_success_rate']:.3f} | {summary['unique_task_success_rate']:.3f} | "
            f"{summary['completion_rate']:.3f} | {summary['average_attempts_per_task']:.3f} | "
            f"{summary['average_semantic_utility']:.3f} | {summary['average_accuracy']:.3f} | "
            f"{summary['average_delay']:.3f} | {summary['average_energy']:.3f} | "
            f"{summary['quality_satisfaction_rate']:.3f} | {summary['deadline_satisfaction_rate']:.3f} | "
            f"{summary['feasible_task_rate']:.3f} | {summary['airspace_conflict_rate']:.3f} | "
            f"{summary['cache_hit_rate']:.3f} | {summary['service_level_0_rate']:.3f} | "
            f"{summary['service_level_1_rate']:.3f} | {summary['service_level_2_rate']:.3f} | "
            f"{summary['service_level_3_rate']:.3f} | "
            f"{summary['average_reward']:.3f} |"
        )
    lines.extend(["", "## Feasibility Breakdown", ""])
    lines.append("| policy | feasible | deadline-limited | quality-limited | quality violation | deadline violation | resource violation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for policy, rows in sorted(by_policy.items()):
        summary = summarize_outcomes(rows)
        lines.append(
            f"| {policy} | {summary['feasible_task_rate']:.3f} | {summary['deadline_limited_rate']:.3f} | "
            f"{summary['quality_limited_rate']:.3f} | {summary['quality_violation_rate']:.3f} | "
            f"{summary['deadline_violation_rate']:.3f} | {summary['resource_violation_rate']:.3f} |"
        )
    lines.extend(["", "## Communication-Compute Breakdown", ""])
    lines.append("| policy | rate Mbps | SNR dB | SINR dB | dist m | elev deg | LoS prob. | path loss | interference | fading | travel delay | upload delay | inference delay | model load delay |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy, rows in sorted(by_policy.items()):
        summary = summarize_outcomes(rows)
        lines.append(
            f"| {policy} | {summary['average_rate_mbps']:.3f} | {summary['average_snr_db']:.3f} | "
            f"{summary['average_sinr_db']:.3f} | {summary['average_distance_3d_m']:.3f} | "
            f"{summary['average_elevation_deg']:.3f} | {summary['average_los_probability']:.3f} | "
            f"{summary['average_path_loss_db']:.3f} | {summary['average_interference_dbm']:.3f} | "
            f"{summary['average_fading_db']:.3f} | {summary['average_travel_delay']:.3f} | {summary['average_upload_delay']:.3f} | "
            f"{summary['average_inference_delay']:.3f} | {summary['average_model_load_delay']:.3f} |"
        )
    lines.extend(["", "## Semantic Cache, MEC, and UAV Breakdown", ""])
    lines.append("| policy | cache hit prob. | semantic cache hit | acc. gain | payload MB | semantic eff. | LUT missing | utility/latency | GPU mem ok | battery ok | battery remaining | UAV util. |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy, rows in sorted(by_policy.items()):
        summary = summarize_outcomes(rows)
        lines.append(
            f"| {policy} | {summary['average_cache_hit_probability']:.3f} | {summary['semantic_cache_hit_rate']:.3f} | "
            f"{summary['average_accuracy_gain']:.3f} | {summary['average_payload_mb']:.3f} | "
            f"{summary['average_semantic_efficiency']:.3f} | {summary['lut_missing_rate']:.3f} | "
            f"{summary['average_utility_per_latency']:.3f} | {summary['gpu_memory_ok_rate']:.3f} | "
            f"{summary['battery_ok_rate']:.3f} | {summary['average_battery_remaining']:.3f} | "
            f"{summary['average_uav_utilization']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_resource_inputs(tasks_path: Path, lut_path: Path, cfg: dict[str, Any] | None = None) -> tuple[list[dict[str, str]], SemanticLUT]:
    payload_cfg = (cfg or {}).get("resource_env", {}).get("payload_mb_by_level", {})
    return read_csv(tasks_path), load_semantic_lut(lut_path, payload_cfg)
