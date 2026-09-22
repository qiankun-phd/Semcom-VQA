from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from vqa_semcom.sim.vqa_resource_env import AllocationDecision, VQAResourceEnv, VQATaskState


class ResourceAllocator(Protocol):
    name: str

    def allocate(self, env: VQAResourceEnv) -> list[AllocationDecision]:
        ...


def _fair_share(total_tasks: int, preferred: float) -> float:
    return min(preferred, 1.0 / max(1, total_tasks))


def _base_decision(env: VQAResourceEnv, task: VQATaskState, level: int, total_tasks: int) -> AllocationDecision:
    uav = env.nearest_uav(task)
    bw = _fair_share(total_tasks, 0.6 if task.risk_level == "critical" else 0.4)
    cpu = _fair_share(total_tasks, 0.6 if task.risk_level == "critical" else 0.4)
    gpu = _fair_share(total_tasks, 0.5 if task.risk_level == "critical" else 0.3)
    power_levels = list(env.cfg["power_levels_w"])
    power = float(power_levels[-1] if task.risk_level == "critical" else power_levels[min(2, len(power_levels) - 1)])
    sensing = "reuse_cache" if level == 0 else "observe"
    return AllocationDecision(
        task_id=task.task_id,
        assigned_uav=uav.uav_id,
        sensing_decision=sensing,
        service_level=level,
        bandwidth_share=bw,
        power_w=power,
        cpu_share=cpu,
        gpu_share=gpu,
    )


def _minimum_quality_level(env: VQAResourceEnv, task: VQATaskState) -> int:
    levels = env.service_levels()
    for level in levels:
        if env.lookup_accuracy(task, level) >= task.epsilon_k:
            return level
    return levels[-1]


class CacheFirstAllocator:
    name = "cache_first"

    def allocate(self, env: VQAResourceEnv) -> list[AllocationDecision]:
        total = len(env.tasks)
        decisions: list[AllocationDecision] = []
        for task in env.tasks:
            if task.freshness_bin == "fresh" and env.lookup_accuracy(task, 0) >= task.epsilon_k:
                level = 0
            else:
                level = _minimum_quality_level(env, task)
            decisions.append(_base_decision(env, task, level, total))
        return decisions


class MinSufficientEvidenceAllocator:
    name = "min_sufficient_evidence"

    def allocate(self, env: VQAResourceEnv) -> list[AllocationDecision]:
        total = len(env.tasks)
        return [_base_decision(env, task, _minimum_quality_level(env, task), total) for task in env.tasks]


class DeadlineAwareGreedyAllocator:
    name = "deadline_aware_greedy"

    def allocate(self, env: VQAResourceEnv) -> list[AllocationDecision]:
        ordered = sorted(
            env.tasks,
            key=lambda t: (-(t.priority), t.tau_k, -env.lookup_accuracy(t, _minimum_quality_level(env, t))),
        )
        total = len(ordered)
        decisions: list[AllocationDecision] = []
        for rank, task in enumerate(ordered):
            level = _minimum_quality_level(env, task)
            decision = _base_decision(env, task, level, total)
            urgency_boost = max(0.0, (total - rank) / max(1, total)) * (0.15 / max(1, total))
            decision = replace(
                decision,
                bandwidth_share=min(1.0, decision.bandwidth_share + urgency_boost),
                cpu_share=min(1.0, decision.cpu_share + urgency_boost),
                gpu_share=min(1.0, decision.gpu_share + urgency_boost),
            )
            decisions.append(decision)
        return _renormalize(decisions)


class JointGreedyResourceAllocator:
    name = "joint_greedy_resource"

    def allocate(self, env: VQAResourceEnv) -> list[AllocationDecision]:
        total = len(env.tasks)
        decisions: list[AllocationDecision] = []
        for task in sorted(env.tasks, key=lambda t: (-t.priority, t.tau_k)):
            decisions.append(self._best_for_task(env, task, total))
        return _renormalize(decisions)

    def _best_for_task(self, env: VQAResourceEnv, task: VQATaskState, total: int) -> AllocationDecision:
        candidates: list[tuple[float, AllocationDecision]] = []
        bandwidths = sorted({_fair_share(total, float(x)) for x in env.cfg["bandwidth_shares"]})
        cpus = sorted({_fair_share(total, float(x)) for x in env.cfg["cpu_shares"]})
        gpus = sorted({_fair_share(total, float(x)) for x in env.cfg["gpu_shares"]})
        powers = [float(x) for x in env.cfg["power_levels_w"]]
        for level in env.service_levels():
            for bw in bandwidths:
                for cpu in cpus:
                    for gpu in gpus:
                        for power in powers:
                            decision = replace(_base_decision(env, task, level, total), bandwidth_share=bw, cpu_share=cpu, gpu_share=gpu, power_w=power)
                            outcome = env.evaluate(task, decision, policy=self.name, resource_ok=True, airspace_conflict=False)
                            violation = float(not outcome.quality_ok) * 50.0 + float(not outcome.deadline_ok) * 25.0
                            service_cost = 0.05 * level
                            score = violation + outcome.delay + 0.4 * outcome.energy + service_cost
                            candidates.append((score, decision))
        return min(candidates, key=lambda item: item[0])[1]


class GoalOrientedPriorityAllocator:
    name = "goal_oriented_priority"

    def allocate(self, env: VQAResourceEnv) -> list[AllocationDecision]:
        ordered = sorted(
            env.tasks,
            key=lambda t: (
                env.feasibility_diagnostics(t)["feasibility_class"] != "feasible",
                -(t.priority / max(0.25, t.tau_k)),
                -t.priority,
                t.tau_k,
            ),
        )
        remaining_bw = 1.0
        remaining_cpu = 1.0
        remaining_gpu = 1.0
        decisions: list[AllocationDecision] = []
        for task in ordered:
            decision = self._reserve_for_task(env, task, remaining_bw, remaining_cpu, remaining_gpu)
            decisions.append(decision)
            remaining_bw = max(0.0, remaining_bw - decision.bandwidth_share)
            remaining_cpu = max(0.0, remaining_cpu - decision.cpu_share)
            remaining_gpu = max(0.0, remaining_gpu - decision.gpu_share)
        return decisions

    def _reserve_for_task(self, env: VQAResourceEnv, task: VQATaskState, remaining_bw: float, remaining_cpu: float, remaining_gpu: float) -> AllocationDecision:
        total = max(1, len(env.tasks))
        fallback = _base_decision(env, task, 0, total)
        fallback = replace(
            fallback,
            bandwidth_share=min(remaining_bw, fallback.bandwidth_share),
            cpu_share=min(remaining_cpu, fallback.cpu_share),
            gpu_share=min(remaining_gpu, fallback.gpu_share),
            power_w=float(env.cfg["power_levels_w"][0]),
        )
        min_quality = _minimum_quality_level(env, task)
        if env.feasibility_diagnostics(task)["feasibility_class"] == "quality_limited":
            return fallback
        level_candidates = [level for level in env.service_levels() if level >= min_quality]
        if task.freshness_bin == "fresh":
            level_candidates = [0, *[l for l in level_candidates if l != 0]]
        best_soft: tuple[float, AllocationDecision] | None = None
        best_success: tuple[float, AllocationDecision] | None = None
        for level in level_candidates:
            sensing = self._sensing_for(task, level)
            for bw in sorted((float(x) for x in env.cfg["bandwidth_shares"]), reverse=False):
                if bw > remaining_bw + 1e-9:
                    continue
                for cpu in sorted((float(x) for x in env.cfg["cpu_shares"]), reverse=False):
                    if cpu > remaining_cpu + 1e-9:
                        continue
                    for gpu in sorted((float(x) for x in env.cfg["gpu_shares"]), reverse=False):
                        if gpu > remaining_gpu + 1e-9:
                            continue
                        for power in sorted(float(x) for x in env.cfg["power_levels_w"]):
                            decision = replace(
                                _base_decision(env, task, level, total),
                                sensing_decision=sensing,
                                bandwidth_share=bw,
                                cpu_share=cpu,
                                gpu_share=gpu,
                                power_w=power,
                            )
                            outcome = env.evaluate(task, decision, policy=self.name, resource_ok=True, airspace_conflict=False)
                            score = (
                                100.0 * float(not outcome.quality_ok)
                                + 60.0 * float(not outcome.deadline_ok)
                                + outcome.delay
                                + 0.35 * outcome.energy
                                + 0.2 * level
                            )
                            if best_soft is None or score < best_soft[0]:
                                best_soft = (score, decision)
                            if outcome.quality_ok and outcome.deadline_ok:
                                if task.priority >= 2.0:
                                    quality_margin = outcome.expected_accuracy - task.epsilon_k
                                    success_score = (
                                        -20.0 * quality_margin
                                        + 0.5 * outcome.delay
                                        + 0.1 * outcome.energy
                                        - 0.05 * (bw + cpu + gpu)
                                    )
                                    if best_success is None or success_score < best_success[0]:
                                        best_success = (success_score, decision)
                                    continue
                                return decision
        if best_success is not None:
            return best_success[1]
        if task.priority >= 2.0 and best_soft is not None:
            return best_soft[1]
        return fallback

    @staticmethod
    def _sensing_for(task: VQATaskState, level: int) -> str:
        if level == 0:
            return "reuse_cache"
        return "observe"


def _renormalize(decisions: list[AllocationDecision]) -> list[AllocationDecision]:
    bw_sum = sum(max(0.0, d.bandwidth_share) for d in decisions)
    cpu_sum = sum(max(0.0, d.cpu_share) for d in decisions)
    gpu_sum = sum(max(0.0, d.gpu_share) for d in decisions)
    out: list[AllocationDecision] = []
    for d in decisions:
        bw = d.bandwidth_share / bw_sum if bw_sum > 1.0 else d.bandwidth_share
        cpu = d.cpu_share / cpu_sum if cpu_sum > 1.0 else d.cpu_share
        gpu = d.gpu_share / gpu_sum if gpu_sum > 1.0 else d.gpu_share
        out.append(replace(d, bandwidth_share=bw, cpu_share=cpu, gpu_share=gpu))
    return out


ALLOCATORS: dict[str, ResourceAllocator] = {
    "cache_first": CacheFirstAllocator(),
    "min_sufficient_evidence": MinSufficientEvidenceAllocator(),
    "deadline_aware_greedy": DeadlineAwareGreedyAllocator(),
    "goal_oriented_priority": GoalOrientedPriorityAllocator(),
    "joint_greedy_resource": JointGreedyResourceAllocator(),
}
