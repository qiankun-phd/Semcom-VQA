from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from vqa_semcom.config import ensure_parent
from vqa_semcom.data.visdrone import VisDroneObject


VEHICLE_CLASSES = {"car", "van", "truck", "bus", "motor", "tricycle", "awning-tricycle"}
PERSON_CLASSES = {"pedestrian", "people"}


@dataclass(frozen=True)
class VQATask:
    image_id: str
    question_type: str
    question: str
    answer: str
    target_class: str
    object_count: int
    risk_level: str
    epsilon_k: float
    tau_k: float
    view_quality_bin: str
    scale_proxy: float
    occlusion_score: float
    truncation_score: float
    density_score: float


def _view_score(
    scale_proxy: float,
    occlusion_score: float,
    truncation_score: float,
    density_score: float,
    scale_reference: float = 0.0035,
    density_penalty_start: float = 6.0,
    density_penalty_per_object: float = 0.025,
    min_density_component: float = 0.55,
) -> float:
    scale_component = min(1.0, scale_proxy / max(1e-9, scale_reference))
    occlusion_component = 1.0 - 0.35 * min(2.0, occlusion_score) / 2.0
    trunc_component = 1.0 - 0.25 * min(1.0, truncation_score)
    density_component = max(
        min_density_component,
        1.0 - density_penalty_per_object * max(0.0, density_score - density_penalty_start),
    )
    return max(0.0, min(1.0, scale_component * occlusion_component * trunc_component * density_component))


def _bin_view_score(score: float, good_score: float = 0.62, medium_score: float = 0.30) -> str:
    if score >= good_score:
        return "good"
    if score >= medium_score:
        return "medium"
    return "poor"


def _view_quality(
    scale_proxy: float,
    occlusion_score: float,
    truncation_score: float,
    density_score: float,
    scale_reference: float = 0.0035,
    good_score: float = 0.62,
    medium_score: float = 0.30,
    density_penalty_start: float = 6.0,
    density_penalty_per_object: float = 0.025,
    min_density_component: float = 0.55,
) -> tuple[str, float]:
    score = _view_score(
        scale_proxy,
        occlusion_score,
        truncation_score,
        density_score,
        scale_reference=scale_reference,
        density_penalty_start=density_penalty_start,
        density_penalty_per_object=density_penalty_per_object,
        min_density_component=min_density_component,
    )
    return _bin_view_score(score, good_score=good_score, medium_score=medium_score), score


def _risk_params(question_type: str, count: int) -> tuple[str, float, float]:
    if question_type == "risk" or count >= 10:
        return "critical", 0.88, 3.0
    if question_type == "relation":
        return "normal", 0.74, 4.5
    if question_type == "attribute":
        return "normal", 0.72, 4.5
    return "normal", 0.70, 5.0


def generate_tasks(objects: list[VisDroneObject], cfg: dict | None = None) -> list[VQATask]:
    thresholds = (cfg or {}).get("thresholds", {})
    view_cfg = (cfg or {}).get("view_quality", {})
    epsilon_normal = float(thresholds.get("epsilon_normal", 0.70))
    epsilon_critical = float(thresholds.get("epsilon_critical", 0.88))
    tau_normal = float(thresholds.get("tau_normal", 5.0))
    tau_critical = float(thresholds.get("tau_critical", 3.0))

    def risk_params(question_type: str, count: int) -> tuple[str, float, float]:
        if question_type == "risk" or count >= 10:
            return "critical", epsilon_critical, tau_critical
        if question_type == "relation":
            return "normal", max(epsilon_normal, 0.74), min(tau_normal, 4.5)
        if question_type == "attribute":
            return "normal", max(epsilon_normal, 0.72), min(tau_normal, 4.5)
        return "normal", epsilon_normal, tau_normal

    by_image: dict[str, list[VisDroneObject]] = defaultdict(list)
    for obj in objects:
        by_image[obj.image_id].append(obj)
    image_stats: dict[str, dict[str, float | str | list[VisDroneObject]]] = {}
    for image_id, image_objects in sorted(by_image.items()):
        density = float(len(image_objects))
        avg_scale = sum(obj.scale_proxy for obj in image_objects) / max(1, len(image_objects))
        avg_occ = sum(obj.occlusion for obj in image_objects) / max(1, len(image_objects))
        avg_trunc = sum(obj.truncation for obj in image_objects) / max(1, len(image_objects))
        view_score = _view_score(
            avg_scale,
            avg_occ,
            avg_trunc,
            density,
            scale_reference=float(view_cfg.get("scale_reference", 0.0035)),
            density_penalty_start=float(view_cfg.get("density_penalty_start", 6.0)),
            density_penalty_per_object=float(view_cfg.get("density_penalty_per_object", 0.025)),
            min_density_component=float(view_cfg.get("min_density_component", 0.55)),
        )
        image_stats[image_id] = {
            "objects": image_objects,
            "density": density,
            "avg_scale": avg_scale,
            "avg_occ": avg_occ,
            "avg_trunc": avg_trunc,
            "view_score": view_score,
        }
    quantile_thresholds: tuple[float, float] | None = None
    if view_cfg.get("binning", "absolute") == "quantile" and len(image_stats) >= 3:
        scores = sorted(float(stat["view_score"]) for stat in image_stats.values())
        medium_threshold = scores[max(0, min(len(scores) - 1, len(scores) // 3))]
        good_threshold = scores[max(0, min(len(scores) - 1, (2 * len(scores)) // 3))]
        if good_threshold > medium_threshold:
            quantile_thresholds = (medium_threshold, good_threshold)
    tasks: list[VQATask] = []
    for image_id, stat in sorted(image_stats.items()):
        image_objects = stat["objects"]
        counts = Counter(obj.category for obj in image_objects)
        density = float(stat["density"])
        avg_occ = float(stat["avg_occ"])
        avg_trunc = float(stat["avg_trunc"])
        view_score = float(stat["view_score"])
        if quantile_thresholds is None:
            view_bin = _bin_view_score(
                view_score,
                good_score=float(view_cfg.get("good_score", 0.62)),
                medium_score=float(view_cfg.get("medium_score", 0.30)),
            )
        else:
            medium_threshold, good_threshold = quantile_thresholds
            view_bin = "good" if view_score >= good_threshold else "medium" if view_score >= medium_threshold else "poor"

        for category, count in sorted(counts.items()):
            if category == "others":
                continue
            risk_level, epsilon, tau = risk_params("presence", count)
            tasks.append(
                VQATask(
                    image_id=image_id,
                    question_type="presence",
                    question=f"Are there {category} objects in this area?",
                    answer="yes" if count > 0 else "no",
                    target_class=category,
                    object_count=count,
                    risk_level=risk_level,
                    epsilon_k=epsilon,
                    tau_k=tau,
                    view_quality_bin=view_bin,
                    scale_proxy=round(view_score, 6),
                    occlusion_score=round(avg_occ, 6),
                    truncation_score=round(avg_trunc, 6),
                    density_score=round(density, 6),
                )
            )
            if category in VEHICLE_CLASSES | PERSON_CLASSES:
                risk_level, epsilon, tau = risk_params("counting", count)
                tasks.append(
                    VQATask(
                        image_id=image_id,
                        question_type="counting",
                        question=f"How many {category} objects are in this area?",
                        answer=str(count),
                        target_class=category,
                        object_count=count,
                        risk_level=risk_level,
                        epsilon_k=epsilon,
                        tau_k=tau,
                        view_quality_bin=view_bin,
                        scale_proxy=round(view_score, 6),
                        occlusion_score=round(avg_occ, 6),
                        truncation_score=round(avg_trunc, 6),
                        density_score=round(density, 6),
                    )
                )
                risk_level, epsilon, tau = risk_params("attribute", count)
                tasks.append(
                    VQATask(
                        image_id=image_id,
                        question_type="attribute",
                        question=f"What is the visible condition of the {category} objects in this area?",
                        answer="occluded" if avg_occ > 0.5 or avg_trunc > 0.35 else "clear",
                        target_class=category,
                        object_count=count,
                        risk_level=risk_level,
                        epsilon_k=epsilon,
                        tau_k=tau,
                        view_quality_bin=view_bin,
                        scale_proxy=round(view_score, 6),
                        occlusion_score=round(avg_occ, 6),
                        truncation_score=round(avg_trunc, 6),
                        density_score=round(density, 6),
                    )
                )

        vehicle_count = sum(count for cls, count in counts.items() if cls in VEHICLE_CLASSES)
        person_count = sum(count for cls, count in counts.items() if cls in PERSON_CLASSES)
        if vehicle_count > 0 and person_count > 0:
            risk_level, epsilon, tau = risk_params("relation", int(max(vehicle_count, person_count)))
            relation_answer = "near" if density >= 5 else "separate"
            tasks.append(
                VQATask(
                    image_id=image_id,
                    question_type="relation",
                    question="Are people close to vehicles in this area?",
                    answer=relation_answer,
                    target_class="person_vehicle",
                    object_count=int(vehicle_count + person_count),
                    risk_level=risk_level,
                    epsilon_k=epsilon,
                    tau_k=tau,
                    view_quality_bin=view_bin,
                    scale_proxy=round(view_score, 6),
                    occlusion_score=round(avg_occ, 6),
                    truncation_score=round(avg_trunc, 6),
                    density_score=round(density, 6),
                )
            )
        crowded_or_blocked = vehicle_count >= 4 or person_count >= 6 or density >= 8
        risk_level, epsilon, tau = risk_params("risk", int(max(vehicle_count, person_count, density)))
        tasks.append(
            VQATask(
                image_id=image_id,
                question_type="risk",
                question="Is this area crowded or blocked?",
                answer="yes" if crowded_or_blocked else "no",
                target_class="scene",
                object_count=int(density),
                risk_level=risk_level,
                epsilon_k=epsilon,
                tau_k=tau,
                view_quality_bin=view_bin,
                scale_proxy=round(view_score, 6),
                occlusion_score=round(avg_occ, 6),
                truncation_score=round(avg_trunc, 6),
                density_score=round(density, 6),
            )
        )
    return tasks


def write_tasks_csv(tasks: list[VQATask], path: Path) -> None:
    ensure_parent(path)
    fieldnames = list(asdict(tasks[0]).keys()) if tasks else list(VQATask.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            writer.writerow(asdict(task))
