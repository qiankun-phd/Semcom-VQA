from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VISDRONE_CATEGORY = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


@dataclass(frozen=True)
class VisDroneObject:
    image_id: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    score: float
    category_id: int
    category: str
    truncation: int
    occlusion: int
    image_width: int = 1920
    image_height: int = 1080

    @property
    def bbox_area(self) -> float:
        return max(0.0, self.bbox_w) * max(0.0, self.bbox_h)

    @property
    def image_area(self) -> float:
        return float(self.image_width * self.image_height)

    @property
    def scale_proxy(self) -> float:
        return self.bbox_area / max(1.0, self.image_area)


def parse_annotation_line(line: str, image_id: str, image_width: int = 1920, image_height: int = 1080) -> VisDroneObject | None:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 8:
        return None
    try:
        bbox_x, bbox_y, bbox_w, bbox_h = (float(parts[i]) for i in range(4))
        score = float(parts[4])
        category_id = int(float(parts[5]))
        truncation = int(float(parts[6]))
        occlusion = int(float(parts[7]))
    except ValueError:
        return None
    if category_id == 0:
        return None
    return VisDroneObject(
        image_id=image_id,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        score=score,
        category_id=category_id,
        category=VISDRONE_CATEGORY.get(category_id, "others"),
        truncation=truncation,
        occlusion=occlusion,
        image_width=image_width,
        image_height=image_height,
    )


def load_visdrone_annotations(root: str | Path, limit_images: int | None = None) -> list[VisDroneObject]:
    root_path = Path(root)
    ann_dir = root_path / "annotations"
    if not ann_dir.exists():
        raise FileNotFoundError(f"VisDrone annotation directory not found: {ann_dir}")
    objects: list[VisDroneObject] = []
    for idx, ann_path in enumerate(sorted(ann_dir.glob("*.txt"))):
        if limit_images is not None and idx >= limit_images:
            break
        image_id = ann_path.stem
        with ann_path.open(encoding="utf-8") as f:
            for line in f:
                obj = parse_annotation_line(line, image_id=image_id)
                if obj is not None:
                    objects.append(obj)
    return objects


def demo_objects() -> list[VisDroneObject]:
    return [
        VisDroneObject("demo_001", 100, 100, 220, 160, 1.0, 4, "car", 0, 0),
        VisDroneObject("demo_001", 380, 120, 70, 140, 1.0, 1, "pedestrian", 0, 1),
        VisDroneObject("demo_002", 50, 400, 42, 38, 1.0, 9, "bus", 1, 2),
        VisDroneObject("demo_003", 720, 610, 58, 70, 1.0, 10, "motor", 0, 1),
        VisDroneObject("demo_003", 790, 620, 54, 68, 1.0, 10, "motor", 0, 1),
        VisDroneObject("demo_004", 500, 420, 180, 130, 1.0, 5, "van", 0, 0),
    ]
