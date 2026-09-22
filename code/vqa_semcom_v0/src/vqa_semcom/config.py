from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML config without requiring PyYAML."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    text = cfg_path.read_text(encoding="utf-8")
    return json.loads(text)


def resolve_path(path: str | Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else project_root() / out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
