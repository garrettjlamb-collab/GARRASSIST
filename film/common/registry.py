from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from film.common.paths import models_dir


def _versions(name: str) -> list[Path]:
    root = models_dir() / name
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
    return sorted(dirs, key=lambda p: int(p.name[1:]))


def latest(name: str) -> tuple[str, Path] | None:
    """The newest version folder of a model that actually holds weights."""
    ready = [p for p in _versions(name) if (p / "best.pt").exists()]
    return (ready[-1].name, ready[-1]) if ready else None


def base_weights_path(name: str) -> Path:
    """Where pretrained base weights live: models/base/<name>, absolute, so ultralytics downloads there
    rather than into whatever directory the command ran from. A copy already sitting in the current
    directory from an older run is moved into place instead of downloaded again."""
    folder = models_dir() / "base"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    stray = Path.cwd() / name
    if not target.exists() and stray.is_file():
        os.replace(stray, target)
    return target


def new_version_dir(name: str) -> tuple[str, Path]:
    existing = _versions(name)
    n = int(existing[-1].name[1:]) + 1 if existing else 1
    folder = models_dir() / name / f"v{n}"
    folder.mkdir(parents=True, exist_ok=False)
    return f"v{n}", folder


def write_manifest(folder: Path, **info) -> Path:
    info.setdefault("written_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path = Path(folder) / "manifest.json"
    path.write_text(json.dumps(info, indent=2))
    return path


def read_manifest(folder: Path) -> dict:
    return json.loads((Path(folder) / "manifest.json").read_text())
