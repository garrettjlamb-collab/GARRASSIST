from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Manifest:
    game_id: str
    source: str
    date: str
    duration_s: float
    width: int
    height: int
    src_fps: float
    fps: float
    profile: str
    stages: dict = field(default_factory=dict)
    model_versions: dict = field(default_factory=dict)

    @classmethod
    def load(cls, game_dir: Path) -> "Manifest":
        return cls(**json.loads((Path(game_dir) / "manifest.json").read_text()))

    def save(self, game_dir: Path) -> Path:
        game_dir = Path(game_dir)
        game_dir.mkdir(parents=True, exist_ok=True)
        path = game_dir / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    def mark(self, stage: str, status: str, **info) -> None:
        entry = {"status": status, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        entry.update(info)
        self.stages[stage] = entry
