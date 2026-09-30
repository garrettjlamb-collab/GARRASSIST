from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("confirmed", "rejected", "unsure")


def decisions_path(game_dir: Path) -> Path:
    return Path(game_dir) / "review.json"


def load_decisions(game_dir: Path) -> dict:
    path = decisions_path(game_dir)
    return json.loads(path.read_text()) if path.exists() else {}


def save_decision(game_dir: Path, event_id: str, decision: dict) -> dict:
    status = decision.get("status")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    entry = {"status": status, "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key in ("made", "team"):
        if key in decision:
            entry[key] = decision[key]
    everything = load_decisions(game_dir)
    everything[event_id] = entry
    Path(game_dir).mkdir(parents=True, exist_ok=True)
    decisions_path(game_dir).write_text(json.dumps(everything, indent=1))
    return entry


def apply_decisions(events: list[dict], decisions: dict) -> list[dict]:
    """Events with the coach's decision attached and their made/team overridden where the coach set them."""
    out = []
    for event in events:
        merged = dict(event)
        decision = decisions.get(event["id"])
        merged["decision"] = decision
        if decision:
            for key in ("made", "team"):
                if key in decision:
                    merged[key] = decision[key]
        out.append(merged)
    return out
