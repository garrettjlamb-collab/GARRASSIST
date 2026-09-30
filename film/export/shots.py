from __future__ import annotations

import csv
import json
from pathlib import Path

from film.common.paths import game_dir
from film.export.clips import build_reel, clip_path
from film.interpret.events import load_events
from film.review.decisions import apply_decisions, load_decisions

FIELDS = ["id", "t", "quarter", "clock_s", "team", "made", "confidence", "status", "flags"]


def reviewed_shots(game_id: str, statuses=("confirmed", "unsure")) -> list[dict]:
    gd = game_dir(game_id)
    merged = apply_decisions([e.to_json() for e in load_events(gd)], load_decisions(gd))
    rows = [e for e in merged if e["type"] == "shot" and e.get("decision") and e["decision"]["status"] in statuses]
    return sorted(rows, key=lambda e: e["t"])


def export_shots(game_id: str) -> tuple[Path, Path]:
    gd = game_dir(game_id)
    rows = reviewed_shots(game_id)
    out = gd / "export"
    out.mkdir(exist_ok=True)
    csv_path, json_path = out / f"{game_id}.shots.csv", out / f"{game_id}.shots.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for e in rows:
            writer.writerow({"id": e["id"], "t": e["t"], "quarter": e["quarter"], "clock_s": e["clock_s"], "team": e["team"],
                             "made": e["made"], "confidence": e["confidence"], "status": e["decision"]["status"],
                             "flags": " ".join(e.get("flags", []))})
    json_path.write_text(json.dumps({"format": "film-shot-times", "schemaVersion": 1, "gameId": game_id, "shots": rows}, indent=1))
    return csv_path, json_path


def _label(e: dict) -> str:
    clock = "--" if e["clock_s"] is None else f"{int(e['clock_s'] // 60)}:{int(e['clock_s'] % 60):02d}"
    result = "MADE" if e["made"] else "MISS" if e["made"] is False else "?"
    return f"Q{e['quarter'] or '?'} {clock} {result}"


def export_reel(game_id: str) -> Path | None:
    gd = game_dir(game_id)
    rows = [e for e in reviewed_shots(game_id, statuses=("confirmed",)) if clip_path(gd, e["id"]).exists()]
    if not rows:
        return None
    out = gd / "export"
    out.mkdir(exist_ok=True)
    return build_reel([(clip_path(gd, e["id"]), _label(e)) for e in rows], out / f"{game_id}.reel.mp4")
