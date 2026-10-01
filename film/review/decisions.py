from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
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
    Path(game_dir).mkdir(parents=True, exist_ok=True)
    path = decisions_path(game_dir)
    with _LOCK:
        everything = load_decisions(game_dir)
        everything[event_id] = entry
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(everything, indent=1))
        os.replace(tmp, path)
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


# How far a re-detected shot may sit from the card the coach decided and still inherit the decision.
# A rejection is only inherited by a near-identical moment: a real shot a second later must be reviewed.
MOVE_TOLERANCE_S = {"confirmed": 2.0, "unsure": 2.0, "rejected": 1.0}
SCOREBOARD_LOOKBACK_S = 6.0      # a scoreboard-only card sits ~2 s before the score change; the shot is earlier
SCOREBOARD_LOOKAHEAD_S = 3.0


def _get(event, name):
    return event[name] if isinstance(event, dict) else getattr(event, name)


def carry_over_decisions(game_dir: Path, old_events: list, new_events: list) -> dict:
    """Move each decision whose card no longer exists onto the re-detected card for the same shot.

    When the detectors improve, a shot's time shifts by a few tenths and its id changes. A decision
    follows its shot to the nearest undecided new card at the same rim within the tolerance; a
    scoreboard-only card hands its decision to the new card that now carries those points. Decisions
    with nowhere to go stay in review.json under their old id (orphaned), never deleted.
    Returns counts of kept, moved and orphaned decisions."""
    path = decisions_path(game_dir)
    with _LOCK:
        decisions = load_decisions(game_dir)
        if not decisions:
            return {"kept": 0, "moved": 0, "orphaned": 0}
        new_ids = {_get(e, "id") for e in new_events}
        old_by_id = {_get(e, "id"): e for e in old_events}
        taken = {i for i in decisions if i in new_ids}
        kept = len(taken)
        candidates: list[tuple[float, str, str]] = []
        for old_id, decision in decisions.items():
            if old_id in new_ids or old_id not in old_by_id:
                continue
            old = old_by_id[old_id]
            from_scoreboard = _get(old, "signals").get("source") == "scoreboard"
            for new in new_events:
                new_id = _get(new, "id")
                if new_id in taken:
                    continue
                gap = _get(new, "t") - _get(old, "t")
                if from_scoreboard:
                    fits = (_get(new, "signals").get("scoreboard") == "made"
                            and -SCOREBOARD_LOOKBACK_S <= gap <= SCOREBOARD_LOOKAHEAD_S)
                else:
                    fits = (_get(new, "rim_id") == _get(old, "rim_id")
                            and abs(gap) <= MOVE_TOLERANCE_S.get(decision.get("status"), 0.0))
                if fits:
                    candidates.append((abs(gap), old_id, new_id))
        moved: dict[str, str] = {}
        for _, old_id, new_id in sorted(candidates):
            if old_id in moved or new_id in taken:
                continue
            moved[old_id] = new_id
            taken.add(new_id)
        for old_id, new_id in moved.items():
            entry = dict(decisions.pop(old_id))
            entry["moved_from"] = old_id
            decisions[new_id] = entry
        orphaned = sum(1 for i in decisions if i not in new_ids)
        if moved:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(decisions, indent=1))
            os.replace(tmp, path)
    return {"kept": kept, "moved": len(moved), "orphaned": orphaned}
