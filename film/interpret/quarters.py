"""Quarter marks the coach sets by hand, for film with no scoreboard to read the game clock from.

Each mark is a (start, end) pair of film times. They say what is live (everything else is warmup,
breaks, halftime) and which quarter a shot is in. Marks win over the scoreboard when both exist: they
are the coach's word."""
from __future__ import annotations

import json
from pathlib import Path

from film.interpret.phases import GamePhases

QUARTERS_FILE = "quarters.json"


def load_quarters(gd: Path) -> list[tuple[float, float]]:
    path = Path(gd) / QUARTERS_FILE
    if not path.exists():
        return []
    return [(float(a), float(b)) for a, b in json.loads(path.read_text())["quarters"]]


def save_quarters(gd: Path, quarters) -> list[tuple[float, float]]:
    marks = sorted((float(a), float(b)) for a, b in quarters)
    for a, b in marks:
        if a >= b:
            raise ValueError(f"a quarter must start before it ends: {a:.1f} to {b:.1f}")
    for (_, end), (start, _) in zip(marks, marks[1:]):
        if start < end:
            raise ValueError(f"quarters overlap: one ends at {end:.1f}, the next starts at {start:.1f}")
    path = Path(gd) / QUARTERS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"quarters": [[a, b] for a, b in marks]}, indent=1))
    return marks


def phases_from_quarters(quarters) -> GamePhases:
    return GamePhases(quarter_length_s=None, live=list(quarters), known=True)


def quarter_of(quarters, t: float) -> int | None:
    for i, (start, end) in enumerate(quarters):
        if start <= t < end:
            return i + 1
    return None


def mark_quarters(events, quarters) -> int:
    """Warmup flags and quarter numbers from the marks; returns how many events are warmups."""
    from film.interpret.events import mark_dead_time

    warmups = mark_dead_time(events, phases_from_quarters(quarters))
    for e in events:
        e.quarter = quarter_of(quarters, e.t)
    return warmups


def apply_quarters(gd: Path) -> dict:
    """Re-mark a game's saved events from its quarter marks, in place: no models, a second or two."""
    from film.interpret.events import load_events, save_events

    quarters = load_quarters(gd)
    if not (Path(gd) / "events.json").exists():          # still being processed: interpret applies them
        return {"events": 0, "warmups": 0, "quarters": len(quarters)}
    events = load_events(gd)
    warmups = mark_quarters(events, quarters) if quarters else sum("warmup" in e.flags for e in events)
    save_events(gd, events)
    return {"events": len(events), "warmups": warmups, "quarters": len(quarters)}
