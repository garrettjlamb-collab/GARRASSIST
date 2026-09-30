from __future__ import annotations

from pathlib import Path

from film.common.paths import game_dir
from film.interpret.events import load_events
from film.review.decisions import load_decisions


def load_reference(path: Path) -> list[float]:
    """One shot time per line: seconds, M:SS of video time, or t=<seconds>. Blank lines and # comments ignored."""
    times = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("t="):
            line = line[2:]
        if ":" in line:
            minutes, seconds = line.split(":", 1)
            times.append(float(minutes) * 60 + float(seconds))
        else:
            times.append(float(line))
    return times


def match_times(machine: list[float], reference: list[float], tol_s: float = 2.0) -> tuple[int, int, int]:
    """Greedy one-to-one matching in time order. Returns (matched, unmatched machine, unmatched reference)."""
    machine, reference = sorted(machine), sorted(reference)
    matched, i, j = 0, 0, 0
    while i < len(machine) and j < len(reference):
        gap = machine[i] - reference[j]
        if abs(gap) <= tol_s:
            matched += 1
            i += 1
            j += 1
        elif gap < 0:
            i += 1
        else:
            j += 1
    return matched, len(machine) - matched, len(reference) - matched


def evaluate(game_id: str, reference_times: list[float] | None = None, tol_s: float = 2.0) -> dict:
    gd = game_dir(game_id)
    events = load_events(gd)
    decisions = load_decisions(gd)
    shots = [e for e in events if e.type == "shot"]
    by_status = {s: [e for e in shots if decisions.get(e.id, {}).get("status") == s] for s in ("confirmed", "rejected", "unsure")}
    confirmed, rejected, unsure = (len(by_status[s]) for s in ("confirmed", "rejected", "unsure"))
    compared = [e for e in by_status["confirmed"] if e.made is not None]
    agree = sum(1 for e in compared if decisions[e.id].get("made", e.made) == e.made)
    result = {
        "events": len(events), "shots": len(shots), "confirmed": confirmed, "rejected": rejected, "unsure": unsure,
        "undecided": len(shots) - confirmed - rejected - unsure,
        "precision": confirmed / (confirmed + rejected) if confirmed + rejected else None,
        "made_compared": len(compared), "made_agreement": agree / len(compared) if compared else None,
    }
    if reference_times is not None:
        machine = [e.t for e in by_status["confirmed"] + by_status["unsure"]]
        matched, _, _ = match_times(machine, reference_times, tol_s)
        result.update({"reference": len(reference_times), "matched": matched,
                       "recall": matched / len(reference_times) if reference_times else None,
                       "precision_vs_reference": matched / len(machine) if machine else None})
    return result
