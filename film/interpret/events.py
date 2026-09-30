from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from film.interpret.arrivals import Arrival
from film.perceive.scoreboard import ScoreboardRead, lookup

CLIP_BEFORE_S = 3.0
CLIP_AFTER_S = 1.5
SCORE_WINDOW_S = 6.0
NEXT_ARRIVAL_MARGIN_S = 0.5
SCORE_LEAD_S = 1.0


@dataclass
class Event:
    id: str
    type: str
    t: float
    rim_id: int
    quarter: int | None
    clock_s: float | None
    team: str | None
    made: bool | None
    confidence: float
    clip: dict
    signals: dict
    flags: list = field(default_factory=list)
    needs_location: bool = True

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Event":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def event_id(t: float, rim_id: int) -> str:
    return f"shot-{int(round(t * 10)):06d}-r{rim_id}"


def score_change(timeline: list[ScoreboardRead], t: float, window_s: float) -> tuple[int, int] | None:
    """(home delta, away delta) between the read at t - SCORE_LEAD_S and the read at t + window_s."""
    before = lookup(timeline, t - SCORE_LEAD_S)
    after = lookup(timeline, t + window_s)
    if before is None or after is None:
        return None
    if None in (before.home, before.away, after.home, after.away):
        return None
    return after.home - before.home, after.away - before.away


def _window_for(i: int, arrivals: list[Arrival]) -> float:
    this = arrivals[i]
    for nxt in arrivals[i + 1:]:
        if nxt.rim_id == this.rim_id:
            return max(0.5, min(SCORE_WINDOW_S, nxt.t - this.t - NEXT_ARRIVAL_MARGIN_S))
    return SCORE_WINDOW_S


def build_shot_events(arrivals: list[Arrival], timeline: list[ScoreboardRead],
                      clf_preds: list[tuple[str, float] | None], game_id: str) -> list[Event]:
    events: list[Event] = []
    for i, (arrival, pred) in enumerate(zip(arrivals, clf_preds)):
        delta = score_change(timeline, arrival.t, _window_for(i, arrivals))
        d_home, d_away = delta if delta is not None else (0, 0)
        home_scored, away_scored = d_home in (1, 2, 3), d_away in (1, 2, 3)
        if delta is None:
            verdict = "unknown"
        elif home_scored != away_scored and (d_home if home_scored else d_away) in (1, 2, 3) \
                and (d_away if home_scored else d_home) == 0:
            verdict = "made"
        elif delta == (0, 0):
            verdict = "missed"
        else:
            verdict = "ambiguous"
        score_made = verdict == "made"
        no_evidence = verdict in ("unknown", "ambiguous")
        no_evidence_flag = "no-scoreboard" if verdict == "unknown" else "scoreboard-ambiguous"
        team = "home" if score_made and home_scored else ("away" if score_made and away_scored else None)
        points = (d_home if home_scored else d_away) if score_made else 0
        read = lookup(timeline, arrival.t)
        flags: list[str] = []
        kind = "shot"
        signals = {"source": arrival.source, "arrival_conf": arrival.conf, "clf": list(pred) if pred else None,
                   "score_delta": list(delta) if delta is not None else None, "scoreboard": verdict, "points": points}
        if pred is None:
            flags.append("no-classifier")
            if no_evidence:
                made, conf = None, 0.3
                flags.append(no_evidence_flag)
            else:
                made, conf = score_made, (0.6 if score_made else 0.5)
        else:
            label, prob = pred
            if label == "none":
                flags.append("clf-none")
                if arrival.source == "motion":
                    kind, made, conf = "unknown", None, 0.2
                else:
                    made, conf = (False if no_evidence else score_made), 0.4
            else:
                clf_made = label == "made"
                if no_evidence:
                    made, conf = clf_made, 0.7 * prob
                    flags.append(no_evidence_flag)
                elif clf_made == score_made:
                    made, conf = clf_made, (0.95 if clf_made else 0.9)
                else:
                    made, conf = score_made, 0.5
                    flags.append("clf-scoreboard-disagree")
        if points == 1:
            flags.append("free-throw?")
        if arrival.source == "motion":
            conf = min(conf, 0.5)
            flags.append("motion-only")
        events.append(Event(
            id=event_id(arrival.t, arrival.rim_id), type=kind, t=arrival.t, rim_id=arrival.rim_id,
            quarter=read.period if read else None, clock_s=read.clock_s if read else None,
            team=team, made=made, confidence=round(conf, 2),
            clip={"start": round(max(0.0, arrival.t - CLIP_BEFORE_S), 2), "end": round(arrival.t + CLIP_AFTER_S, 2)},
            signals=signals, flags=flags,
        ))
    return events


def save_events(game_dir: Path, events: list[Event]) -> Path:
    path = Path(game_dir) / "events.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.to_json() for e in events], indent=1))
    return path


def load_events(game_dir: Path) -> list[Event]:
    path = Path(game_dir) / "events.json"
    if not path.exists():
        return []
    return [Event.from_json(d) for d in json.loads(path.read_text())]
