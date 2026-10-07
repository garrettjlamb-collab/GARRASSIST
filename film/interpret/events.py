from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from film.interpret.arrivals import Arrival
from film.perceive.scoreboard import ScoreboardRead, lookup

CLIP_BEFORE_S = 8.0       # the clip shows the play building up to the shot...
CLIP_AFTER_S = 3.0        # ...and ends once the ball is through or off the rim
SCORE_WINDOW_S = 6.0
NEXT_ARRIVAL_MARGIN_S = 0.5
SCORE_LEAD_S = 1.0
SB_ONLY_LOOKBACK_S = 7.0
SB_ONLY_LEAD_S = 2.0
SB_ONLY_RIM_ID = -1


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
    """(home delta, away delta) between the read at t - SCORE_LEAD_S and the read at t + window_s.

    None when the "after" read is not later than t: a dropout across the arrival is no evidence."""
    before = lookup(timeline, t - SCORE_LEAD_S)
    after = lookup(timeline, t + window_s)
    if before is None or after is None or after.t <= t:
        return None
    if None in (before.home, before.away, after.home, after.away):
        return None
    return after.home - before.home, after.away - before.away


def _window_for(i: int, arrivals: list[Arrival]) -> float:
    """The score window, cut short at the next arrival at any rim, so one basket is credited once."""
    if i + 1 >= len(arrivals):
        return SCORE_WINDOW_S
    gap = arrivals[i + 1].t - arrivals[i].t
    return max(0.5, min(SCORE_WINDOW_S, gap - NEXT_ARRIVAL_MARGIN_S))


# Without a scoreboard a card's confidence comes from how the two classifiers agree (see film.interpret.fuse).
# Strong agreement was a real shot with the right call on 90% of such cards across six reviewed games: the
# same as a scoreboard that agrees with the classifier, so it earns the same "safe to accept" confidence.
AGREEMENT_CONFIDENCE = {"strong": 0.92, "fair": 0.8, "split": 0.4}


def build_shot_events(arrivals: list[Arrival], timeline: list[ScoreboardRead],
                      clf_preds: list[tuple[str, float] | None], game_id: str,
                      opinions: list[dict | None] | None = None) -> list[Event]:
    """`clf_preds` is the made/missed/none call per arrival; `opinions` (optional, see fuse) says how the
    ball-path and picture classifiers each voted and how well they agreed."""
    events: list[Event] = []
    opinions = opinions or [None] * len(arrivals)
    for i, (arrival, pred) in enumerate(zip(arrivals, clf_preds)):
        opinion = opinions[i]
        agreement = opinion.get("agreement") if opinion else None
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
        if opinion:
            signals.update({"clf_path": opinion.get("path"), "clf_pictures": opinion.get("pictures"), "agreement": agreement})
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
                    made, conf = clf_made, AGREEMENT_CONFIDENCE.get(agreement, 0.7 * prob)
                    flags.append(no_evidence_flag)
                    if agreement == "split":
                        flags.append("classifiers-disagree")
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


def scoreboard_only_events(timeline: list[ScoreboardRead], events: list[Event]) -> list[Event]:
    """A made basket for every scoreboard rise that no event explains.

    Walks the reads that carry both scores; a rise of 1 to 3 points by exactly one team, with no event
    (at any rim) in the SB_ONLY_LOOKBACK_S before it (or less than SCORE_LEAD_S after it), becomes a shot placed SB_ONLY_LEAD_S before the
    rise. It has no rim, so no montage and no classifier; the coach places it."""
    arrival_times = sorted(e.t for e in events)
    out: list[Event] = []
    prev: ScoreboardRead | None = None
    for read in timeline:
        if read.home is None or read.away is None:
            continue
        if prev is not None:
            dh, da = read.home - prev.home, read.away - prev.away
            if (dh in (1, 2, 3) and da == 0) or (da in (1, 2, 3) and dh == 0):
                rise_t = read.t
                # score_change credits an arrival with a rise up to SCORE_LEAD_S before it, so an arrival
                # that close after the rise has already claimed it.
                explained = any(rise_t - SB_ONLY_LOOKBACK_S <= t < rise_t + SCORE_LEAD_S for t in arrival_times)
                if not explained:
                    out.append(_scoreboard_only_event(timeline, rise_t, dh, da))
        prev = read
    return out


def _scoreboard_only_event(timeline: list[ScoreboardRead], rise_t: float, dh: int, da: int) -> Event:
    t = round(max(0.0, rise_t - SB_ONLY_LEAD_S), 2)
    points = dh or da
    read = lookup(timeline, t)
    flags = ["scoreboard-only"] + (["free-throw?"] if points == 1 else [])
    return Event(
        id=f"shot-{int(round(t * 10)):06d}-sb", type="shot", t=t, rim_id=SB_ONLY_RIM_ID,
        quarter=read.period if read else None, clock_s=read.clock_s if read else None,
        team="home" if dh else "away", made=True, confidence=0.5,
        clip={"start": round(max(0.0, t - CLIP_BEFORE_S), 2), "end": round(t + CLIP_AFTER_S, 2)},
        signals={"source": "scoreboard", "score_delta": [dh, da], "scoreboard": "made", "points": points},
        flags=flags, needs_location=True,
    )


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


def mark_dead_time(events: list[Event], phases) -> int:
    """Flag events outside the game's quarters as warmups; returns how many.

    Arrivals in pregame, breaks, halftime or post-game time are warmup shots. A scoreboard-only event
    is a basket the operator posted, so it stays this game's even during a break, unless the board
    had already been reset for the next game."""
    from film.interpret.phases import is_live, is_this_game

    marked = 0
    for e in events:
        if e.signals.get("source") == "scoreboard":
            live = is_this_game(phases, e.t)
        else:
            live = is_live(phases, e.t)
        e.signals["phase"] = "live" if live else "warmup"
        if live:
            e.flags = [f for f in e.flags if f != "warmup"]             # marks can move: a flag is never stale
        else:
            if "warmup" not in e.flags:
                e.flags.append("warmup")
            marked += 1
    return marked
