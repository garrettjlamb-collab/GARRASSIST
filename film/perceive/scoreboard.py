from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from film.common.video import iter_frames
from film.perceive.sevenseg import box_fraction, decode_cell, decode_number, led_mask

RECENT_S = 10.0       # plausibility checks only apply against a read this recent
MAX_SCORE_JUMP = 3
MIN_CONF = 0.5


@dataclass
class ScoreboardRead:
    t: float
    period: int | None
    clock_s: float | None
    home: int | None
    away: int | None
    home_fouls: int | None
    away_fouls: int | None
    conf: float

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "ScoreboardRead":
        return cls(**data)


def _profile_path(name: str):
    return files("film.perceive.profiles") / f"{name.replace('-', '_')}.json"


def has_scoreboard(name: str) -> bool:
    try:
        return _profile_path(name).is_file()
    except (TypeError, ValueError):
        return False


def load_profile(name: str) -> dict:
    return json.loads(_profile_path(name).read_text())


def _cells(field: dict) -> list[tuple[int, int, int, int]]:
    return [tuple(c) for c in field["cells"]]


def read_clock(mask: np.ndarray, field: dict, threshold: float = 0.3) -> float | None:
    """Clock cells read as M M : S S when both separator dots are lit, S S . t when only the bottom one is."""
    digits = [decode_cell(mask, cell, threshold) for cell in _cells(field)]
    if any(d == -1 for d in digits) or all(d is None for d in digits):
        return None
    d = [0 if x is None else x for x in digits]
    top = box_fraction(mask, field["sep_top"]) >= 0.2
    bottom = box_fraction(mask, field["sep_bottom"]) >= 0.2
    if top and bottom:
        return float(d[0] * 600 + d[1] * 60 + d[2] * 10 + d[3])
    if bottom:
        return float(d[0] * 10 + d[1] + d[2] / 10)
    return None


def read_scoreboard(frame_bgr: np.ndarray, profile: dict, t: float = 0.0) -> ScoreboardRead:
    fields = profile["fields"]
    threshold = float(profile.get("threshold", 0.3))        # LED-dot skins fill a segment zone only sparsely
    masks: dict[str, np.ndarray] = {}

    def mask_for(color: str) -> np.ndarray:
        if color not in masks:
            masks[color] = led_mask(frame_bgr, color)
        return masks[color]

    def number(name: str) -> int | None:
        field = fields[name]
        return decode_number(mask_for(field["color"]), _cells(field), threshold)

    home, away, period = number("home"), number("away"), number("period")
    # A skin whose clock font the segment zones cannot read leaves the clock out; the scores still give
    # made-or-missed, and quarter marks supply what the clock would have.
    clock = read_clock(mask_for(fields["clock"]["color"]), fields["clock"], threshold) if "clock" in fields else None
    home_fouls = number("home_fouls") if "home_fouls" in fields else None
    away_fouls = number("away_fouls") if "away_fouls" in fields else None
    got = sum(v is not None for v in (home, away, clock, period))
    return ScoreboardRead(t=t, period=period, clock_s=clock, home=home, away=away,
                          home_fouls=home_fouls, away_fouls=away_fouls, conf=got / 4)


def build_timeline(source: Path, profile: dict, limit_s: float | None = None) -> list[ScoreboardRead]:
    return [read_scoreboard(frame, profile, t=t) for t, frame in iter_frames(source, fps=1.0, end=limit_s)]


def _plausible(read: ScoreboardRead, last: ScoreboardRead) -> bool:
    if read.t - last.t > RECENT_S:
        return True
    if read.period is not None and last.period is not None and read.period < last.period:
        return False
    same_period = read.period == last.period
    if same_period and read.clock_s is not None and last.clock_s is not None and read.clock_s > last.clock_s + 0.5:
        return False
    for now, before in ((read.home, last.home), (read.away, last.away)):
        if now is not None and before is not None and (now < before or now - before > MAX_SCORE_JUMP):
            return False
    return True


def clean_timeline(reads: list[ScoreboardRead]) -> list[ScoreboardRead]:
    kept: list[ScoreboardRead] = []
    for read in reads:
        if read.conf < MIN_CONF:
            continue
        if kept and not _plausible(read, kept[-1]):
            continue
        kept.append(read)
    return kept


def lookup(timeline: list[ScoreboardRead], t: float) -> ScoreboardRead | None:
    if not timeline:
        return None
    times = [r.t for r in timeline]
    i = bisect.bisect_right(times, t) - 1
    return timeline[i] if i >= 0 else None


def save_timeline(reads: list[ScoreboardRead], path: Path) -> Path:
    path = Path(path)
    path.write_text("".join(json.dumps(r.to_json()) + "\n" for r in reads))
    return path


def load_timeline(path: Path) -> list[ScoreboardRead]:
    path = Path(path)
    if not path.exists():
        return []
    return [ScoreboardRead.from_json(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
