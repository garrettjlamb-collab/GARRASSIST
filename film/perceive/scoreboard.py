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
    clock_alts: list | None = None    # clock values the reader could not choose between (glyph skins)

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


def read_clock(mask: np.ndarray, field: dict, threshold: float = 0.3, patterns: dict | None = None) -> float | None:
    """Clock cells read as M M : S S when both separator dots are lit, S S . t when only the bottom one is.
    A field with `format: "m:ss"` has no readable dots: its cells are always minutes then seconds, and a
    read needs the minutes cell and both seconds cells."""
    digits = [decode_cell(mask, cell, threshold, patterns) for cell in _cells(field)]
    if any(d == -1 for d in digits) or all(d is None for d in digits):
        return None
    if field.get("format") == "m:ss":
        if any(x is None for x in digits[-2:]):              # both seconds digits; a blank minutes cell is 0:ss
            return None
        d = [0 if x is None else x for x in digits]
        return float(d[0] * 600 + d[1] * 60 + d[2] * 10 + d[3])
    d = [0 if x is None else x for x in digits]
    top = box_fraction(mask, field["sep_top"]) >= 0.2
    bottom = box_fraction(mask, field["sep_bottom"]) >= 0.2
    if top and bottom:
        return float(d[0] * 600 + d[1] * 60 + d[2] * 10 + d[3])
    if bottom:
        return float(d[0] * 10 + d[1] + d[2] / 10)
    return None


_GLYPHS: dict[str, object] = {}


def _glyphs_for(profile: dict):
    from film.perceive.glyphs import Glyphs, glyphs_path

    name = profile.get("glyphs")
    if not name:
        return None
    if name not in _GLYPHS:
        path = glyphs_path(name)
        _GLYPHS[name] = Glyphs.load(Path(str(path))) if path.is_file() else None
    return _GLYPHS[name]


def _glyph_digits(frame_bgr, profile: dict, name: str, lit_cache: dict) -> list[int | None]:
    from film.perceive.glyphs import lit_image

    field = profile["fields"][name]
    if field["color"] not in lit_cache:
        lit_cache[field["color"]] = lit_image(frame_bgr, field["color"])
    glyphs = _glyphs_for(profile)
    return [glyphs.read(lit_cache[field["color"]], cell, f"{name}{i}") for i, cell in enumerate(_cells(field))]


def _number_from(digits: list[int | None]) -> int | None:
    if all(d is None for d in digits) or any(d == -1 for d in digits):
        return None
    value, started = 0, False
    for d in digits:
        if d is None:
            if started:
                return None
            continue
        started, value = True, value * 10 + d
    return value


def _clock_from(digits: list[int | None]) -> float | None:
    """Cells M M S S, no readable colon: both seconds digits needed, a blank minutes cell is 0."""
    if any(d == -1 for d in digits) or any(d is None for d in digits[-2:]) or digits[-2] > 5:
        return None
    d = [0 if x is None else x for x in digits]
    return float(d[0] * 600 + d[1] * 60 + d[2] * 10 + d[3])


ALT_MARGIN = 0.06       # a cell's runner-up within this of its best is kept as an alternative


def _clock_alternatives(frame_bgr, profile: dict, digits: list, lit_cache: dict) -> list[float]:
    """When a clock cell is unsure (-1), the values its two best digits would make: the minutes digit of a
    skin that hides half of it cannot tell 0 from 6 alone, but the countdown around it can (resolve_clock)."""
    glyphs = _glyphs_for(profile)
    field = profile["fields"]["clock"]
    options: list[list[int | None]] = []
    for i, (cell, d) in enumerate(zip(_cells(field), digits)):
        if d != -1:
            options.append([d])
            continue
        ranked = glyphs.rank(lit_cache[field["color"]], cell, f"clock{i}") or []
        if not ranked:
            return []
        options.append([k for k, sc in ranked[:2] if ranked[0][1] - sc <= ALT_MARGIN])
    values = []
    for combo in __import__("itertools").product(*options):
        v = _clock_from(list(combo))
        if v is not None:
            values.append(v)
    return sorted(set(values))[:4]


RESOLVE_S = 90.0


def resolve_clock(reads: list[ScoreboardRead]) -> list[ScoreboardRead]:
    """A read with alternatives takes the one the countdown allows: no higher than the last sure read and
    no lower than that read minus the seconds since (the clock runs or stops, never faster), and likewise
    against the next sure read. Run forward and backward so a resolved read helps its neighbours."""
    out = list(reads)
    for order in (range(len(out)), range(len(out) - 1, -1, -1)):
        last = None
        for i in order:
            r = out[i]
            if r.clock_s is not None:
                last = r
                continue
            if not r.clock_alts or last is None or abs(r.t - last.t) > RESOLVE_S:
                continue
            gap = abs(r.t - last.t)
            if r.t > last.t:
                fits = [v for v in r.clock_alts if last.clock_s - gap - 1 <= v <= last.clock_s]
            else:
                fits = [v for v in r.clock_alts if last.clock_s <= v <= last.clock_s + gap + 1]
            if len(fits) == 1:
                out[i] = ScoreboardRead(**{**r.to_json(), "clock_s": fits[0], "clock_alts": None})
                last = out[i]
    return out


def read_scoreboard(frame_bgr: np.ndarray, profile: dict, t: float = 0.0) -> ScoreboardRead:
    fields = profile["fields"]
    glyph_fields = set(profile.get("glyph_fields", [])) if _glyphs_for(profile) is not None else set()
    lit_cache: dict = {}
    threshold = float(profile.get("threshold", 0.3))        # LED-dot skins fill a segment zone only sparsely
    patterns = profile.get("patterns")                       # a skin's own digit shapes
    masks: dict[str, np.ndarray] = {}

    def mask_for(color: str) -> np.ndarray:
        if color not in masks:
            masks[color] = led_mask(frame_bgr, color)
        return masks[color]

    def number(name: str) -> int | None:
        field = fields[name]
        if name in glyph_fields:
            return _number_from(_glyph_digits(frame_bgr, profile, name, lit_cache))
        return decode_number(mask_for(field["color"]), _cells(field), threshold, patterns)

    home, away, period = number("home"), number("away"), number("period")
    # A skin whose clock font the segment zones cannot read leaves the clock out; the scores still give
    # made-or-missed, and quarter marks supply what the clock would have.
    clock_alts = None
    if "clock" in glyph_fields:
        digits = _glyph_digits(frame_bgr, profile, "clock", lit_cache)
        clock = _clock_from(digits)
        if clock is None:
            clock_alts = _clock_alternatives(frame_bgr, profile, digits, lit_cache) or None
    else:
        clock = read_clock(mask_for(fields["clock"]["color"]), fields["clock"], threshold, patterns) if "clock" in fields else None
    home_fouls = number("home_fouls") if "home_fouls" in fields else None
    away_fouls = number("away_fouls") if "away_fouls" in fields else None
    got = sum(v is not None for v in (home, away, clock, period))
    return ScoreboardRead(t=t, period=period, clock_s=clock, home=home, away=away,
                          home_fouls=home_fouls, away_fouls=away_fouls, conf=got / 4, clock_alts=clock_alts)


def build_timeline(source: Path, profile: dict, limit_s: float | None = None) -> list[ScoreboardRead]:
    return [read_scoreboard(frame, profile, t=t) for t, frame in iter_frames(source, fps=1.0, end=limit_s)]


REPAIR_RECENT_S = 90.0
REPAIR_NEAR_S = 10.0


def repair_clock(reads: list[ScoreboardRead]) -> list[ScoreboardRead]:
    """Reads whose seconds continue the countdown from the last trusted read but whose minute is off by
    exactly one take the minute the countdown implies. Skins whose minutes digit loses strokes read 6 as 5
    and 9 as 8; the seconds digits are sound. A read that is not within a minute of the countdown is left
    alone (a reset, a stopped clock, or a true misread for the plausibility checks to judge)."""
    out = []
    last: ScoreboardRead | None = None
    for r in sorted(reads, key=lambda x: x.t):
        fixed = r
        if r.clock_s is not None and last is not None and r.t - last.t <= REPAIR_RECENT_S:
            expected = last.clock_s - (r.t - last.t)
            off = abs(r.clock_s - expected)
            # a minute off by one within REPAIR_RECENT_S; off by more (a glyph skin's half-hidden minutes
            # digit: 4 read as 2) only right after a trusted read, where the countdown leaves no doubt
            minute_slip = off == 60.0 or (off in (120.0, 180.0, 240.0, 300.0) and r.t - last.t <= REPAIR_NEAR_S)
            if expected >= 0 and minute_slip and (r.clock_s % 60) == (expected % 60):
                fixed = ScoreboardRead(**{**r.to_json(), "clock_s": float(expected)})
        out.append(fixed)
        if fixed.clock_s is not None:
            last = fixed
    return out


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
    for read in repair_clock(resolve_clock(reads)):
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
