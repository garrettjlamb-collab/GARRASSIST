"""When the game is actually being played, read from the scoreboard clock.

Hudl's burned-in scoreboard runs a countdown for everything: pregame warmups, the breaks between
quarters, halftime, and the post-game countdown before the next game. Only the quarters start from
the regulation quarter length. Every setting of the clock near that length is a hypothesis, followed
forward with its frame locked: a read carries it on when the seconds fit a clock that ran for at most
the time elapsed, and the minutes field shows something the true minutes could look like on a
seven-segment board. The minutes digit is not trusted on its own, because on the Marcos de Niza skin
it loses strokes (6 reads 5, 9 reads 3, 8 reads 9), lags a second or three (7:51 reads 8:51) and
drops its tens digit at halftime (14:06 reads 4:06); the seconds digits are sound. A hypothesis is a
quarter when its clock moved, the reads taken while it ran mostly agree with it, and the board scored
during it; it ends at the buzzer, or where the clock was next set. A score that falls back and stays
down (the board reset for the next game) ends this game: everything after it is post-game.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from film.perceive.scoreboard import ScoreboardRead

SLACK_S = 2.0               # a read continues a countdown when its seconds fit the clock within this
ZERO_READS = 2              # the buzzer: the clock read 0:00 this many reads running
OVER_CONFIRM_S = 120.0      # a score that falls and does not come back within this ended the game
# A seven-segment digit misreads by losing strokes, so a true digit can only read as one drawn with a
# subset of its segments: 8 as anything, 9 as 3 or 5, 6 as 5, 0 as 7. The minutes digit also lags a
# second or three behind the clock (7:51 reads 8:51) and its tens digit drops out (14:06 reads 4:06).
# All three were counted over five Marcos de Niza games on 2026-10-12. A clock set anew is none of
# these: 0:00 cannot read 8:00, nor 1:00 read 8:00.
SEGMENTS = {0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg", 5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg"}
SUBDIGITS = {m: {r for r in SEGMENTS if set(SEGMENTS[r]) <= set(SEGMENTS[m])} for m in SEGMENTS}


def plausible_minutes(true_minutes: int) -> set[int]:
    """What the minutes field can show when the clock stands at `true_minutes`."""
    tens, units = divmod(true_minutes, 10)
    out = {true_minutes, true_minutes + 1}
    out |= {tens * 10 + r for r in SUBDIGITS[units]}
    if tens:
        out |= {r for r in SUBDIGITS[units]}                           # the tens digit dropped out
    return out
RESUME_S = 150.0            # a countdown interrupted by a timeout clock or unreadable reads resumes within this
MIN_QUARTER_S = 120.0       # starting values below this are breaks, never quarters
MIN_BREAK_S = 60.0          # ... and a whole minute at least this long is some clock being set (the 1:00 break)
LIVE_TOLERANCE_S = 2.0      # a countdown is a quarter when it starts within this of the quarter length
KNOWN_TOLERANCE_S = 20.0    # ... or this much short of it, when the profile states the length and reads have gaps
OVERTIME_FRACTION = 0.5     # overtime is this much of a quarter (NFHS: four minutes)
END_GRACE_S = 3.0           # a quarter stays live this long after its buzzer (buzzer-beaters)
END_STRETCH = 3.0           # reads that stop with time on the clock: it ran out within this many times that


SCORE_LEAD_S = 90.0         # with no clock: live play starts this long before the first score change
SCORE_TAIL_S = 120.0        # ... and ends this long after the last one
BREAK_GAP_S = 240.0         # a stretch this long with no score change is a break (halftime), not play
MIN_CHANGES = 2             # a run of play has at least this many score changes


@dataclass
class GamePhases:
    quarter_length_s: float | None
    live: list[tuple[float, float]] = field(default_factory=list)      # [start, end) of each quarter
    known: bool = False                                              # False: no usable clock, treat all as live
    game_over_t: float | None = None                                 # the board was reset for the next game here
    quarters: dict[int, tuple[float, float]] = field(default_factory=dict)   # period number -> its live span


def _phases_from_scores(reads: list[ScoreboardRead]) -> GamePhases:
    """A board that shows scores and the period but no clock: live play runs from just before the first
    score change to just after the last, and the period says which quarter."""
    changes = []
    last_total = None
    for r in reads:
        if r.home is None or r.away is None:
            continue
        total = r.home + r.away
        if last_total is not None and total > last_total:
            changes.append(r.t)
        last_total = total
    if len(changes) < MIN_CHANGES:
        return GamePhases(quarter_length_s=None, known=False, game_over_t=_game_over_t(reads))
    period_at = {r.t: r.period for r in reads}
    # Runs of scoring: a new run starts at a long gap or when the board's period changes. The period digit
    # is not trusted for the quarter number (it misreads during breaks); runs are numbered in order.
    runs: list[list[float]] = [[changes[0]]]
    for prev, t in zip(changes, changes[1:]):
        if t - prev > BREAK_GAP_S or period_at.get(t) != period_at.get(prev):
            runs.append([t])
        else:
            runs[-1].append(t)
    runs = [r for r in runs if len(r) >= MIN_CHANGES]
    spans: list[tuple[float, float]] = []
    for r in runs:
        lo, hi = r[0] - SCORE_LEAD_S, r[-1] + SCORE_TAIL_S
        if spans and lo < spans[-1][1]:
            lo = spans[-1][1]
        spans.append((lo, hi))
    quarters = {i + 1: span for i, span in enumerate(spans)}
    return GamePhases(quarter_length_s=None, live=spans, known=True, game_over_t=_game_over_t(reads), quarters=quarters)


GLITCH_S = 5.0              # a clock read this far from the median of its neighbours is a misread, not a reset


def _deglitched(reads: list[ScoreboardRead]) -> list[ScoreboardRead]:
    """Clock reads with lone outliers dropped: a read is kept when it sits within GLITCH_S of the median of
    the five reads around it. A real reset moves every following read, so it survives."""
    clocked = [r for r in reads if r.clock_s is not None]
    kept = []
    for i, r in enumerate(clocked):
        window = sorted(x.clock_s for x in clocked[max(0, i - 2):i + 3])
        median = window[len(window) // 2]
        if abs(r.clock_s - median) <= GLITCH_S:
            kept.append(r)
    return kept


def _continues(anchor_t: float, anchor_v: float, t: float, raw: float, offset: float = 0.0) -> float | None:
    """The clock value that carries the countdown on from the anchor, if the read's seconds allow one.

    Between the anchor and this read the clock ran for at most the time elapsed and may have stopped,
    so the value lies in [anchor - elapsed, anchor] give or take SLACK_S. The minutes are whatever makes
    the read's seconds land in that window, as long as the board could show them that way. After
    a long gap more than one can: the one that keeps the countdown's `offset` (raw minus value, the
    misread in force at the anchor) wins, so a countdown's frame never drifts; else the nearest to the raw."""
    lo, hi = max(0.0, anchor_v - (t - anchor_t) - SLACK_S), anchor_v + SLACK_S
    sec, raw_m = raw % 60.0, math.floor(raw / 60.0)
    m_lo, m_hi = math.ceil((lo - sec) / 60.0), math.floor((hi - sec) / 60.0)
    candidates = [m * 60.0 + sec for m in range(m_lo, m_hi + 1) if raw_m in plausible_minutes(m)]
    if not candidates:
        return None
    return min(candidates, key=lambda v: (abs((raw - v) - offset) >= 0.5, abs(v - raw)))


@dataclass
class _Hypothesis:
    """One setting of the clock near the quarter length, followed forward with its frame locked.

    `accepted` holds (t, value, raw) for every read that carries the countdown on; a read that does not
    fit (a timeout clock laid over the game clock, a misread) is skipped and the hypothesis dies once
    nothing has fitted for RESUME_S."""
    first_t: float
    value0: float
    accepted: list[tuple[float, float, float]]

    @property
    def last_t(self) -> float:
        return self.accepted[-1][0]

    @property
    def start_t(self) -> float | None:
        """When the clock first moved: the read before the first value below the setting. None if never."""
        for (t_prev, _, _), (_, v, _) in zip(self.accepted, self.accepted[1:]):
            if v < self.value0 - 0.5:
                return t_prev
        return None

    @property
    def zero_t(self) -> float | None:
        rows = self.accepted
        for i in range(len(rows) - ZERO_READS + 1):
            if all(v < 0.5 for _, v, _ in rows[i:i + ZERO_READS]):
                return rows[i][0]
        return None

    @property
    def end_t(self) -> float:
        zero = self.zero_t
        return zero if zero is not None else self.last_t

    @property
    def agrees(self) -> bool:
        """The frame is right when the reads taken while the clock ran mostly say what it says."""
        running = [round((raw - v) / 60.0) for (_, v0, _), (_, v, raw) in zip(self.accepted, self.accepted[1:]) if v < v0 - 0.5]
        return bool(running) and Counter(running).most_common(1)[0][0] == 0


def _is_jump(prev: ScoreboardRead | None, r: ScoreboardRead) -> bool:
    """The clock was set: this read does not carry on from the one before it, however long the gap
    (a clock that ran unread through a gap still reads lower, never higher)."""
    if prev is None:
        return True
    return not (prev.clock_s - (r.t - prev.t) - SLACK_S <= r.clock_s <= prev.clock_s + SLACK_S)


def _is_setting(prev: ScoreboardRead | None, r: ScoreboardRead, at_least: float = MIN_QUARTER_S) -> bool:
    """A game or break clock being set: a whole minute long enough to be one, out of the blue."""
    return r.clock_s % 60.0 == 0 and r.clock_s >= at_least and _is_jump(prev, r)


def _track(clocked: list[ScoreboardRead], start: int, value0: float) -> _Hypothesis:
    r0 = clocked[start]
    hyp = _Hypothesis(first_t=r0.t, value0=value0, accepted=[(r0.t, value0, r0.clock_s)])
    anchor_t, anchor_v, offset = r0.t, value0, r0.clock_s - value0
    for i in range(start + 1, len(clocked)):
        r = clocked[i]
        if r.t - anchor_t > RESUME_S:
            break
        value = _continues(anchor_t, anchor_v, r.t, r.clock_s, offset)
        if value is None:
            if _is_setting(clocked[i - 1], r):
                break                                                 # the clock was set anew: this countdown is over
            continue
        anchor_t, anchor_v, offset = r.t, value, r.clock_s - value
        hyp.accepted.append((r.t, value, r.clock_s))
        if len(hyp.accepted) >= ZERO_READS and all(v < 0.5 for _, v, _ in hyp.accepted[-ZERO_READS:]):
            break                                                     # the buzzer
    return hyp


def _settings(clocked: list[ScoreboardRead], at_least: float = MIN_QUARTER_S) -> list[tuple[int, float]]:
    """(index, raw) of every read that starts a clock: a whole minute that does not carry on from the read before."""
    return [(i, r.clock_s) for i, r in enumerate(clocked) if _is_setting(clocked[i - 1] if i else None, r, at_least)]


def _hypotheses(clocked: list[ScoreboardRead], length: float, tolerance: float) -> list[_Hypothesis]:
    """Every clock setting that could be the quarter length, tracked forward. A whole minute whose
    minutes digit could be a misread of the length's is taken to be the length; a read a little short
    of it (the first read after a gap) is taken as it is."""
    out = []
    minute = length // 60 if length % 60.0 == 0 else None
    for i, r in enumerate(clocked):
        if not _is_jump(clocked[i - 1] if i else None, r):
            continue
        raw = r.clock_s
        if minute is not None and raw % 60.0 == 0 and int(raw // 60) in plausible_minutes(int(minute)):
            out.append(_track(clocked, i, length))
        elif length - tolerance <= raw <= length + SLACK_S:
            out.append(_track(clocked, i, raw))
    return out


def _period_mode(reads: list[ScoreboardRead], start: float, end: float) -> int | None:
    periods = Counter(r.period for r in reads if start <= r.t < end and r.period is not None)
    return periods.most_common(1)[0][0] if periods else None


def _game_over_t(reads: list[ScoreboardRead]) -> float | None:
    """The first time the combined score falls and stays down: the board was reset for the next game.
    A fall that comes back within OVER_CONFIRM_S is a misread digit."""
    scored = [(r.t, r.home + r.away) for r in reads if r.home is not None and r.away is not None]
    high = None
    for i, (t, total) in enumerate(scored):
        if high is not None and total < high:
            if not any(later >= high for lt, later in scored[i + 1:] if lt - t <= OVER_CONFIRM_S):
                return t
            continue
        high = total if high is None else max(high, total)
    return None


def _scored_within(reads: list[ScoreboardRead], start: float, end: float) -> bool:
    last = None
    for r in reads:
        if r.t < start or r.home is None or r.away is None:
            continue
        if r.t >= end:
            break
        total = r.home + r.away
        if last is not None and total != last:
            return True
        last = total
    return False


def game_phases(timeline: list[ScoreboardRead], quarter_length_s: float | None = None) -> GamePhases:
    """`quarter_length_s`, when the camera profile states it, beats the commonest clock setting (a board
    that counts down 9:00 over and over before the game would otherwise set the quarter to 9:00); reads
    with gaps may start a quarter a little short; a quarter must see some scoring when the board shows
    scores and the length is known; and after four quarters a clock set to half the length is overtime."""
    reads = sorted(timeline, key=lambda r: r.t)
    clocked = _deglitched(reads)
    settings = [raw for _, raw in _settings(clocked) if raw >= MIN_QUARTER_S]
    if not settings:
        if reads and all(r.clock_s is None for r in reads):
            return _phases_from_scores(reads)
        return GamePhases(quarter_length_s=None, known=False, game_over_t=_game_over_t(reads))
    known_length = quarter_length_s is not None
    quarter = float(quarter_length_s) if known_length else float(Counter(settings).most_common(1)[0][0])
    tolerance = KNOWN_TOLERANCE_S if known_length else LIVE_TOLERANCE_S
    any_scores = any(r.home is not None and r.away is not None for r in reads)
    over = _game_over_t(reads)
    candidates = [(h, False) for h in _hypotheses(clocked, quarter, tolerance)]
    if known_length:
        candidates += [(h, True) for h in _hypotheses(clocked, quarter * OVERTIME_FRACTION, tolerance)]
    set_times = [clocked[i].t for i, _ in _settings(clocked, MIN_BREAK_S)]
    live: list[tuple[float, float]] = []
    firm: list[float] = []                                            # each quarter's buzzer or last read
    periods: list[int | None] = []
    for hyp, overtime in sorted(candidates, key=lambda c: (c[0].start_t if c[0].start_t is not None else c[0].first_t, c[1])):
        start = hyp.start_t
        if start is None or not hyp.agrees:
            continue
        if overtime and len(live) < 4:
            continue
        end = hyp.end_t
        if hyp.zero_t is None:
            # The reads stopped with time on the clock (unreadable board, or a buzzer nobody read): the
            # clock ran out within a few times that, and never later than the next setting of a clock.
            cap = next((t for t in set_times if t > hyp.last_t), math.inf)
            end = min(hyp.last_t + hyp.accepted[-1][1] * END_STRETCH, cap)
        if over is not None:
            if start >= over:
                continue
            end = min(end, over)
        if live and start < live[-1][1]:
            if start < firm[-1]:
                continue                                              # the same clock, seen from an earlier setting
            live[-1] = (live[-1][0], start)                           # the last quarter ran on only until this clock was set
        if known_length and any_scores and not _scored_within(reads, start, end):
            continue                                                  # a dead countdown, not a quarter
        live.append((start, end if over is not None and end >= over else end + END_GRACE_S))
        firm.append(hyp.end_t)
        periods.append(_period_mode(reads, start, end))
    numbers = periods if all(p is not None for p in periods) and periods == sorted(set(periods)) else list(range(1, len(live) + 1))
    return GamePhases(quarter_length_s=float(quarter), live=live, known=True, game_over_t=over, quarters=dict(zip(numbers, live)))


def is_live(phases: GamePhases, t: float) -> bool:
    if not phases.known:
        return True
    return any(start <= t < end for start, end in phases.live)


def is_this_game(phases: GamePhases, t: float) -> bool:
    """False only after the scoreboard was reset for the next game."""
    return phases.game_over_t is None or t < phases.game_over_t
