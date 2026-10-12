"""When the game is actually being played, read from the scoreboard clock.

Hudl's burned-in scoreboard runs a countdown for everything: pregame warmups, the breaks between
quarters, halftime, and the post-game countdown before the next game. Only the quarters start from
the regulation quarter length, so the timeline is split wherever the clock is reset upward, each
stretch is labelled by the clock value it started from, and the most common long starting value is
taken as the quarter length. A score that falls back (the board reset for the next game) ends this
game: everything after it is post-game.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from film.perceive.scoreboard import ScoreboardRead

RESET_JUMP_S = 2.0          # a clock rise bigger than this starts a new stretch
START_WINDOW_S = 10.0       # a stretch's starting clock is the highest clock in its first seconds
MIN_QUARTER_S = 120.0       # starting values below this are breaks, never quarters
LIVE_TOLERANCE_S = 2.0      # a stretch is a quarter when it starts within this of the quarter length
KNOWN_TOLERANCE_S = 20.0    # ... or within this, when the profile states the quarter length and reads have gaps
END_GRACE_S = 3.0           # a quarter stays live this long after its clock is reset (buzzer-beaters)


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
    """A board that shows scores and the period but no clock (the Marcos de Niza skin): live play runs
    from just before the first score change to just after the last, and the period says which quarter."""
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


def _stretches(reads: list[ScoreboardRead]) -> list[tuple[float, float]]:
    """(start_t, starting clock) for each stretch between upward clock resets."""
    out: list[tuple[float, float]] = []
    last_clock: float | None = None
    start_t: float | None = None
    best: float | None = None
    for r in _deglitched(reads):
        if r.clock_s is None:
            continue
        if last_clock is None or r.clock_s > last_clock + RESET_JUMP_S:
            if start_t is not None and best is not None:
                out.append((start_t, best))
            start_t, best = r.t, r.clock_s
        elif start_t is not None and r.t - start_t <= START_WINDOW_S:
            best = max(best, r.clock_s)
        last_clock = r.clock_s
    if start_t is not None and best is not None:
        out.append((start_t, best))
    return out


def _game_over_t(reads: list[ScoreboardRead]) -> float | None:
    """The first time the combined score falls: the board was reset for the next game."""
    prev_total: int | None = None
    for r in reads:
        if r.home is None or r.away is None:
            continue
        total = r.home + r.away
        if prev_total is not None and total < prev_total:
            return r.t
        prev_total = total
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
    """`quarter_length_s`, when the camera profile states it, beats the commonest long start (a board that
    counts down 9:00 over and over before the game would otherwise set the quarter to 9:00), reads with gaps
    may start a quarter a little short, and a quarter must see some scoring."""
    reads = sorted(timeline, key=lambda r: r.t)
    stretches = _stretches(reads)
    long_starts = [round(c) for _, c in stretches if c >= MIN_QUARTER_S]
    if not long_starts:
        if reads and all(r.clock_s is None for r in reads):
            return _phases_from_scores(reads)
        return GamePhases(quarter_length_s=None, known=False, game_over_t=_game_over_t(reads))
    known_length = quarter_length_s is not None
    quarter = float(quarter_length_s) if known_length else Counter(long_starts).most_common(1)[0][0]
    tolerance = KNOWN_TOLERANCE_S if known_length else LIVE_TOLERANCE_S
    any_scores = any(r.home is not None and r.away is not None for r in reads)
    over = _game_over_t(reads)
    end_of_time = reads[-1].t + 1.0
    live: list[tuple[float, float]] = []
    for i, (start, clock) in enumerate(stretches):
        end = stretches[i + 1][0] if i + 1 < len(stretches) else end_of_time
        if over is not None:
            if start >= over:
                continue
            end = min(end, over)
        if quarter - tolerance <= clock <= quarter + LIVE_TOLERANCE_S:
            if known_length and any_scores and not _scored_within(reads, start, end):
                continue                                              # a dead countdown, not a quarter
            live.append((start, end if over is not None and end >= over else end + END_GRACE_S))
    return GamePhases(quarter_length_s=float(quarter), live=live, known=True, game_over_t=over)


def is_live(phases: GamePhases, t: float) -> bool:
    if not phases.known:
        return True
    return any(start <= t < end for start, end in phases.live)


def is_this_game(phases: GamePhases, t: float) -> bool:
    """False only after the scoreboard was reset for the next game."""
    return phases.game_over_t is None or t < phases.game_over_t
