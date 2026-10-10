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
END_GRACE_S = 3.0           # a quarter stays live this long after its clock is reset (buzzer-beaters)


SCORE_LEAD_S = 90.0         # with no clock: live play starts this long before the first score change
SCORE_TAIL_S = 120.0        # ... and ends this long after the last one


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
    if len(changes) < 2:
        return GamePhases(quarter_length_s=None, known=False, game_over_t=_game_over_t(reads))
    start, end = changes[0] - SCORE_LEAD_S, changes[-1] + SCORE_TAIL_S
    spans: dict[int, list[float]] = {}
    for r in reads:
        if r.period is not None and start <= r.t <= end:
            spans.setdefault(r.period, [r.t, r.t])[1] = r.t
    quarters = {p: (lo, hi) for p, (lo, hi) in sorted(spans.items())}
    return GamePhases(quarter_length_s=None, live=list(quarters.values()), known=True,
                      game_over_t=_game_over_t(reads), quarters=quarters)


def _stretches(reads: list[ScoreboardRead]) -> list[tuple[float, float]]:
    """(start_t, starting clock) for each stretch between upward clock resets."""
    out: list[tuple[float, float]] = []
    last_clock: float | None = None
    start_t: float | None = None
    best: float | None = None
    for r in reads:
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


def game_phases(timeline: list[ScoreboardRead]) -> GamePhases:
    reads = sorted(timeline, key=lambda r: r.t)
    stretches = _stretches(reads)
    long_starts = [round(c) for _, c in stretches if c >= MIN_QUARTER_S]
    if not long_starts:
        if reads and all(r.clock_s is None for r in reads):
            return _phases_from_scores(reads)
        return GamePhases(quarter_length_s=None, known=False, game_over_t=_game_over_t(reads))
    quarter = Counter(long_starts).most_common(1)[0][0]
    over = _game_over_t(reads)
    end_of_time = reads[-1].t + 1.0
    live: list[tuple[float, float]] = []
    for i, (start, clock) in enumerate(stretches):
        end = stretches[i + 1][0] if i + 1 < len(stretches) else end_of_time
        if over is not None:
            if start >= over:
                continue
            end = min(end, over)
        if abs(clock - quarter) <= LIVE_TOLERANCE_S:
            live.append((start, end if over is not None and end >= over else end + END_GRACE_S))
    return GamePhases(quarter_length_s=float(quarter), live=live, known=True, game_over_t=over)


def is_live(phases: GamePhases, t: float) -> bool:
    if not phases.known:
        return True
    return any(start <= t < end for start, end in phases.live)


def is_this_game(phases: GamePhases, t: float) -> bool:
    """False only after the scoreboard was reset for the next game."""
    return phases.game_over_t is None or t < phases.game_over_t
