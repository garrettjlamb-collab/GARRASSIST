"""A film with no scoreboard, recorded on a camera that stopped at every break: put the live stat book on it.

The Mingus Union film (2026-11-18) is a sideline camera with no overlay, paused at timeouts and quarter
breaks: 65 minutes of film for 85 minutes of game. No clock to read, and the shot cards on a camera like
that are too few to align on. Two things are left, and they are enough:

1. **Where the camera stopped.** A restart is a jump between one frame and the next far larger than the
   frames around it (`find_cuts`). Between two jumps the film runs in real time, so a stretch of film is
   the stat book's clock plus one offset.
2. **Where the camera points.** It turns to follow the ball, so its left-right swing says which basket the
   play is at, and the stat book says which team shot, and each team attacks one basket a half. For each
   stretch, the offset that puts our shots at one end and theirs at the other is the one
   (`fit_offsets`). The ends swap at halftime, which the fit requires.

Offsets only fall from one stretch to the next (the camera skips time, never repeats it), and the fit
takes the best monotone set by dynamic programming, so a short stretch with three shots cannot land
somewhere the stretches either side of it rule out.

Pure: numbers in, numbers out. `film camstops` reads the film and writes `camstops.json`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CUT_RATIO = 6.0          # a restart's frame jump is this many times the jumps around it ...
CUT_MIN = 12.0           # ... and at least this big (mean grey difference, 0-255, at 96x54)
CUT_NEIGHBOURS = 5       # frames either side that set "around it"
MIN_STRETCH_S = 20.0     # two jumps closer than this are one restart
PAN_STEP_CAP = 60.0      # a frame-to-frame shift past this (px at 240 wide) is a cut or a blur, not a pan
PAN_WINDOW_S = 90.0      # the camera's slow drift, removed before reading which end it faces
LOG_LAG_S = 3.0          # he taps a shot in about this long after it happens
MIN_SHOTS = 3            # a stretch with fewer logged shots has no say in its own offset
GAP_COST_PER_S = 0.0005  # a camera stop costs this per second, so an unevidenced stretch takes the shortest stop
OFFSET_RANGE_S = (-3000.0, 600.0)


@dataclass
class Stretch:
    start: float          # film seconds
    end: float
    offset: float         # film time = logged seconds (from the first play) + offset
    shots: int            # logged field-goal attempts that land in it
    score: float          # mean of side x camera position over those shots; ~0 is no evidence


def find_cuts(diffs: list[tuple[float, float]], ratio: float = CUT_RATIO, floor: float = CUT_MIN,
              k: int = CUT_NEIGHBOURS, min_gap: float = MIN_STRETCH_S) -> list[float]:
    """Film times where the camera restarted, from (t, mean frame difference) pairs."""
    v = [x for _, x in diffs]
    cuts: list[float] = []
    for i in range(k, len(diffs) - k):
        t, x = diffs[i]
        around = sorted(v[i - k:i] + v[i + 1:i + k + 1])[k]
        if x > ratio * max(around, 1.0) and x > floor:
            if cuts and t - cuts[-1] < min_gap:
                continue
            cuts.append(t)
    return cuts


def camera_position(pan: list[tuple[float, float]], bounds: list[float], rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Which way the camera faces over time, standardised within each stretch: (t, position).

    `pan` is (t, horizontal shift since the previous sample). Summed within a stretch it is the camera's
    heading; the drift over `PAN_WINDOW_S` comes out, leaving the swing from one end to the other."""
    t = np.array([p[0] for p in pan], dtype=float)
    dx = np.array([p[1] for p in pan], dtype=float)
    pos = np.zeros_like(t)
    k = max(1, int(PAN_WINDOW_S * rate))
    for a, b in zip(bounds, bounds[1:]):
        m = (t > a + 0.5) & (t < b)
        if m.sum() < 2:
            continue
        p = np.cumsum(np.where(np.abs(dx[m]) < PAN_STEP_CAP, dx[m], 0.0))
        pad = np.pad(p, (k // 2, k - k // 2 - 1), mode="edge")
        q = p - np.convolve(pad, np.ones(k) / k, "valid")[: len(p)]
        pos[m] = q / (q.std() + 1e-6)
    return t, pos


def fit_offsets(bounds: list[float], cam_t: np.ndarray, cam_pos: np.ndarray,
                shots: list[tuple[float, int, int]], step: float = 1.0) -> list[Stretch]:
    """One offset per stretch of film. `shots` is (logged seconds, +1 us / -1 them, period).

    For a sign s (which end we attack in the first half; the second half is the other end) and an offset d,
    a stretch scores sum(side * half_sign * position at the shot's film time). Offsets must not rise from
    one stretch to the next; the best monotone path wins, both signs tried."""
    grid = np.arange(OFFSET_RANGE_S[0], OFFSET_RANGE_S[1] + step, step)
    secs = np.array([s for s, _, _ in shots], dtype=float)
    sides = np.array([side for _, side, _ in shots], dtype=float)
    half = np.array([1.0 if (p or 1) <= 2 else -1.0 for _, _, p in shots])
    stretches = list(zip(bounds, bounds[1:]))
    best_path, best_total = None, -np.inf
    for sign in (1.0, -1.0):
        scores = np.zeros((len(stretches), len(grid)))
        counts = np.zeros_like(scores, dtype=int)
        for i, (a, b) in enumerate(stretches):
            for j, d in enumerate(grid):
                ft = secs + d - LOG_LAG_S
                m = (ft >= a + 3) & (ft <= b - 3)
                n = int(m.sum())
                counts[i, j] = n
                if n >= MIN_SHOTS:
                    pos = np.interp(ft[m], cam_t, cam_pos)
                    scores[i, j] = float(np.sum(sign * half[m] * sides[m] * pos))
        # dynamic programming: offset index never rises from stretch i-1 to stretch i
        total = scores[0].copy()
        back = np.zeros_like(scores, dtype=int)
        for i in range(1, len(stretches)):
            # best previous offset at or above each one, less the stop's cost: a suffix max of total - c*d
            v = total - GAP_COST_PER_S * grid
            arg = np.empty(len(grid), dtype=int)
            run = len(grid) - 1
            for j in range(len(grid) - 1, -1, -1):
                if v[j] > v[run]:
                    run = j
                arg[j] = run
            total = v[arg] + GAP_COST_PER_S * grid + scores[i]
            back[i] = arg
        j = int(np.argmax(total))
        if total[j] > best_total:
            path = [j]
            for i in range(len(stretches) - 1, 0, -1):
                j = back[i, j]
                path.append(j)
            path.reverse()
            best_total = total[path[-1]]
            best_path = (path, scores, counts)
    path, scores, counts = best_path
    out = []
    for i, ((a, b), j) in enumerate(zip(stretches, path)):
        n = int(counts[i, j])
        out.append(Stretch(start=round(a, 2), end=round(b, 2), offset=float(grid[j]), shots=n,
                           score=round(float(scores[i, j]) / n, 2) if n else 0.0))
    return out


def film_time(stretches: list[Stretch], logged_s: float, slack: float = 2.0) -> tuple[float | None, int | None]:
    """A logged moment's film time and stretch index, or (None, None) when the camera was off."""
    for i, s in enumerate(stretches):
        t = logged_s + s.offset
        if s.start - slack <= t <= s.end + slack:
            return min(max(t, s.start), s.end), i
    return None, None
