"""Where a shot was taken from, from what the detectors saw in the seconds before it reached the rim. The camera turns,
so every ball sighting is first carried into one frame's pixels (the frames' court mappings give the turn). A flying
ball then draws a parabola in time: the earliest sighting on it is the release, the person holding it there is the
shooter, and the shooter's feet through the court mapping are the spot."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from film.interpret.court import FT_LINE_Y, WIDTH, Mapping, fold

REFERENCE_WIDTH = 1920                     # the pixel distances below are for film this wide
STATIC_PX, STATIC_MIN_FRAMES = 12.0, 6     # a "ball" that stays put in the gym for this many frames is not the ball
RIM_REACH_PX = 130.0                       # a flight ends this close to a rim
SEED_BEFORE_S, SEED_AFTER_S = 1.8, 0.3     # and this close to the moment the ball reached it
MAX_GAP_S = 0.4                            # the ball may go unseen this long in flight
MAX_BALL_PX_PER_S = 2200.0
GRAVITY_PX = 900.0                         # about 32 ft/s^2 at about 28 px/ft
FLIGHT_TOL_PX, FLIGHT_TOL_PX_PER_S = 25.0, 150.0
BEAM = 24
RISE_PX = 40.0
SAME_FRAME_S = 0.02
FLIGHT_MIN_POINTS = 3
BACK_S, BACK_STEP_S = 0.8, 0.1             # how far the flight is run back unseen to reach the shooter's hands
COURT_MARGIN_FT = 3.0
REACH_WIDTHS = 1.5                         # further than this from anyone, the shooter is flagged as doubtful
ARM_HEIGHTS, ARM_WIDTHS = 0.45, 0.25       # raised arms reach this far above and beside a person's box
SAME_PERSON_IOU = 0.3
NEIGHBOUR_FRAMES = 2                       # a shooter hidden behind someone at the release is looked for this many frames either side


@dataclass(frozen=True)
class Look:
    """One frame: the time, the mapping there, and the detector's boxes as (x1, y1, x2, y2, conf)."""
    t: float
    mapping: Mapping | None
    people: list[tuple[float, float, float, float, float]]
    balls: list[tuple[float, float, float, float, float]]


@dataclass
class Spot:
    x: float | None = None
    y: float | None = None
    release_t: float | None = None
    method: str = "none"
    arc_points: int = 0
    arc_span_s: float = 0.0
    reach_px: float | None = None
    flags: list = field(default_factory=list)
    box: tuple | None = None


def _carry(H_from: np.ndarray, H_to_inv: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """A pixel in one frame to the pixel showing the same place in another: through the court plane, which is right
    for any point when the camera only turns."""
    v = H_to_inv @ (H_from @ np.array([x, y, 1.0]))
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def ball_sightings(looks: list[Look], scale: float = 1.0) -> list[tuple[float, float, float, float]]:
    """(t, x, y, conf) of every ball box in one frame's pixels (the last frame with a mapping), less anything that stays
    put in the gym."""
    anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
    if anchor is None:
        return []
    inv = np.linalg.inv(anchor.H)
    seen = []
    for k in looks:
        if k.mapping is None:
            continue
        for x1, y1, x2, y2, conf in k.balls:
            p = _carry(k.mapping.H, inv, (x1 + x2) / 2, (y1 + y2) / 2)
            if p is not None:
                seen.append((k.t, p[0], p[1], conf))
    keep = []
    for s in seen:
        near = {o[0] for o in seen if np.hypot(o[1] - s[1], o[2] - s[2]) <= STATIC_PX * scale}
        if len(near) < STATIC_MIN_FRAMES:
            keep.append(s)
    return keep


def _flight_rank(chain: list, scale: float) -> tuple[bool, float, int]:
    """A shot goes up before it comes down: a ball that only falls is the ball through the net or off the rim."""
    peak = min(p[2] for p in chain)
    return (len(chain) >= FLIGHT_MIN_POINTS and chain[0][2] - peak >= RISE_PX * scale, chain[-1][0] - chain[0][0], len(chain))


def find_flight(sightings: list[tuple[float, float, float, float]], rims_px: list[tuple[float, float]],
                end_t: float | None = None, scale: float = 1.0) -> list[tuple[float, float, float, float]]:
    """The ball's flight traced backward from a rim: starting from a sighting close to a rim, each earlier sighting must
    be where the flight so far says the ball was. The longest such flight, earliest first."""
    pts = sorted(sightings)
    seeds = [p for p in pts if any(np.hypot(p[1] - rx, p[2] - ry) <= RIM_REACH_PX * scale for rx, ry in rims_px)
             and (end_t is None or end_t - SEED_BEFORE_S <= p[0] <= end_t + SEED_AFTER_S)]
    best: list = []
    best_rank = (False, -1.0, 0)
    beam = [[s] for s in seeds]
    while beam:
        grown = []
        for chain in beam:
            head = chain[0]                  # the earliest sighting so far
            v = None
            if len(chain) >= 2:
                dt = chain[1][0] - head[0]
                v = ((chain[1][1] - head[1]) / dt, (chain[1][2] - head[2]) / dt)
            fits = []
            for p in pts:
                dt = head[0] - p[0]
                if dt <= 1e-6 or dt > MAX_GAP_S:
                    continue
                gap = np.hypot(head[1] - p[1], head[2] - p[2])
                if gap > MAX_BALL_PX_PER_S * scale * dt:
                    continue
                if v is not None:
                    px = head[1] - v[0] * dt
                    py = head[2] - v[1] * dt + 0.5 * GRAVITY_PX * scale * dt * dt
                    if np.hypot(px - p[1], py - p[2]) > (FLIGHT_TOL_PX + FLIGHT_TOL_PX_PER_S * dt) * scale:
                        continue
                fits.append((dt, p))
            nearest = min((dt for dt, _ in fits), default=0.0)
            for dt, p in fits:                   # the next sighting back, not one further on with this one skipped
                if dt <= nearest + SAME_FRAME_S:
                    grown.append([p, *chain])
            found = bool(fits)
            if not found and _flight_rank(chain, scale) > best_rank:
                best, best_rank = chain, _flight_rank(chain, scale)
        grown.sort(key=lambda c: (-(c[-1][0] - c[0][0]), -len(c)))
        beam = grown[:BEAM]
    return best


def _on_court(box, mapping: Mapping) -> bool:
    x1, y1, x2, y2, _ = box
    return mapping.contains((x1 + x2) / 2, y2, margin=COURT_MARGIN_FT)


def _reach(box, x: float, y: float) -> float:
    """How far a point is from where a person's hands can be, in pixels: 0 inside their box or in the space above it
    that raised arms reach."""
    x1, y1, x2, y2, _ = box
    w, h = x2 - x1, y2 - y1
    dx = max(x1 - ARM_WIDTHS * w - x, 0.0, x - x2 - ARM_WIDTHS * w)
    dy = max(y1 - ARM_HEIGHTS * h - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def shooter_at(look: Look, ball_px: tuple[float, float]) -> tuple[tuple, float] | None:
    """The on-court person the ball is nearest to, with how far: a person whose box holds the ball ties to the one whose
    head is closest to it."""
    if look.mapping is None:
        return None
    people = [b for b in look.people if _on_court(b, look.mapping)]
    if not people:
        return None
    def key(b):
        return (_reach(b, *ball_px), np.hypot((b[0] + b[2]) / 2 - ball_px[0], b[1] - ball_px[1]))
    best = min(people, key=key)
    return best, _reach(best, *ball_px)


def locate(looks: list[Look], rims_px: list[tuple[float, float]], arrival_t: float | None = None, scale: float = 1.0) -> Spot:
    """The spot from the ball's flight traced back from the rim."""
    anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
    flight = find_flight(ball_sightings(looks, scale), rims_px, end_t=arrival_t, scale=scale) if anchor is not None else []
    if len(flight) < FLIGHT_MIN_POINTS or anchor is None:
        return Spot(flags=["no-flight"], arc_points=len(flight))
    first, second = flight[0], flight[1]
    dt = second[0] - first[0]
    vx, vy = (second[1] - first[1]) / dt, (second[2] - first[2]) / dt
    done: tuple | None = None
    order = {id(k): i for i, k in enumerate(looks)}
    for back in np.arange(0.0, BACK_S + 1e-6, BACK_STEP_S):
        t = first[0] - back
        centre = min(looks, key=lambda k: abs(k.t - t))
        if abs(centre.t - t) > BACK_STEP_S:
            continue
        here = (first[1] - vx * back, first[2] - vy * back + 0.5 * GRAVITY_PX * scale * back * back)
        i = order[id(centre)]
        for look in sorted(looks[max(0, i - NEIGHBOUR_FRAMES): i + NEIGHBOUR_FRAMES + 1], key=lambda k: abs(k.t - t)):
            if look.mapping is None:
                continue
            ball = _carry(anchor.H, np.linalg.inv(look.mapping.H), *here)
            hit = shooter_at(look, ball) if ball else None
            if hit is None:
                continue
            box, reach = hit
            if done is None or reach < done[1]:
                done = (box, reach, look, look.t)
            if reach == 0.0:
                break
        if done is not None and done[1] == 0.0:
            break
    if done is None:
        return Spot(flags=["no-shooter"], arc_points=len(flight))
    box, reach, look, t = done
    at = look.mapping.to_court((box[0] + box[2]) / 2, box[3])
    if at is None:
        return Spot(flags=["no-shooter"], arc_points=len(flight))
    flags = [] if reach <= REACH_WIDTHS * (box[2] - box[0]) else ["shooter-far-from-ball"]
    return Spot(x=at[0], y=at[1], release_t=t, method="flight", arc_points=len(flight),
                arc_span_s=round(flight[-1][0] - first[0], 2), reach_px=round(reach, 1), flags=flags, box=tuple(round(float(v), 1) for v in box[:4]))


FT_NEAR_FT = 3.5                           # the shooter stands this close to the middle of the free-throw line
FT_STILL_SHARE = 0.8
FT_STILL_FT = 3.0                          # and hardly moves while the others line up
FT_WINDOW_S = (2.6, 0.9)                   # how long before the ball reaches the rim the shooter stands there


def free_throw_setup(looks: list[Look], end_t: float) -> bool:
    """Someone stands still at the free-throw line while the ball goes up: most frames of the seconds before the shot
    have a person within reach of the middle of the line."""
    window = [k for k in looks if end_t - FT_WINDOW_S[0] <= k.t <= end_t - FT_WINDOW_S[1] and k.mapping is not None]
    if not window:
        return False
    spots = []
    for k in window:
        near = []
        for b in k.people:
            at = k.mapping.to_court((b[0] + b[2]) / 2, b[3])
            if at is None:
                continue
            x, y = fold(*at, k.mapping.length)
            d = float(np.hypot(x - WIDTH / 2, y - FT_LINE_Y))
            if d <= FT_NEAR_FT:
                near.append((d, x, y))
        if near:
            spots.append(min(near)[1:])
    if len(spots) / len(window) < FT_STILL_SHARE:
        return False
    xs, ys = zip(*spots)
    return float(np.hypot(np.ptp(xs), np.ptp(ys))) <= FT_STILL_FT


FREE_THROW_SPOT = (WIDTH / 2, FT_LINE_Y)
FULL_CONF_POINTS = 8                       # a flight seen this many times is as sure as a flight gets


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def set_aside(looks: list[Look], t: float, box, min_iou: float = SAME_PERSON_IOU) -> list[Look]:
    """The looks without this person: followed by overlap from the frame at t, both ways, through the whole window."""
    i = min(range(len(looks)), key=lambda j: abs(looks[j].t - t))
    out = list(looks)
    for order in (range(i, len(looks)), range(i - 1, -1, -1)):
        last = tuple(box)
        for j in order:
            k = looks[j]
            same = max(k.people, key=lambda b: _iou(b, last), default=None)
            if same is None or _iou(same, last) < min_iou:
                continue                       # hidden in this frame: keep looking from where they were last seen
            out[j] = replace(k, people=[b for b in k.people if b is not same])
            last = same
    return out


def shot_spot(looks: list[Look], rims_px: list[tuple[float, float]], arrival_t: float | None, t: float, points: int | None,
              scale: float = 1.0) -> dict:
    """Everything an event gains: the spot on the half court (x, y are None when nothing places it), its confidence and
    how it was found. +1 on the scoreboard, or someone standing still at the line while the ball goes up, is a free throw
    at the middle of the line: the flight would pick whichever player in the lane the ball passes in the picture."""
    if points == 1 or free_throw_setup(looks, t):
        return {"x": FREE_THROW_SPOT[0], "y": FREE_THROW_SPOT[1], "conf": 0.9, "method": "free-throw", "flags": [], "shot_type": "free throw",
                "release_t": None, "flight_points": 0, "reach_px": None, "box": None}
    s = locate(looks, rims_px, arrival_t=arrival_t, scale=scale)
    if s.x is None:
        return {"x": None, "y": None, "conf": None, "method": "none", "flags": s.flags, "shot_type": None, "release_t": None,
                "flight_points": s.arc_points, "reach_px": None, "box": None}
    look = min(looks, key=lambda k: abs(k.t - s.release_t))
    x, y = fold(s.x, s.y, look.mapping.length)
    x, y = float(min(max(x, 0.0), WIDTH)), float(min(max(y, 0.0), 47.0))
    conf = min(1.0, s.arc_points / FULL_CONF_POINTS) * look.mapping.conf * (0.5 if s.flags else 1.0)
    return {"x": round(x, 1), "y": round(y, 1), "conf": round(float(conf), 2), "method": "flight", "flags": s.flags, "shot_type": None,
            "release_t": round(float(s.release_t), 2), "flight_points": s.arc_points, "reach_px": s.reach_px, "box": s.box}
