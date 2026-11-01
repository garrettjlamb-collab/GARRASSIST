"""The shooter by their body. Whoever a ball's flight first touches in the picture may only be standing under it; a shooter has their
arms up with the ball at their hands around the moment the flight starts. And a jump shooter's feet are off the floor at the release,
which through the court mapping reads as feet further from the camera, so the spot comes from the last moment they were on it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from film.interpret.court import Mapping
from film.interpret.location import COURT_MARGIN_FT, GRAVITY_PX, _carry, _iou

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE = 5, 6, 9, 10, 11, 12, 15, 16
KP_CONF = 0.3
BEFORE_S, AFTER_S = 1.0, 0.3               # around the flight's first sighting, the moments the release can be
ARMS_FULL = 0.8                            # wrists this many torsos above the shoulders are as up as arms get
ONE_ARM = 0.7                              # one wrist up counts for less than two
BOTH_UP = 0.2
BALL_FALLOFF = 1.2                         # the ball's distance from a wrist, in torsos, at which the score falls by e
MIN_TORSO_OF_BOX, MIN_TORSO_PX = 0.2, 8.0
TORSO_OF_BOX = 0.3                         # a torso's share of the box when the hips cannot be seen
TAKEOFF_S = 0.5                            # how far back to look for the feet on the floor
SAME_PERSON_IOU = 0.25
SAME_PERSON_FT = 3.0


@dataclass(frozen=True)
class Pick:
    score: float
    t: float
    box: tuple[float, float, float, float]
    feet: tuple[float, float]              # full-court feet, on the floor
    length: float
    mapping_conf: float


def arms_and_ball(person: dict, ball: tuple[float, float]) -> float:
    """0 to 1: wrists above the shoulders by a share of the torso, with the ball at a wrist."""
    kp = np.array(person["kp"])
    if kp[LEFT_SHOULDER, 2] < KP_CONF or kp[RIGHT_SHOULDER, 2] < KP_CONF:
        return 0.0
    wrists = [kp[i] for i in (LEFT_WRIST, RIGHT_WRIST) if kp[i, 2] >= KP_CONF]
    if not wrists:
        return 0.0
    shoulders = (kp[LEFT_SHOULDER, :2] + kp[RIGHT_SHOULDER, :2]) / 2
    box_h = person["box"][3] - person["box"][1]
    if kp[LEFT_HIP, 2] >= KP_CONF and kp[RIGHT_HIP, 2] >= KP_CONF:
        torso = float(np.linalg.norm((kp[LEFT_HIP, :2] + kp[RIGHT_HIP, :2]) / 2 - shoulders))
    else:
        torso = TORSO_OF_BOX * box_h
    torso = max(torso, MIN_TORSO_OF_BOX * box_h, MIN_TORSO_PX)
    heights = [(shoulders[1] - w[1]) / torso for w in wrists]
    arms = float(np.clip(max(heights) / ARMS_FULL, 0.0, 1.0)) * (1.0 if len(wrists) == 2 and min(heights) > BOTH_UP else ONE_ARM)
    near = min(float(np.hypot(w[0] - ball[0], w[1] - ball[1])) for w in wrists) / torso
    return arms * float(np.exp(-near / BALL_FALLOFF))


def foot_pixel(person: dict) -> tuple[float, float]:
    """The lower ankle when the ankles can be seen, else the bottom of the box."""
    kp = person["kp"]
    ankles = [kp[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if kp[i][2] >= KP_CONF]
    if ankles:
        return float(np.mean([a[0] for a in ankles])), float(max(a[1] for a in ankles))
    b = person["box"]
    return (b[0] + b[2]) / 2, b[3]


def feet_on_floor(frames: list[dict], i: int, person: dict, mapping_at: Callable[[float], Mapping | None], camera_xy: tuple[float, float] | None) -> tuple[float, float] | None:
    """Where the person's feet were on the floor: back through the moments before the release, the point nearest the camera, because a
    foot in the air reads as further from it. With no camera, the feet at the release."""
    mapping = mapping_at(frames[i]["t"])
    here = mapping.to_court(*foot_pixel(person)) if mapping is not None else None
    if camera_xy is None:
        return here
    best, best_d = here, None if here is None else float(np.hypot(here[0] - camera_xy[0], here[1] - camera_xy[1]))
    last = person["box"]
    for j in range(i - 1, -1, -1):
        if frames[i]["t"] - frames[j]["t"] > TAKEOFF_S:
            break
        same = max(frames[j]["people"], key=lambda p: _iou(p["box"], last), default=None)
        if same is None or _iou(same["box"], last) < SAME_PERSON_IOU:
            continue
        last = same["box"]
        mp = mapping_at(frames[j]["t"])
        p = mp.to_court(*foot_pixel(same)) if mp is not None else None
        if p is None:
            continue
        d = float(np.hypot(p[0] - camera_xy[0], p[1] - camera_xy[1]))
        if best_d is None or d < best_d:
            best, best_d = p, d
    return best


def picks(frames: list[dict], flight: list[tuple[float, float, float, float]], anchor: Mapping, mapping_at: Callable[[float], Mapping | None],
          min_score: float, scale: float = 1.0, camera_xy: tuple[float, float] | None = None) -> list[Pick]:
    """People who look like they are releasing this flight, best first, one pick per person. `flight` is in the pixels of `anchor`'s frame."""
    first, second = flight[0], flight[1]
    dt = second[0] - first[0]
    vx, vy = (second[1] - first[1]) / dt, (second[2] - first[2]) / dt
    seen = {round(p[0], 1): p for p in flight}
    inv = None
    scored = []
    for i, frame in enumerate(frames):
        t = frame["t"]
        if not first[0] - BEFORE_S <= t <= first[0] + AFTER_S:
            continue
        mapping = mapping_at(t)
        if mapping is None:
            continue
        p = seen.get(round(t, 1))
        if p is not None:
            here = p[1], p[2]
        else:
            back = first[0] - t
            here = first[1] - vx * back, first[2] - vy * back + 0.5 * GRAVITY_PX * scale * back * back
        ball = _carry(anchor.H, np.linalg.inv(mapping.H), *here)
        if ball is None:
            continue
        for person in frame["people"]:
            b = person["box"]
            if not mapping.contains((b[0] + b[2]) / 2, b[3], margin=COURT_MARGIN_FT):
                continue
            score = arms_and_ball(person, ball)
            if score >= min_score:
                scored.append((score, i, person, mapping))
    out: list[Pick] = []
    for score, i, person, mapping in sorted(scored, key=lambda s: -s[0]):
        feet = feet_on_floor(frames, i, person, mapping_at, camera_xy)
        if feet is None or any(np.hypot(feet[0] - o.feet[0], feet[1] - o.feet[1]) <= SAME_PERSON_FT for o in out):
            continue
        out.append(Pick(score=round(score, 3), t=frames[i]["t"], box=tuple(person["box"][:4]), feet=feet, length=mapping.length, mapping_conf=mapping.conf))
    return out
