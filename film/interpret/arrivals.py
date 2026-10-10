from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

MIN_GAP_S = 1.5
MAX_FRAME_GAP = 3
# A real ball is detected with confidence at least once on its way to the rim; wall pads and banners
# boxed while the camera pans never are. Measured on the Marcos game with ball@v4: false crossings
# peak at 0.25-0.33, real ones at 0.5 or more.
MIN_PEAK_CONF = 0.4
PEAK_WINDOW_FRAMES = 8
# Zoomed in on the basket, a ball rolling round the rim crosses the rim line more than once. A second
# crossing within this window is the same shot unless the ball went back up above the rim by this much in
# between (a rebound and a putback). Measured on sixteen reviewed games: a 3 s window and a rise of one rim
# width drop 79 of 123 probable duplicates and lose 13 of 1,419 single shots; 4 s would lose 41.
REPEAT_WINDOW_S = 3.0
RISE_WIDTHS = 1.0
MOTION_Z = 3.0
MERGE_WINDOW_S = 2.0


@dataclass
class Arrival:
    t: float
    frame: int
    rim_id: int
    rim_box: tuple[float, float, float, float]
    source: str
    conf: float


def _rim_lookup(rim_df: pd.DataFrame) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    return {(int(r.frame), int(r.track_id)): (float(r.x1), float(r.y1), float(r.x2), float(r.y2)) for r in rim_df.itertuples()}


def find_ball_arrivals(ball_df: pd.DataFrame, rim_df: pd.DataFrame, fps: float, min_gap_s: float = MIN_GAP_S) -> list[Arrival]:
    """The ball's centre crossing the rim's top edge downward while horizontally over the rim."""
    if ball_df.empty or rim_df.empty:
        return []
    rims = _rim_lookup(rim_df)
    out: list[Arrival] = []
    for rim_id, group in ball_df.sort_values("frame").groupby("ref_id"):
        rim_id = int(rim_id)
        prev: tuple[int, float] | None = None
        last_t = -math.inf
        highest_since = math.inf                      # the ball's highest point (smallest y) since the last arrival
        recent: list[tuple[int, float]] = []          # (frame, confidence) of the last detections at this rim
        for b in group.itertuples():
            recent = [(f, c) for f, c in recent if int(b.frame) - f <= PEAK_WINDOW_FRAMES] + [(int(b.frame), float(b.conf))]
            rim = rims.get((int(b.frame), rim_id))
            if rim is None:
                prev = None
                continue
            rx1, ry1, rx2, _ = rim
            width = max(rx2 - rx1, 1.0)
            cx, cy = (float(b.x1) + float(b.x2)) / 2, (float(b.y1) + float(b.y2)) / 2
            over_rim = rx1 - width <= cx <= rx2 + width
            crossed = prev is not None and int(b.frame) - prev[0] <= MAX_FRAME_GAP and prev[1] < ry1 <= cy
            sure = max(c for _, c in recent) >= MIN_PEAK_CONF
            went_back_up = highest_since < ry1 - RISE_WIDTHS * width
            repeat = float(b.t) - last_t < REPEAT_WINDOW_S and not went_back_up
            if crossed and over_rim and sure and float(b.t) - last_t >= min_gap_s and not repeat:
                out.append(Arrival(t=float(b.t), frame=int(b.frame), rim_id=rim_id, rim_box=rim, source="ball", conf=0.8))
                last_t = float(b.t)
                highest_since = math.inf
            else:
                highest_since = min(highest_since, cy)
            prev = (int(b.frame), cy)
    return sorted(out, key=lambda a: a.t)


def find_motion_arrivals(motion_df: pd.DataFrame, rim_df: pd.DataFrame, fps: float,
                         z: float = MOTION_Z, min_gap_s: float = MIN_GAP_S) -> list[Arrival]:
    """Local peaks of rim motion energy (above the whole-frame energy) more than z standard deviations out.

    The mean and spread are the whole game's, over every rim id: a short-lived id (a brief false rim,
    or a rim re-acquired after a pan) has too few rows for its own statistics to mean anything."""
    if motion_df.empty or rim_df.empty:
        return []
    rims = _rim_lookup(rim_df)
    half = max(1, int(round(0.5 * fps)))
    all_rel = (motion_df["energy"] - motion_df["energy_global"]).to_numpy(dtype=float)
    if all_rel.std() == 0:
        return []
    threshold = all_rel.mean() + z * all_rel.std()
    out: list[Arrival] = []
    for rim_id, group in motion_df.sort_values("frame").groupby("rim_id"):
        rim_id = int(rim_id)
        rel = (group["energy"] - group["energy_global"]).to_numpy(dtype=float)
        frames = group["frame"].to_numpy()
        times = group["t"].to_numpy(dtype=float)
        if len(rel) < 3:
            continue
        last_t = -math.inf
        for i in range(len(rel)):
            if rel[i] < threshold or rel[i] < rel[max(0, i - half): i + half + 1].max():
                continue
            if times[i] - last_t < min_gap_s:
                continue
            rim = rims.get((int(frames[i]), rim_id))
            if rim is None:
                continue
            out.append(Arrival(t=float(times[i]), frame=int(frames[i]), rim_id=rim_id, rim_box=rim, source="motion", conf=0.4))
            last_t = float(times[i])
    return sorted(out, key=lambda a: a.t)


def merge_arrivals(primary: list[Arrival], secondary: list[Arrival], window_s: float = MERGE_WINDOW_S) -> list[Arrival]:
    """Every primary arrival, plus secondary ones not within window_s of a primary at the same rim."""
    out = list(primary)
    for s in secondary:
        if any(p.rim_id == s.rim_id and abs(p.t - s.t) <= window_s for p in primary):
            continue
        out.append(s)
    return sorted(out, key=lambda a: (a.t, a.rim_id))
