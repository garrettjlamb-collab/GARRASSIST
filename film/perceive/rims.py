from __future__ import annotations

import math

import numpy as np
import pandas as pd

from film.perceive.detectors import Box, Detector

MAX_RIMS = 2
SIDE_IDS = {"left": 1, "right": 2}
TRACKED_FIRST_ID = 3


class RimTracker:
    """Stable ids for rim boxes across frames: match by centre distance, expire after a gap.

    Distance rather than overlap, because the Hudl camera pans and a fast pan moves a rim
    by more than its own width between frames at 10 fps.
    """

    def __init__(self, max_gap_frames: int = 30, distance_factor: float = 2.0, first_id: int = 1):
        self.max_gap = max_gap_frames
        self.factor = distance_factor
        self.next_id = first_id
        self.active: dict[int, tuple[Box, int]] = {}

    def update(self, boxes: list[Box], frame: int) -> list[tuple[int, Box]]:
        self.active = {tid: (b, f) for tid, (b, f) in self.active.items() if frame - f <= self.max_gap}
        assigned: list[tuple[int, Box]] = []
        used: set[int] = set()
        for box in sorted(boxes, key=lambda b: -b.conf):
            best, best_d = None, math.inf
            for tid, (prev, _) in self.active.items():
                if tid in used:
                    continue
                d = math.hypot(box.cx - prev.cx, box.cy - prev.cy)
                if d < self.factor * max(prev.w, prev.h, 1.0) and d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.active[best] = (box, frame)
            assigned.append((best, box))
        return assigned


# Where the rim reference sits on a backboard, as fractions of the backboard box. Matched by eye to the
# Hudl templates' box: a strip at the bottom of the backboard about 0.6 of its width, 0.24 as tall as wide.
RIM_WIDTH_OF_BACKBOARD = 0.6
RIM_ASPECT = 0.24
RIM_BOTTOM_ABOVE_BOARD_BOTTOM = 0.05


def rim_from_backboard(board: Box) -> Box:
    """The rim reference for a backboard box, for detectors trained on backboards: on a wide shot the rim
    itself is a few pixels, the backboard is big and crisp."""
    w = RIM_WIDTH_OF_BACKBOARD * board.w
    y2 = board.y2 - RIM_BOTTOM_ABOVE_BOARD_BOTTOM * board.h
    return Box(board.cx - w / 2, y2 - RIM_ASPECT * w, board.cx + w / 2, y2, board.conf, "rim")


def rim_boxes(boxes: list[Box]) -> list[Box]:
    """The rims among a detector's boxes; when it boxed none, the ones its backboards imply."""
    rims = [b for b in boxes if b.cls == "rim"]
    return rims or [rim_from_backboard(b) for b in boxes if b.cls == "backboard"]


def rim_row(frame: int, t: float, track_id: int, box: Box, model_version: str) -> dict:
    return {"frame": frame, "t": t, "kind": "rim", "track_id": track_id, "ref_id": None,
            "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "conf": box.conf,
            "on_court": None, "court_x": None, "court_y": None, "team": None, "model_version": model_version}


def detect_rims(frame_bgr: np.ndarray, detector: Detector, tracker: RimTracker,
                frame: int, t: float, model_version: str) -> list[dict]:
    """Rim rows for one frame. A box that names its physical basket (a fixed-camera template's side)
    takes that basket's fixed id, so ids survive any gap and any re-run; other boxes go through the
    tracker, whose ids start above the fixed ones (construct it with first_id=TRACKED_FIRST_ID)."""
    boxes = rim_boxes(detector.detect(frame_bgr))
    boxes = sorted(boxes, key=lambda b: -b.conf)[:MAX_RIMS]
    fixed = [(SIDE_IDS[b.side], b) for b in boxes if b.side in SIDE_IDS]
    tracked = tracker.update([b for b in boxes if b.side not in SIDE_IDS], frame)
    return [rim_row(frame, t, tid, box, model_version) for tid, box in sorted(fixed + tracked, key=lambda p: p[0])]


def smooth_rims(df: pd.DataFrame, window: int = 5, max_gap: int = 30) -> pd.DataFrame:
    """Rolling median of each rim's box, within each visit: a gap of more than max_gap frames starts a
    new visit, so a rim re-acquired after a pan is not pulled toward where it was minutes earlier."""
    if df.empty:
        return df
    df = df.sort_values(["track_id", "frame"]).copy()
    # The first row of each rim has no previous frame: with nullable integers that is a missing value, not False.
    visit = df.groupby("track_id")["frame"].diff().gt(max_gap).fillna(False).astype(bool).groupby(df["track_id"]).cumsum()
    keys = [df["track_id"], visit]
    for col in ("x1", "y1", "x2", "y2"):
        df[col] = df.groupby(keys)[col].transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
    return df.sort_values("frame", kind="stable").reset_index(drop=True)
