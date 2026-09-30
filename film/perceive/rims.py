from __future__ import annotations

import math

import numpy as np
import pandas as pd

from film.perceive.detectors import Box, Detector

MAX_RIMS = 2


class RimTracker:
    """Stable ids for rim boxes across frames: match by centre distance, expire after a gap.

    Distance rather than overlap, because the Hudl camera pans and a fast pan moves a rim
    by more than its own width between frames at 10 fps.
    """

    def __init__(self, max_gap_frames: int = 30, distance_factor: float = 2.0):
        self.max_gap = max_gap_frames
        self.factor = distance_factor
        self.next_id = 1
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


def rim_row(frame: int, t: float, track_id: int, box: Box, model_version: str) -> dict:
    return {"frame": frame, "t": t, "kind": "rim", "track_id": track_id, "ref_id": None,
            "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "conf": box.conf,
            "on_court": None, "court_x": None, "court_y": None, "team": None, "model_version": model_version}


def detect_rims(frame_bgr: np.ndarray, detector: Detector, tracker: RimTracker,
                frame: int, t: float, model_version: str) -> list[dict]:
    boxes = [b for b in detector.detect(frame_bgr) if b.cls == "rim"]
    boxes = sorted(boxes, key=lambda b: -b.conf)[:MAX_RIMS]
    return [rim_row(frame, t, tid, box, model_version) for tid, box in tracker.update(boxes, frame)]


def smooth_rims(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["track_id", "frame"]).copy()
    for col in ("x1", "y1", "x2", "y2"):
        df[col] = df.groupby("track_id")[col].transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
    return df.sort_values("frame", kind="stable").reset_index(drop=True)
