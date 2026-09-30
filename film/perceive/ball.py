from __future__ import annotations

import numpy as np

from film.perceive.detectors import Box, Detector

CROP_SIZE = 320
UP_SHIFT = 0.15   # fraction of the crop by which its centre sits above the rim


def rim_crop_box(rim: Box, frame_w: int, frame_h: int, size: int = CROP_SIZE) -> tuple[int, int, int, int]:
    """A native-resolution square around the rim with more room above it than below, clamped to the frame."""
    size = min(size, frame_w, frame_h)
    cx, cy = rim.cx, rim.cy - UP_SHIFT * size
    x1 = int(round(cx - size / 2))
    y1 = int(round(cy - size / 2))
    x1 = max(0, min(x1, frame_w - size))
    y1 = max(0, min(y1, frame_h - size))
    return x1, y1, x1 + size, y1 + size


def crop(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return frame_bgr[y1:y2, x1:x2]


def ball_row(frame: int, t: float, rim_id: int, box: Box, model_version: str) -> dict:
    return {"frame": frame, "t": t, "kind": "ball", "track_id": None, "ref_id": rim_id,
            "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "conf": box.conf,
            "on_court": None, "court_x": None, "court_y": None, "team": None, "model_version": model_version}


def detect_ball_near_rim(frame_bgr: np.ndarray, rims: list[tuple[int, Box]], detector: Detector,
                         frame: int, t: float, model_version: str) -> list[dict]:
    h, w = frame_bgr.shape[:2]
    rows = []
    for rim_id, rim in rims:
        cb = rim_crop_box(rim, w, h)
        balls = [b for b in detector.detect(crop(frame_bgr, cb)) if b.cls == "ball"]
        if not balls:
            continue
        best = max(balls, key=lambda b: b.conf).shifted(cb[0], cb[1])
        rows.append(ball_row(frame, t, rim_id, best, model_version))
    return rows
