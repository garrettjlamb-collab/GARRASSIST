from __future__ import annotations

import cv2
import numpy as np

from film.perceive.detectors import Box

TILE = 96
GLOBAL_SIZE = (96, 54)


def zone_box(rim: Box, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    side = int(round(3 * max(rim.w, rim.h, 8.0)))
    side = min(side, frame_w, frame_h)
    x1 = int(round(rim.cx - side / 2))
    y1 = int(round(rim.cy - side / 2))
    x1 = max(0, min(x1, frame_w - side))
    y1 = max(0, min(y1, frame_h - side))
    return x1, y1, x1 + side, y1 + side


def gray_tile(frame_bgr: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


class MotionEnergy:
    """Mean absolute frame difference in a tight box around each rim, alongside the whole-frame difference."""

    def __init__(self):
        self.prev_rim: dict[int, np.ndarray] = {}
        self.prev_global: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray, rims: list[tuple[int, Box]], frame: int, t: float) -> list[dict]:
        h, w = frame_bgr.shape[:2]
        whole = gray_tile(frame_bgr, (0, 0, w, h), GLOBAL_SIZE).astype(np.int16)
        energy_global = float(np.abs(whole - self.prev_global).mean()) if self.prev_global is not None else 0.0
        self.prev_global = whole
        rows = []
        for rim_id, rim in rims:
            tile = gray_tile(frame_bgr, zone_box(rim, w, h), (TILE, TILE)).astype(np.int16)
            prev = self.prev_rim.get(rim_id)
            energy = float(np.abs(tile - prev).mean()) if prev is not None else 0.0
            self.prev_rim[rim_id] = tile
            rows.append({"frame": frame, "t": t, "rim_id": rim_id, "energy": energy, "energy_global": energy_global})
        return rows
