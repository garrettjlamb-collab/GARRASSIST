"""Dead end, kept as a record (2026-11-01): improved 3 of 7 knocked courts and flew one off the picture, chasing the crowd's edges. Pull a rough court onto the painted lines. The court's lines through a mapping should lie on edges in the picture; from a rough start a local search over the mapping lowers
their distance to the nearest edge. It only ever starts from a draft, because a gym's floor holds other sports' lines and logos: on its own it would settle on those."""
from __future__ import annotations

import cv2
import numpy as np

from film.export.court_sheet import court_lines
from film.interpret.court import Mapping

CAP_PX = 14.0
WORK_WIDTH = 960
STEPS_PX = (24.0, 12.0, 6.0, 3.0)
ROUNDS = 6
MIN_ON_FLOOR = 60


def edge_distance(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Distance to the nearest strong edge at every pixel, at working size, and the scale that goes with it."""
    scale = WORK_WIDTH / image_bgr.shape[1]
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    edges = cv2.Canny(gray, 30, 90)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), scale


def _samples(length: float) -> np.ndarray:
    pts = []
    for line in court_lines(length):
        pts.append(np.array(line)[::2])
    return np.vstack(pts)


def _cost(back: np.ndarray, feet: np.ndarray, dist: np.ndarray, scale: float) -> tuple[float, int]:
    v = back @ np.column_stack([feet, np.ones(len(feet))]).T
    ok = v[2] > 1e-9
    x, y = v[0, ok] / v[2, ok] * scale, v[1, ok] / v[2, ok] * scale
    h, w = dist.shape
    keep = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    if keep.sum() < MIN_ON_FLOOR:
        return CAP_PX, int(keep.sum())
    d = dist[y[keep].astype(int), x[keep].astype(int)]
    return float(np.minimum(d, CAP_PX).mean()), int(keep.sum())


def snap(image_bgr: np.ndarray, mapping: Mapping, corners: list[tuple[float, float]]) -> dict:
    """The search moves the pixels of four floor points, which keeps the mapping a real one. `corners` are feet. Returns the moved pixels and the costs before and after."""
    dist, scale = edge_distance(image_bgr)
    feet = _samples(mapping.length)
    src = np.array(corners, np.float32)
    back0 = np.linalg.inv(mapping.H)
    start = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64), back0).reshape(-1, 2)

    def back_of(p):
        return cv2.getPerspectiveTransform(src, (start + p.reshape(-1, 2)).astype(np.float32))

    def f(p):
        return _cost(back_of(p), feet, dist, scale)[0] + 1e-4 * float(np.abs(p).mean())

    before = _cost(back0, feet, dist, scale)[0]
    p = np.zeros(8)
    best = f(p)
    unit = image_bgr.shape[1] / WORK_WIDTH
    for step in STEPS_PX:                  # one coordinate at a time: deterministic, and needs nothing beyond numpy
        for _ in range(ROUNDS):
            moved = False
            for i in range(8):
                for sign in (1.0, -1.0):
                    q = p.copy()
                    q[i] += sign * step * unit
                    c = f(q)
                    if c < best - 1e-4:
                        p, best, moved = q, c, True
                        break
            if not moved:
                break
    return {"pixels": start + p.reshape(-1, 2), "before": round(before, 2), "after": round(_cost(back_of(p), feet, dist, scale)[0], 2), "moved_px": round(float(np.abs(p).max()), 1)}
