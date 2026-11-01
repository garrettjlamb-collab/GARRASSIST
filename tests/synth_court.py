"""A seeded synthetic gym for the registration tests: a textured panorama with the court drawn through a known
feet -> panorama map, and frames cut from it through known homographies, the way a turning camera sees a gym."""
from __future__ import annotations

import cv2
import numpy as np

from film.interpret.court import template

PANO_W, PANO_H = 2600, 800
FRAME_W, FRAME_H = 640, 360
# Full-court feet -> panorama pixels: y (along the court) runs right, x (across) runs down. A rotation of the
# tracker's x-right / y-up picture, so a labelling through it passes fit_mapping's orientation guard.
P = np.array([[0.0, 24.0, 150.0], [12.0, 0.0, 100.0], [0.0, 0.0, 1.0]])


def panorama(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(60, 200, (PANO_H, PANO_W, 3)).astype(np.uint8), (0, 0), 3)
    for _ in range(1400):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, PANO_W)), int(rng.integers(0, PANO_H))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(img, (x, y), int(rng.integers(3, 14)), color, -1)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(4, 30)), y + int(rng.integers(4, 30))), color, -1)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-40, 40)), y + int(rng.integers(-40, 40))), color, 2)
    return img


def crop(x0: float, y0: float = 150.0, zoom: float = 1.0) -> np.ndarray:
    """Panorama pixels -> frame pixels for a frame whose top-left is (x0, y0) in the panorama, magnified by zoom."""
    return np.array([[zoom, 0.0, -x0 * zoom], [0.0, zoom, -y0 * zoom], [0.0, 0.0, 1.0]])


def frame(pano: np.ndarray, G: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(pano, G, (FRAME_W, FRAME_H))


def apply(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = H @ np.array([x, y, 1.0])
    return (float(v[0] / v[2]), float(v[1] / v[2]))


def court_points_in(G: np.ndarray, inside_only: bool = True, length: float = 94.0) -> dict[str, tuple[float, float]]:
    """The template's named points (a court `length` ft long) in a frame's pixels; only those inside the frame."""
    out = {}
    for name, feet in template(length).items():
        px = apply(G @ P, *feet)
        if not inside_only or (0 <= px[0] < FRAME_W and 0 <= px[1] < FRAME_H):
            out[name] = px
    return out


def feet_of(G: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """The truth: full-court feet under a frame pixel."""
    return apply(np.linalg.inv(G @ P), px, py)
