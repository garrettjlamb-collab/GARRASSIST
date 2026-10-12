"""The court as the tracker draws it, and the eighteen points the court finder looks for.

Coordinates are the tracker's: 50 ft wide (x), 47 ft deep (y) per half, baseline at y = 0, basket at
(25, 5.25). The full court is 94 ft; the far end is the near end rotated about centre court."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

WIDTH, HALF_DEPTH, FULL_DEPTH = 50.0, 47.0, 94.0
BASKET = (25.0, 5.25)
THREE_RADIUS = 19.75                    # NFHS: the arc meets the baseline, no straight corner segment
LANE_LEFT, LANE_RIGHT, FT_LINE_Y = 17.0, 33.0, 19.0
FT_CIRCLE_RADIUS = 6.0

_ARC_HALF_WIDTH = math.sqrt(THREE_RADIUS ** 2 - BASKET[1] ** 2)       # where the arc meets y = 0

_NEAR = {
    "corner-left": (0.0, 0.0), "corner-right": (WIDTH, 0.0),
    "lane-left": (LANE_LEFT, 0.0), "lane-right": (LANE_RIGHT, 0.0),
    "ft-left": (LANE_LEFT, FT_LINE_Y), "ft-right": (LANE_RIGHT, FT_LINE_Y),
    "ft-top": (BASKET[0], FT_LINE_Y + FT_CIRCLE_RADIUS),
    "arc-left": (BASKET[0] - _ARC_HALF_WIDTH, 0.0), "arc-right": (BASKET[0] + _ARC_HALF_WIDTH, 0.0),
}


def _rotate(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (WIDTH - x, FULL_DEPTH - y)


POINT_NAMES: list[str] = [f"near-{k}" for k in _NEAR] + [f"far-{k}" for k in _NEAR]
TEMPLATE: dict[str, tuple[float, float]] = {**{f"near-{k}": v for k, v in _NEAR.items()},
                                            **{f"far-{k}": _rotate(v) for k, v in _NEAR.items()}}
END_OF_POINT: dict[str, str] = {n: n.split("-", 1)[0] for n in POINT_NAMES}


def _mirror_name(name: str) -> str:
    if "left" in name:
        return name.replace("left", "right")
    if "right" in name:
        return name.replace("right", "left")
    return name


FLIP_INDEX: list[int] = [POINT_NAMES.index(_mirror_name(n)) for n in POINT_NAMES]


def fold(x: float, y: float) -> tuple[float, float]:
    """A full-court point onto the half court of the end it is in: the far end rotates about centre."""
    if y > HALF_DEPTH:
        return (WIDTH - x, FULL_DEPTH - y)
    return (x, y)


MIN_POINTS = 4
GOOD_ERROR_PX = 3.0                    # reprojection error at which confidence is full


@dataclass(frozen=True)
class Mapping:
    H: np.ndarray                      # pixels -> full-court feet (homogeneous)
    points_used: int
    error_px: float
    conf: float
    # Advisory only: which end the labels called near, by majority of the inlier points. On a sideline view it is not
    # reliable (fliplr augmentation and per-frame labelling can swap near and far); x/y are unaffected because fold
    # cancels the choice. A shot's end comes from the rim, not from this.
    end: str                           # "near" or "far": the end most of the points belong to

    def to_court(self, px: float, py: float) -> tuple[float, float] | None:
        """Full-court feet for a pixel, or None when the pixel is at or above the horizon (no floor point there)."""
        v = self.H @ np.array([px, py, 1.0])
        if v[2] <= 0:                  # fit_mapping scales H so w > 0 on the court; w <= 0 is the sky side
            return None
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def to_half(self, px: float, py: float) -> tuple[float, float] | None:
        p = self.to_court(px, py)
        return None if p is None else fold(*p)

    def contains(self, px: float, py: float, margin: float = 2.0) -> bool:
        p = self.to_court(px, py)
        if p is None:
            return False
        x, y = p
        return -margin <= x <= WIDTH + margin and -margin <= y <= FULL_DEPTH + margin


def fit_mapping(points: dict[str, tuple[float, float]], reproj_tol_px: float = 8.0) -> Mapping | None:
    """A homography from the named pixel points to the court template, RANSAC against bad points.
    None with fewer than four points, or fewer than four inliers."""
    names = [n for n in points if n in TEMPLATE]
    if len(names) < MIN_POINTS:
        return None
    # Fit template feet -> pixels so the RANSAC threshold is in pixels: OpenCV measures it in destination units.
    src = np.array([TEMPLATE[n] for n in names], dtype=np.float64)
    dst = np.array([points[n] for n in names], dtype=np.float64)
    back, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_tol_px)
    if back is None:
        return None
    inliers = [n for n, keep in zip(names, mask.ravel()) if keep]
    if len(inliers) < MIN_POINTS:
        return None
    if len(inliers) < len(names):
        back, _ = cv2.findHomography(np.array([TEMPLATE[n] for n in inliers]), np.array([points[n] for n in inliers]), 0)
        if back is None:
            return None
    # Orientation guard. For a camera above the floor, det of the Jacobian of feet -> pixels is det(back) / w^3 and has
    # one sign over the whole visible court. The template is right-handed (x right along the near baseline as seen from
    # the near end, y away from it) and image y grows downward, so a true labelling gives a negative determinant
    # everywhere; a positive one is the court seen in a mirror (left and right names swapped). Refuse both that and
    # mixed signs (inliers on both sides of the horizon), rather than fit a mirrored court silently.
    ws = np.array([(back @ np.array([*TEMPLATE[n], 1.0]))[2] for n in inliers])
    jac = np.linalg.det(back) / ws ** 3
    if not np.all(jac < 0):
        return None
    if ws[0] < 0:                      # scale so w > 0 on the court both ways: Mapping.to_court reads w <= 0 as sky
        back = -back
    try:
        H = np.linalg.inv(back)
    except np.linalg.LinAlgError:
        return None
    errors = []
    for n in inliers:
        v = back @ np.array([*TEMPLATE[n], 1.0])
        errors.append(float(np.hypot(v[0] / v[2] - points[n][0], v[1] / v[2] - points[n][1])))
    error = float(np.mean(errors))
    conf = min(1.0, len(inliers) / 8.0 + 0.5) * min(1.0, GOOD_ERROR_PX / max(error, GOOD_ERROR_PX))
    end = "far" if sum(END_OF_POINT[n] == "far" for n in inliers) * 2 > len(inliers) else "near"
    return Mapping(H=H, points_used=len(inliers), error_px=error, conf=round(conf, 3), end=end)
