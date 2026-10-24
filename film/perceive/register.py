"""Lines one frame up with another by the gym's own markings: SIFT features, a ratio test, a RANSAC homography.

A camera that only turns sees the whole gym through one homography from any frame to any other: Hudl Focus pans
and zooms inside a fixed sensor, and an iPad on a tripod turns on its mount (a hand-held one moves a few
centimetres, which at twenty metres is under a pixel). So a few frames with the court marked place the court in
every frame that lines up with one of them (film/perceive/court_refs.py)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from film.ingest.profile import HUDL_OVERLAY_REGION

WORK_WIDTH = 960                  # features are found on the frame shrunk to this width; answers are in full pixels
N_FEATURES = 2000
RATIO = 0.75                      # Lowe's ratio test
RANSAC_PX = 3.0                   # at work width
MIN_INLIERS = 40
MIN_INLIER_SHARE = 0.25           # of the matches that pass the ratio test
MAX_AREA_RATIO = 6.0              # a frame cannot grow or shrink more than this from one frame to the next


@dataclass(frozen=True)
class Features:
    keypoints: np.ndarray         # (n, 2) full-resolution pixels
    descriptors: np.ndarray       # (n, 128) float32
    width: int
    height: int
    scale: float                  # work width / full width (1.0 for a frame already at or under it)


@dataclass(frozen=True)
class Match:
    H: np.ndarray                 # a's full-resolution pixels -> b's
    inliers: int
    share: float


_SIFT = None
_MATCHER = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=N_FEATURES)
    return _SIFT


def _matcher():
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = cv2.BFMatcher(cv2.NORM_L2)          # brute force: the same answer every run
    return _MATCHER


def mask_regions(profile: str) -> list[tuple[int, int, int, int]]:
    """Full-resolution boxes that hold no gym: a burned-in scoreboard sits still while the camera turns, and its
    features would vote for the camera not having moved."""
    return [HUDL_OVERLAY_REGION] if profile.startswith("hudl") else []


def features(frame_bgr: np.ndarray, mask_regions: list | tuple = ()) -> Features:
    h, w = frame_bgr.shape[:2]
    s = WORK_WIDTH / w if w > WORK_WIDTH else 1.0
    small = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s != 1.0 else frame_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = np.full(gray.shape, 255, np.uint8)
    for x1, y1, x2, y2 in mask_regions:
        mask[int(y1 * s):int(np.ceil(y2 * s)), int(x1 * s):int(np.ceil(x2 * s))] = 0
    kp, desc = _sift().detectAndCompute(gray, mask)
    pts = (np.array([k.pt for k in kp], np.float32).reshape(-1, 2) / s) if kp else np.zeros((0, 2), np.float32)
    if desc is None:
        desc = np.zeros((0, 128), np.float32)
    return Features(keypoints=pts, descriptors=desc, width=w, height=h, scale=s)


def plausible(H: np.ndarray, w: int, h: int) -> bool:
    """Whether H could be a turning camera's frame-to-frame map: the frame stays in front of the camera (no corner
    across the horizon), keeps its handedness and stays convex, and neither grows nor shrinks past MAX_AREA_RATIO."""
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    v = H @ corners
    if not (np.all(v[2] > 1e-9) or np.all(v[2] < -1e-9)):
        return False
    quad = (v[:2] / v[2]).T.astype(np.float32)
    if not cv2.isContourConvex(quad):
        return False
    signed = cv2.contourArea(quad, oriented=True)
    source = cv2.contourArea(corners[:2].T.astype(np.float32), oriented=True)
    if signed * source <= 0:
        return False
    ratio = abs(signed) / abs(source)
    return 1.0 / MAX_AREA_RATIO <= ratio <= MAX_AREA_RATIO


def match(a: Features, b: Features) -> Match | None:
    """The homography from a's pixels to b's, or None when the two frames do not show enough of the same gym."""
    if len(a.descriptors) < MIN_INLIERS or len(b.descriptors) < MIN_INLIERS:
        return None
    pairs = _matcher().knnMatch(a.descriptors, b.descriptors, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = a.keypoints[[g.queryIdx for g in good]]
    dst = b.keypoints[[g.trainIdx for g in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX / b.scale)
    if H is None or mask is None:
        return None
    n = int(mask.sum())
    if n < MIN_INLIERS or n / len(good) < MIN_INLIER_SHARE or not plausible(H, a.width, a.height):
        return None
    return Match(H=H, inliers=n, share=round(n / len(good), 3))


def project(points: dict[str, tuple[float, float]], H: np.ndarray) -> dict[str, tuple[float, float]]:
    """Named pixel points carried through a homography. A point may land outside the frame: it still pins the fit."""
    out = {}
    for name, (x, y) in points.items():
        v = H @ np.array([x, y, 1.0])
        if abs(v[2]) > 1e-12:
            out[name] = (float(v[0] / v[2]), float(v[1] / v[2]))
    return out
