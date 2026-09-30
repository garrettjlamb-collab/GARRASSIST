from __future__ import annotations

from importlib import resources
from typing import Sequence

import cv2
import numpy as np

PROFILES = ("hudl-tactical", "nfhs-auto", "nfhs-broadcast", "handheld")

# The Hudl Focus scoreboard overlay, in a 1920x1080 frame: x1, y1, x2, y2.
HUDL_OVERLAY_REGION = (0, 880, 340, 1080)
PANEL_MIN = 0.35
CORR_MIN = 0.3

_template: np.ndarray | None = None


def _overlay_template() -> np.ndarray:
    global _template
    if _template is None:
        data = resources.files("film.ingest").joinpath("hudl_overlay.png").read_bytes()
        _template = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return _template


def overlay_features(frame_bgr: np.ndarray, region=HUDL_OVERLAY_REGION) -> tuple[float, float]:
    """(fraction of lit LED-coloured pixels, fraction of dark-blue panel pixels) inside the region."""
    x1, y1, x2, y2 = region
    h, w = frame_bgr.shape[:2]
    if h < y2 or w < x2:
        return 0.0, 0.0
    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    led = (sat > 90) & (val > 140) & ((hue <= 40) | (hue >= 165) | ((hue >= 50) & (hue <= 95)))
    panel = (hue >= 100) & (hue <= 130) & (sat > 100) & (val > 50) & (val < 235)
    return float(led.mean()), float(panel.mean())


def overlay_correlation(frame_bgr: np.ndarray, region=HUDL_OVERLAY_REGION) -> float:
    """Pearson correlation of the region's grayscale pixels with the Hudl overlay template."""
    x1, y1, x2, y2 = region
    h, w = frame_bgr.shape[:2]
    if h < y2 or w < x2:
        return 0.0
    crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    tmpl = _overlay_template().astype(np.float32).ravel()
    if crop.shape != tmpl.shape or crop.std() == 0 or tmpl.std() == 0:
        return 0.0
    return float(np.corrcoef(crop, tmpl)[0, 1])


def detect_profile(frames: Sequence[np.ndarray]) -> str:
    if not frames:
        return "unknown"
    corr = np.median([overlay_correlation(f) for f in frames])
    panel = np.median([overlay_features(f)[1] for f in frames])
    if corr > CORR_MIN and panel > PANEL_MIN:
        return "hudl-tactical"
    return "unknown"
