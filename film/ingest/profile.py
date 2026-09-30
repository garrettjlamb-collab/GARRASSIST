from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

PROFILES = ("hudl-tactical", "nfhs-auto", "nfhs-broadcast", "handheld")

# The Hudl Focus scoreboard overlay, in a 1920x1080 frame: x1, y1, x2, y2.
HUDL_OVERLAY_REGION = (0, 880, 340, 1080)
LED_MIN = 0.02
PANEL_MIN = 0.35


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


def detect_profile(frames: Sequence[np.ndarray]) -> str:
    if not frames:
        return "unknown"
    features = np.array([overlay_features(f) for f in frames])
    led, panel = np.median(features[:, 0]), np.median(features[:, 1])
    if led > LED_MIN and panel > PANEL_MIN:
        return "hudl-tactical"
    return "unknown"
