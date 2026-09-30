from __future__ import annotations

import cv2
import numpy as np

# Where each segment sits inside a digit cell, as fractions of the cell's width and height.
SEGMENT_ZONES: dict[str, tuple[float, float, float, float]] = {
    "a": (0.25, 0.03, 0.75, 0.17),
    "b": (0.76, 0.15, 0.98, 0.47),
    "c": (0.76, 0.53, 0.98, 0.85),
    "d": (0.25, 0.83, 0.75, 0.97),
    "e": (0.02, 0.53, 0.24, 0.85),
    "f": (0.02, 0.15, 0.24, 0.47),
    "g": (0.32, 0.44, 0.68, 0.56),
}
DIGIT_PATTERNS: dict[str, int] = {
    "abcdef": 0, "bc": 1, "abdeg": 2, "abcdg": 3, "bcfg": 4,
    "acdfg": 5, "acdefg": 6, "abc": 7, "abcdefg": 8, "abcdfg": 9,
}
# OpenCV hue runs 0..180.
HUE_RANGES = {"red": [(0, 30), (150, 180)], "amber": [(13, 40)], "green": [(50, 95)]}


def led_mask(frame_bgr: np.ndarray, color: str) -> np.ndarray:
    """Boolean mask of bright, saturated pixels of the given LED colour."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (sat > 90) & (val > 140)
    if color == "any":
        return bright
    in_hue = np.zeros_like(bright)
    for lo, hi in HUE_RANGES[color]:
        in_hue |= (hue >= lo) & (hue <= hi)
    return bright & in_hue


def box_fraction(mask: np.ndarray, box) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    patch = mask[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def zone_fraction(mask: np.ndarray, cell, zone) -> float:
    x, y, w, h = cell
    zx1, zy1, zx2, zy2 = zone
    return box_fraction(mask, (x + zx1 * w, y + zy1 * h, x + zx2 * w, y + zy2 * h))


def lit_segments(mask: np.ndarray, cell, threshold: float = 0.3) -> str:
    return "".join(s for s, zone in SEGMENT_ZONES.items() if zone_fraction(mask, cell, zone) >= threshold)


def decode_cell(mask: np.ndarray, cell, threshold: float = 0.3) -> int | None:
    """The digit in a cell: None when blank, -1 when the lit pattern is not a digit."""
    lit = lit_segments(mask, cell, threshold)
    if not lit:
        return None
    return DIGIT_PATTERNS.get(lit, -1)


def decode_number(mask: np.ndarray, cells, threshold: float = 0.3) -> int | None:
    """Right-aligned digits across cells. None if blank, unreadable, or a blank follows a digit."""
    digits = [decode_cell(mask, cell, threshold) for cell in cells]
    if all(d is None for d in digits) or any(d == -1 for d in digits):
        return None
    value, started = 0, False
    for d in digits:
        if d is None:
            if started:
                return None
            continue
        started = True
        value = value * 10 + d
    return value
