"""The court as the tracker draws it, and the eighteen points the court finder looks for.

Coordinates are the tracker's: 50 ft wide (x), 47 ft deep (y) per half, baseline at y = 0, basket at
(25, 5.25). The full court is 94 ft; the far end is the near end rotated about centre court."""
from __future__ import annotations

import math

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
