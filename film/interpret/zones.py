"""The shot tracker's zone and shot-type rules (coaching-sim-web/public/shot-tracker/index.html, getZone and
suggestType), with the high-school arc: a plain 19.75 ft distance test, and a three within 10 ft of the
baseline is a corner three. Film shots classify exactly as hand-charted ones will once the tracker moves to
the same arc."""
from __future__ import annotations

import math

from film.interpret.court import BASKET, LANE_LEFT, LANE_RIGHT, THREE_RADIUS

RESTRICTED_RADIUS = 4.0
PAINT_TOP = 19.0
CORNER_Y = 10.0
FLOATER_RADIUS = 9.0

ZONE_LABELS = {
    "restricted": "Restricted Area", "paint": "Paint", "mid-left": "Mid-Range Left",
    "mid-right": "Mid-Range Right", "mid-center": "Mid-Range Center",
    "corner-3-left": "Left Corner 3", "corner-3-right": "Right Corner 3",
    "wing-3-left": "Left Wing 3", "wing-3-right": "Right Wing 3", "top-3": "Top of Key 3",
}


def _polar(x: float, y: float) -> tuple[float, float]:
    dx, dy = x - BASKET[0], y - BASKET[1]
    return math.hypot(dx, dy), math.degrees(math.atan2(dx, dy))


def is_three(x: float, y: float) -> bool:
    return _polar(x, y)[0] > THREE_RADIUS


def zone_of(x: float, y: float) -> str:
    dist, ang = _polar(x, y)
    if dist <= RESTRICTED_RADIUS:
        return "restricted"
    if is_three(x, y):
        if y < CORNER_Y:
            return "corner-3-left" if x < BASKET[0] else "corner-3-right"
        if ang < -35:
            return "wing-3-left"
        if ang > 35:
            return "wing-3-right"
        return "top-3"
    if LANE_LEFT <= x <= LANE_RIGHT and y <= PAINT_TOP:
        return "paint"
    if ang < -25:
        return "mid-left"
    if ang > 25:
        return "mid-right"
    return "mid-center"


def shot_type_of(x: float, y: float) -> str:
    dist, _ = _polar(x, y)
    if dist <= RESTRICTED_RADIUS:
        return "rim"
    if is_three(x, y):
        return "three"
    if dist <= FLOATER_RADIUS and y <= PAINT_TOP:
        return "floater"
    return "midrange"


def near_zone_line(x: float, y: float, feet: float = 1.5) -> bool:
    """Within `feet` of the arc or the paint's edges: the zone could go either way."""
    dist, _ = _polar(x, y)
    if abs(dist - THREE_RADIUS) <= feet:
        return True
    if y <= PAINT_TOP + feet and (abs(x - LANE_LEFT) <= feet or abs(x - LANE_RIGHT) <= feet):
        return True
    return LANE_LEFT - feet <= x <= LANE_RIGHT + feet and abs(y - PAINT_TOP) <= feet
