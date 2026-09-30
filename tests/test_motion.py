import numpy as np

from film.perceive.detectors import Box
from film.perceive.motion import MotionEnergy, zone_box


def test_zone_box_is_three_rim_widths():
    rim = Box(900, 500, 950, 530, 0.9, "rim")
    x1, y1, x2, y2 = zone_box(rim, 1920, 1080)
    assert x2 - x1 == 150 and y2 - y1 == 150
    assert (x1 + x2) / 2 == rim.cx and (y1 + y2) / 2 == rim.cy


def test_energy_is_zero_when_static_and_positive_when_something_moves_at_the_rim():
    rim = Box(900, 500, 950, 530, 0.9, "rim")
    base = np.full((1080, 1920, 3), 40, np.uint8)
    me = MotionEnergy()
    first = me.update(base, [(1, rim)], 0, 0.0)
    assert first[0]["energy"] == 0.0 and first[0]["energy_global"] == 0.0
    second = me.update(base.copy(), [(1, rim)], 1, 0.1)
    assert second[0]["energy"] == 0.0
    moved = base.copy()
    moved[502:527, 912:937] = 255                    # a bright blob at the rim
    third = me.update(moved, [(1, rim)], 2, 0.2)
    assert third[0]["energy"] > 5.0
    assert third[0]["energy"] > third[0]["energy_global"]
