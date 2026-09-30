import json
from pathlib import Path

import cv2
import numpy as np

from film.perceive.sevenseg import (
    DIGIT_PATTERNS, SEGMENT_ZONES, box_fraction, decode_cell, decode_number, led_mask, lit_segments,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"
PROFILE = Path(__file__).resolve().parents[1] / "film" / "perceive" / "profiles" / "hudl_tactical.json"
RED = (0, 0, 255)


def paint(img, cell, segments, color=RED):
    x, y, w, h = cell
    for s in segments:
        zx1, zy1, zx2, zy2 = SEGMENT_ZONES[s]
        cv2.rectangle(img, (int(x + zx1 * w), int(y + zy1 * h)), (int(x + zx2 * w), int(y + zy2 * h)), color, -1)


def test_every_digit_decodes_from_its_pattern():
    cell = (10, 10, 23, 41)
    for pattern, digit in DIGIT_PATTERNS.items():
        img = np.zeros((64, 48, 3), np.uint8)
        paint(img, cell, pattern)
        mask = led_mask(img, "red")
        assert lit_segments(mask, cell) == pattern
        assert decode_cell(mask, cell) == digit


def test_blank_and_unknown_cells():
    cell = (10, 10, 23, 41)
    blank = np.zeros((64, 48, 3), np.uint8)
    assert decode_cell(led_mask(blank, "red"), cell) is None
    odd = np.zeros((64, 48, 3), np.uint8)
    paint(odd, cell, "ag")
    assert decode_cell(led_mask(odd, "red"), cell) == -1


def test_decode_number_is_right_aligned():
    cells = [(10, 10, 23, 41), (40, 10, 23, 41), (70, 10, 23, 41)]
    img = np.zeros((64, 110, 3), np.uint8)
    paint(img, cells[1], "abcdg")   # 3
    paint(img, cells[2], "abdeg")   # 2
    mask = led_mask(img, "red")
    assert decode_number(mask, cells) == 32
    assert decode_number(led_mask(np.zeros_like(img), "red"), cells) is None
    gap = np.zeros((64, 110, 3), np.uint8)
    paint(gap, cells[0], "abcdg")   # a digit followed by a blank is not a number
    assert decode_number(led_mask(gap, "red"), cells) is None


def test_colour_masks_are_selective():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:] = (47, 160, 211)   # BGR of the amber clock digits
    assert led_mask(img, "amber").all()
    assert not led_mask(img, "red").any()
    assert not led_mask(img, "green").any()
    img[:] = (159, 213, 48)   # BGR of the green period digit
    assert led_mask(img, "green").all()


def test_real_frame_home_score_and_period():
    profile = json.loads(PROFILE.read_text())
    frame = cv2.imread(str(FIXTURES / "hudl_tactical_t1500.jpg"))
    home = profile["fields"]["home"]
    period = profile["fields"]["period"]
    assert decode_number(led_mask(frame, home["color"]), [tuple(c) for c in home["cells"]]) == 32
    assert decode_number(led_mask(frame, period["color"]), [tuple(c) for c in period["cells"]]) == 2
    clock = profile["fields"]["clock"]
    amber = led_mask(frame, clock["color"])
    assert box_fraction(amber, clock["sep_top"]) >= 0.2
    assert box_fraction(amber, clock["sep_bottom"]) >= 0.2
