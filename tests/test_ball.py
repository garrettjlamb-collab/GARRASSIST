import numpy as np

from film.perceive.ball import CROP_SIZE, crop, detect_ball_near_rim, rim_crop_box
from film.perceive.detectors import Box, FakeDetector


def test_crop_box_is_square_shifted_up_and_clamped():
    rim = Box(900, 500, 950, 530, 0.9, "rim")
    x1, y1, x2, y2 = rim_crop_box(rim, 1920, 1080)
    assert x2 - x1 == CROP_SIZE and y2 - y1 == CROP_SIZE
    assert x1 + CROP_SIZE / 2 == rim.cx
    assert y1 + CROP_SIZE / 2 < rim.cy               # more room above than below
    corner = rim_crop_box(Box(5, 5, 55, 35, 0.9, "rim"), 1920, 1080)
    assert corner[:2] == (0, 0)
    small = rim_crop_box(rim, 320, 240)
    assert small[2] - small[0] == 240 and small[3] - small[1] == 240


def test_crop_returns_the_region():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert crop(img, (10, 20, 30, 50)).shape == (30, 20, 3)


def test_detect_ball_maps_back_and_keeps_best():
    rim = Box(900, 500, 950, 530, 0.9, "rim")
    cb = rim_crop_box(rim, 1920, 1080)
    det = FakeDetector(per_call=[[Box(10, 10, 20, 20, 0.4, "ball"), Box(100, 100, 112, 112, 0.8, "ball"), Box(0, 0, 9, 9, 0.99, "rim")]])
    rows = detect_ball_near_rim(np.zeros((1080, 1920, 3), np.uint8), [(7, rim)], det, frame=12, t=1.2, model_version="ball@v1")
    assert len(rows) == 1
    r = rows[0]
    assert (r["x1"], r["y1"]) == (100 + cb[0], 100 + cb[1])
    assert r["kind"] == "ball" and r["ref_id"] == 7 and r["track_id"] is None and r["conf"] == 0.8


def test_no_ball_no_row():
    det = FakeDetector(per_call=[[]])
    assert detect_ball_near_rim(np.zeros((1080, 1920, 3), np.uint8), [(1, Box(900, 500, 950, 530, 0.9, "rim"))], det, 0, 0.0, "x") == []


def test_a_big_rim_gets_a_proportionally_bigger_crop():
    """A zoomed camera shows the rim (and the ball) bigger than the Hudl view the detector was trained on:
    the crop grows with the rim so that, shrunk to the detector's size, everything looks the size it learned."""
    from film.perceive.ball import CROP_WIDTHS

    hudl = Box(900, 500, 985, 520, 0.9, "rim")                      # 85 wide: the biggest Hudl rim box
    assert rim_crop_box(hudl, 1920, 1080)[2] - rim_crop_box(hudl, 1920, 1080)[0] == CROP_SIZE
    zoomed = Box(900, 500, 1050, 536, 0.9, "rim")                   # 150 wide
    x1, y1, x2, y2 = rim_crop_box(zoomed, 1920, 1080)
    assert x2 - x1 == y2 - y1 == round(CROP_WIDTHS * 150) and x1 + (x2 - x1) / 2 == zoomed.cx


def test_detect_ball_on_a_big_crop_shrinks_it_and_maps_back():
    rim = Box(900, 500, 1050, 536, 0.9, "rim")
    cb = rim_crop_box(rim, 1920, 1080)
    scale = (cb[2] - cb[0]) / CROP_SIZE
    seen = []

    def fn(image, i):
        seen.append(image.shape[:2])
        return [Box(100, 100, 120, 120, 0.8, "ball")]

    [r] = detect_ball_near_rim(np.zeros((1080, 1920, 3), np.uint8), [(1, rim)], FakeDetector(fn=fn), 0, 0.0, "x")
    assert seen == [(CROP_SIZE, CROP_SIZE)]
    assert abs(r["x1"] - (cb[0] + 100 * scale)) < 1e-6 and abs(r["y2"] - (cb[1] + 120 * scale)) < 1e-6
