import numpy as np

from film.train.montage import GRID, TILE, build_montage, montage_box, montage_from_tiles


def test_montage_box_is_three_widths_and_looks_below_the_rim():
    x1, y1, x2, y2 = montage_box((900, 500, 950, 530), 1920, 1080)
    assert x2 - x1 == 150 and y2 - y1 == 150
    assert (x1 + x2) / 2 == 925
    assert (y1 + y2) / 2 > 515


def test_montage_from_tiles_pads_by_repeating_the_last_tile():
    tiles = [np.full((TILE, TILE, 3), i, np.uint8) for i in range(4)]
    m = montage_from_tiles(tiles)
    assert m.shape == (TILE * GRID, TILE * GRID, 3)
    assert m[0, 0, 0] == 0 and m[0, TILE, 0] == 1 and m[TILE, 0, 0] == 3 and m[2 * TILE, 2 * TILE, 0] == 3


def test_build_montage_from_video(synthetic_video):
    m = build_montage(synthetic_video, t=1.0, rim_box=(100, 60, 150, 90), frame_w=320, frame_h=240)
    assert m.shape == (TILE * GRID, TILE * GRID, 3) and m.dtype == np.uint8
