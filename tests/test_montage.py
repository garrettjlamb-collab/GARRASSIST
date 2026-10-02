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


def test_montage_follows_a_rim_that_moves_across_the_frame(synthetic_video):
    """The Hudl camera pans, so a crop fixed at the arrival frame loses the rim within a second."""
    import pandas as pd
    from film.train.montage import rim_follower
    # The synthetic video's white bar moves 5 px a frame at 30 fps; pretend it is the rim, tracked at 10 fps.
    rows = [{"frame": f, "track_id": 1, "x1": float((f * 3 * 5) % 320), "y1": 100.0, "x2": float((f * 3 * 5) % 320 + 10), "y2": 110.0}
            for f in range(0, 30)]
    follow = rim_follower(pd.DataFrame(rows), rim_id=1, fps=10.0)
    assert follow(1.0) == (150.0, 100.0, 160.0, 110.0) and follow(99.0) is None
    first = (30.0, 100.0, 40.0, 110.0)
    followed = build_montage(synthetic_video, t=0.6, rim_box=first, frame_w=320, frame_h=240, rim_at=follow)
    fixed = build_montage(synthetic_video, t=0.6, rim_box=first, frame_w=320, frame_h=240)
    def centre_brightness(montage, i):
        r, c = divmod(i, GRID)
        tile = montage[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
        return float(tile[:, TILE // 2 - 6:TILE // 2 + 6].mean())
    assert all(centre_brightness(followed, i) > 150 for i in range(GRID * GRID))     # the bar stays centred in every tile
    assert sum(centre_brightness(fixed, i) > 150 for i in range(GRID * GRID)) <= 2   # a fixed crop loses it almost at once


def test_rim_follower_ignores_rows_without_a_box():
    """Games perceived before the smoothing fix carry a blank box on the first row of each rim."""
    import pandas as pd
    from film.train.montage import rim_follower

    rows = [{"frame": 10, "track_id": 1, "x1": float("nan"), "y1": float("nan"), "x2": float("nan"), "y2": float("nan")},
            {"frame": 12, "track_id": 1, "x1": 30.0, "y1": 100.0, "x2": 40.0, "y2": 110.0}]
    assert rim_follower(pd.DataFrame(rows), rim_id=1, fps=10.0)(1.0) == (30.0, 100.0, 40.0, 110.0)
