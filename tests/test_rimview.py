import numpy as np

from film.interpret.rimview import GRID, N_FRAMES, TILE, read_rim_view, split_tiles, view_square


def test_the_view_is_a_square_under_the_hoop_sized_by_the_rim():
    cx, cy, side = view_square((900.0, 500.0, 1000.0, 524.0))                 # rim box 100 wide, centre (950, 512)
    assert side == 160 and cx == 950 and cy > 512                            # 1.6 rim widths, looking below the rim
    shifted = view_square((900.0, 500.0, 1000.0, 524.0), hoop_dx=0.5)
    assert shifted[0] == 1000 and shifted[1:] == (cy, side)                   # centred on the hoop, not the box


def test_a_view_is_sixteen_frames_in_a_grid(synthetic_video):
    sheet = read_rim_view(synthetic_video, 1.0, (100.0, 60.0, 150.0, 90.0))
    assert sheet.shape == (GRID * TILE, GRID * TILE, 3) and sheet.dtype == np.uint8
    tiles = split_tiles(sheet)
    assert len(tiles) == N_FRAMES == GRID * GRID and tiles[0].shape == (TILE, TILE, 3)
    assert np.array_equal(tiles[5], sheet[TILE:2 * TILE, TILE:2 * TILE])     # row by row, in time order


def test_a_rim_at_the_edge_of_the_frame_is_padded_not_dropped(synthetic_video):
    sheet = read_rim_view(synthetic_video, 1.0, (0.0, 0.0, 50.0, 20.0))
    assert sheet.shape == (GRID * TILE, GRID * TILE, 3)


def test_a_clip_that_runs_off_the_end_of_the_film_repeats_its_last_frame(synthetic_video):
    sheet = read_rim_view(synthetic_video, 2.9, (100.0, 60.0, 150.0, 90.0))   # the film is 3 s long
    tiles = split_tiles(sheet)
    assert np.array_equal(tiles[-1], tiles[-2])


def test_the_view_follows_a_rim_that_moves(video_factory):
    def ramp(i, img):
        img[:] = (np.arange(img.shape[1]) // 2)[None, :, None]               # brightness tells the x position

    video = video_factory(painter=ramp)
    rim, moved = (100.0, 60.0, 150.0, 90.0), (160.0, 60.0, 210.0, 90.0)
    fixed = split_tiles(read_rim_view(video, 1.0, rim))
    followed = split_tiles(read_rim_view(video, 1.0, rim, rim_at=lambda t: moved if t >= 1.5 else None))
    assert abs(float(followed[0].mean()) - float(fixed[0].mean())) < 3
    assert 22 < float(followed[-1].mean()) - float(fixed[-1].mean()) < 38     # 60 px to the right is 30 brighter
