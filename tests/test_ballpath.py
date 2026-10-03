import numpy as np

from film.interpret.ballpath import FEATURES, hoop_offset, path_crop_box, path_features, read_ball_path
from film.perceive.detectors import Box, FakeDetector


def _path(points, rim_w=50.0, n=25):
    """A path with one ball per listed frame: {frame index: (dx, dy, conf)}; index 3 is the arrival."""
    return {"rim_w": rim_w, "frames": [{"dt": round((i - 3) / 15, 3), "balls": [list(points[i])] if i in points else []}
                                       for i in range(n)]}


def _features(path, hoop_dx=0.0):
    return dict(zip(FEATURES, path_features(path, hoop_dx)))


def test_path_crop_box_is_a_square_that_reaches_further_below_the_rim_than_above():
    x1, y1, x2, y2 = path_crop_box((900, 500, 950, 530), 1920, 1080)
    assert (x2 - x1, y2 - y1) == (320, 320) and (x1 + x2) / 2 == 925
    assert y2 - 515 > 515 - y1
    assert path_crop_box((0, 0, 50, 30), 1920, 1080)[:2] == (0, 0)             # clamped to the frame


def test_a_ball_dropping_straight_under_the_hoop_reads_as_in_the_column():
    made = _features(_path({3: (0.0, 0.0, 0.8), 5: (0.02, 0.4, 0.7), 7: (0.0, 0.9, 0.7), 9: (-0.03, 1.3, 0.6)}))
    assert made["column_frames"] > 0 and made["depth"] > 0.2 and made["back_up"] == 0 and made["sideways"] < 0.05


def test_a_ball_bouncing_away_reads_as_sideways_and_back_up():
    missed = _features(_path({3: (0.0, 0.0, 0.8), 6: (0.5, -0.3, 0.7), 9: (0.9, -0.5, 0.7), 12: (1.3, 0.2, 0.6)}))
    assert missed["column_frames"] == 0 and missed["back_up"] > 0 and missed["sideways"] > 0.15
    assert missed["depth"] < _features(_path({3: (0.0, 0.0, 0.8), 7: (0.0, 0.9, 0.7)}))["depth"]


def test_no_ball_at_all_is_distinguishable_from_any_shot():
    empty = _features(_path({}))
    assert empty["seen"] == 0 and empty["peak_conf"] == 0 and empty["column_frames"] == 0
    assert _features(_path({3: (0.0, 0.0, 0.8)}))["seen"] > 0


def test_the_column_is_measured_from_the_hoop_not_from_the_rim_box_centre():
    """On Hudl film the 'rim' box is the top of the backboard, so the hoop sits to one side of its centre."""
    drop = {3: (0.5, 0.0, 0.8), 5: (0.5, 0.5, 0.7), 7: (0.5, 0.9, 0.7)}
    assert _features(_path(drop), hoop_dx=0.0)["column_frames"] == 0
    assert _features(_path(drop), hoop_dx=0.5)["column_frames"] > 0


def test_hoop_offset_is_the_median_of_confident_balls_at_arrival():
    paths = [_path({3: (0.4 + 0.01 * i, 0.0, 0.8), 15: (2.0, 1.0, 0.9)}) for i in range(7)]
    assert abs(hoop_offset(paths) - 0.43) < 1e-6
    assert hoop_offset(paths[:3]) == 0.0                                        # too few to trust: the rim centre
    assert hoop_offset([_path({3: (0.4, 0.0, 0.3)}) for _ in range(7)]) == 0.0  # unsure detections do not count


def test_read_ball_path_reports_balls_relative_to_the_rim_in_rim_widths(synthetic_video):
    rim = (100.0, 60.0, 150.0, 90.0)                                           # centre (125, 75), 50 wide
    x1, y1, _, _ = path_crop_box(rim, 320, 240)
    seen = []

    def fn(image, i):
        seen.append(image.shape)
        return [Box(150.0 - x1 - 5, 100.0 - y1 - 5, 150.0 - x1 + 5, 100.0 - y1 + 5, 0.6, "ball")]   # centre (150, 100)

    path = read_ball_path(synthetic_video, 1.0, rim, 320, 240, FakeDetector(fn=fn))
    assert path["rim_w"] == 50.0 and len(path["frames"]) == 25 == len(seen)
    assert abs(path["frames"][0]["dt"] + 0.2) < 0.04 and abs(path["frames"][-1]["dt"] - 1.4) < 0.04
    assert path["frames"][3]["balls"] == [[0.5, 0.5, 0.6]]


def test_read_ball_path_follows_a_moving_rim(video_factory):
    def ramp(i, img):
        img[:] = (np.arange(img.shape[1]) // 2)[None, :, None]                 # brightness tells the x position

    video = video_factory(painter=ramp)
    rim, moved = (100.0, 60.0, 150.0, 90.0), (140.0, 60.0, 190.0, 90.0)
    left_edges = []
    det = FakeDetector(fn=lambda image, i: left_edges.append(float(image[:, 0].mean())) or [])
    read_ball_path(video, 1.0, rim, 320, 240, det)
    fixed, left_edges[:] = list(left_edges), []
    read_ball_path(video, 1.0, rim, 320, 240, det, rim_at=lambda t: moved if t >= 1.5 else None)
    assert abs(left_edges[0] - fixed[0]) < 3                                    # before the rim moves: the same crop
    assert 14 < left_edges[-1] - fixed[-1] < 26                                 # after: the crop went 40 px right with it


def test_a_big_rim_is_read_at_the_detectors_size_and_in_rim_widths(video_factory):
    from film.interpret.ballpath import CROP_SIZE

    video = video_factory(size=(1280, 720))
    rim = (400.0, 200.0, 550.0, 236.0)                                           # 150 wide, centre (475, 218)
    x1, y1, x2, y2 = path_crop_box(rim, 1280, 720)
    scale = (x2 - x1) / CROP_SIZE
    assert scale > 1.5
    shapes = []

    def fn(image, i):
        shapes.append(image.shape[:2])
        cx, cy = (475 - x1) / scale, (293 - y1) / scale                         # a ball 75 px below the rim's centre
        return [Box(cx - 5, cy - 5, cx + 5, cy + 5, 0.6, "ball")]

    path = read_ball_path(video, 1.0, rim, 1280, 720, FakeDetector(fn=fn))
    assert set(shapes) == {(CROP_SIZE, CROP_SIZE)}
    assert path["frames"][3]["balls"] == [[0.0, 0.5, 0.6]]
