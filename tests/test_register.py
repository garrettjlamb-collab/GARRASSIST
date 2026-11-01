import numpy as np
import pytest

from film.perceive.register import features, mask_regions, match, plausible, project
from synth_court import apply, crop, frame, panorama


@pytest.fixture(scope="module")
def pano():
    return panorama()


def test_two_overlapping_frames_line_up_to_a_pixel(pano):
    G1, G2 = crop(500.0), crop(820.0, 170.0, zoom=1.15)
    m = match(features(frame(pano, G1)), features(frame(pano, G2)))
    assert m is not None and m.inliers >= 40
    truth = G2 @ np.linalg.inv(G1)
    for x, y in [(400, 100), (600, 300), (450, 200), (550, 50)]:            # the part of frame 1 that frame 2 sees
        got, want = apply(m.H, x, y), apply(truth, x, y)
        assert np.hypot(got[0] - want[0], got[1] - want[1]) < 1.5


def test_frames_of_different_places_do_not_line_up(pano):
    other = panorama(seed=99)
    assert match(features(frame(pano, crop(300.0))), features(frame(other, crop(300.0)))) is None
    assert match(features(frame(pano, crop(0.0))), features(frame(pano, crop(1900.0)))) is None   # no overlap


def test_masked_regions_give_no_features(pano):
    f = features(frame(pano, crop(400.0)), mask_regions=[(0, 200, 200, 360)])
    assert len(f.keypoints) > 100
    inside = (f.keypoints[:, 0] < 200) & (f.keypoints[:, 1] > 200)
    assert not inside.any()


def test_big_frames_are_matched_at_work_width_and_answered_in_full_pixels(pano):
    G1, G2 = crop(500.0, zoom=3.0), crop(560.0, 160.0, zoom=3.0)
    import cv2
    big1, big2 = (cv2.warpPerspective(pano, G, (1920, 1080)) for G in (G1, G2))
    a, b = features(big1), features(big2)
    assert a.scale == pytest.approx(0.5) and a.keypoints[:, 0].max() > 1000
    m = match(a, b)
    assert m is not None
    got, want = apply(m.H, 1500, 700), apply(G2 @ np.linalg.inv(G1), 1500, 700)
    assert np.hypot(got[0] - want[0], got[1] - want[1]) < 3.0


def test_plausible_refuses_a_collapse_and_a_mirror():
    assert plausible(np.eye(3), 640, 360)
    assert not plausible(np.diag([0.05, 0.05, 1.0]), 640, 360)                 # the frame shrinks to a speck
    assert not plausible(np.diag([-1.0, 1.0, 1.0]), 640, 360)                  # seen in a mirror
    assert not plausible(np.array([[1, 0, 0], [0, 1, 0], [-0.01, 0, 1.0]]), 640, 360)  # the horizon crosses the frame


def test_project_carries_named_points():
    H = np.array([[1.0, 0, 10], [0, 1.0, -5], [0, 0, 1]])
    assert project({"a": (1.0, 2.0)}, H) == {"a": (11.0, -3.0)}


def test_hudl_profiles_mask_the_burned_in_scoreboard():
    assert mask_regions("hudl-mdn") and mask_regions("hudl-tactical")
    assert mask_regions("handheld") == []
