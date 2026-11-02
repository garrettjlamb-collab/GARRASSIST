import numpy as np
import pytest

from film.interpret.camera import from_homography
from film.interpret.flight3d import G, Path3D, fit_path, rims_of
from film.interpret.court import Mapping
from film.perceive.court_run import mapping_between

W, H = 1920, 1080
F = 1400.0
EYE = np.array([25.0, -30.0, 24.0])        # a camera past the near baseline, up in the seats, looking at the near end


def _camera_matrix():
    forward = np.array([25.0, 15.0, 0.0]) - EYE
    z = forward / np.linalg.norm(forward)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ EYE
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    return K @ np.column_stack([R, t])


P = _camera_matrix()
FLOOR_H = np.linalg.inv(P[:, [0, 1, 3]])   # pixels -> feet


def _pix(x, y, z):
    v = P @ np.array([x, y, z, 1.0])
    return v[0] / v[2], v[1] / v[2]


def _shot(start=(10.0, 25.0), release_z=7.0, rim=(25.0, 5.25, 10.0), T=1.2, ta=20.0):
    """A ball from `start` to the rim, as (t, x, y, conf) sightings at 10 per second."""
    sx, sy = start
    vx, vy = (rim[0] - sx) / T, (rim[1] - sy) / T
    vz = (rim[2] - release_z + 0.5 * G * T * T) / T
    out = []
    for i in range(int(T * 10) + 1):
        tau = i / 10.0
        px, py = _pix(sx + vx * tau, sy + vy * tau, release_z + vz * tau - 0.5 * G * tau * tau)
        out.append((ta - T + tau, px, py, 0.6))
    return out


def test_the_camera_comes_back_out_of_the_floor_homography():
    cam = from_homography(FLOOR_H, W, H)
    assert abs(cam.focal - F) / F < 0.01
    assert np.linalg.norm(cam.centre - EYE) < 0.5
    got = cam.project((25.0, 5.25, 10.0))
    want = _pix(25.0, 5.25, 10.0)
    assert np.hypot(got[0] - want[0], got[1] - want[1]) < 1.0


def test_a_whole_flight_gives_the_spot_it_came_from():
    cam = from_homography(FLOOR_H, W, H)
    path = fit_path(cam, _shot(), rims_of(94.0), 19.6, 20.4, 94.0, release_z=7.0)
    assert isinstance(path, Path3D)
    assert np.hypot(path.spot[0] - 10.0, path.spot[1] - 25.0) < 1.0
    assert abs(path.ta - 20.0) < 0.1


def test_only_the_end_of_a_flight_still_says_where_it_came_from():
    cam = from_homography(FLOOR_H, W, H)
    late = [s for s in _shot() if s[0] >= 19.4]
    path = fit_path(cam, late, rims_of(94.0), 19.6, 20.4, 94.0, release_z=7.0)
    assert path is not None and np.hypot(path.spot[0] - 10.0, path.spot[1] - 25.0) < 3.0


def test_stray_boxes_do_not_join_the_path():
    cam = from_homography(FLOOR_H, W, H)
    noise = [(19.0 + i * 0.13, 200.0 + 120 * i, 900.0 - 30 * i, 0.3) for i in range(8)]
    path = fit_path(cam, _shot() + noise, rims_of(94.0), 19.6, 20.4, 94.0, release_z=7.0)
    assert path.points == len(_shot()) and np.hypot(path.spot[0] - 10.0, path.spot[1] - 25.0) < 1.5


def test_a_ball_that_is_not_a_shot_has_no_path():
    cam = from_homography(FLOOR_H, W, H)
    along_floor = [(19.0 + i * 0.1, 300.0 + 60 * i, 800.0, 0.5) for i in range(10)]
    assert fit_path(cam, along_floor, rims_of(94.0), 19.6, 20.4, 94.0) is None


def _map(H_):
    return Mapping(H=H_, points_used=8, error_px=1.0, conf=1.0, end="near", length=94.0)


def test_mappings_blend_between_fits_whatever_their_sign():
    a = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    b = np.array([[1.0, 0, -10.0], [0, 1.0, 0], [0, 0, 1.0]])
    got = mapping_between([(0.0, _map(a)), (1.0, _map(-b))], 0.5)
    v = got.H @ np.array([10.0, 0.0, 1.0])
    assert abs(v[0] / v[2] - 5.0) < 0.3


def test_mappings_far_apart_are_not_blended():
    a = np.eye(3)
    got = mapping_between([(0.0, _map(a)), (10.0, _map(a * 2))], 2.0)
    assert np.allclose(got.H, a)


def test_a_low_camera_with_the_horizon_low_in_the_picture_still_reads_the_floor_after_blending():
    # A camera 5 ft up, level: the horizon is the picture's middle row at 540, so the upper part of the picture is above it.
    eye, f = np.array([25.0, -20.0, 5.0]), 1400.0
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    K = np.array([[f, 0, 960.0], [0, f, 540.0], [0, 0, 1.0]])
    P = K @ np.column_stack([R, -R @ eye])
    H = np.linalg.inv(P[:, [0, 1, 3]])
    v = P @ np.array([25.0, 10.0, 0.0, 1.0])
    px, py = v[0] / v[2], v[1] / v[2]
    fixed = _map(H if (H @ np.array([px, py, 1.0]))[2] > 0 else -H)
    got = mapping_between([(0.0, fixed), (1.0, fixed)], 0.5)
    at = got.to_court(px, py)
    assert at is not None and abs(at[0] - 25.0) < 0.1 and abs(at[1] - 10.0) < 0.1
