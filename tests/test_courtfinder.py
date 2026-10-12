import types

import numpy as np

from film.interpret.court import POINT_NAMES, TEMPLATE
from film.perceive.courtfinder import CourtFinder, FakeCourtFinder, points_from_result


def _result(xy, conf):
    kp = types.SimpleNamespace(xy=np.array([xy]), conf=np.array([conf]))
    boxes = types.SimpleNamespace(conf=np.array([0.9]))
    return types.SimpleNamespace(keypoints=kp, boxes=boxes)


def test_points_from_result_keeps_confident_named_points():
    xy = [[10.0 * i, 5.0 * i] for i in range(len(POINT_NAMES))]
    conf = [0.9] * len(POINT_NAMES)
    conf[3] = 0.1
    pts = points_from_result(_result(xy, conf))
    assert len(pts) == len(POINT_NAMES) - 1 and POINT_NAMES[3] not in pts and pts[POINT_NAMES[0]] == (0.0, 0.0)
    assert points_from_result(types.SimpleNamespace(keypoints=None, boxes=None)) == {}


def test_court_finder_maps_a_frame_when_enough_points_are_confident():
    names = ["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"]
    pix = {n: (TEMPLATE[n][0] * 10 + 100, 500 - TEMPLATE[n][1] * 10) for n in names}
    finder = FakeCourtFinder([pix, {}])
    m = finder.mapping(np.zeros((8, 8, 3), np.uint8))
    assert m is not None and m.points_used == 5
    x, y = m.to_half(100 + 250, 500 - 100)
    assert abs(x - 25.0) < 0.1 and abs(y - 10.0) < 0.1
    assert finder.mapping(np.zeros((8, 8, 3), np.uint8)) is None


def test_load_returns_none_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    assert CourtFinder.load("cpu") is None


def test_points_come_from_the_most_confident_instance():
    n = len(POINT_NAMES)
    xy_a = [[1.0, 1.0]] * n
    xy_b = [[2.0, 2.0]] * n
    kp = types.SimpleNamespace(xy=np.array([xy_a, xy_b]), conf=np.array([[0.9] * n, [0.9] * n]))
    boxes = types.SimpleNamespace(conf=np.array([0.2, 0.9]))
    pts = points_from_result(types.SimpleNamespace(keypoints=kp, boxes=boxes))
    assert pts[POINT_NAMES[0]] == (2.0, 2.0)


def test_missing_keypoint_confidence_keeps_every_point():
    n = len(POINT_NAMES)
    xy = [[float(i), float(i)] for i in range(n)]
    kp = types.SimpleNamespace(xy=np.array([xy]), conf=None)
    pts = points_from_result(types.SimpleNamespace(keypoints=kp, boxes=types.SimpleNamespace(conf=np.array([0.9]))))
    assert len(pts) == n


def test_wrong_point_count_is_refused():
    xy = [[0.0, 0.0]] * (len(POINT_NAMES) - 1)
    kp = types.SimpleNamespace(xy=np.array([xy]), conf=None)
    try:
        points_from_result(types.SimpleNamespace(keypoints=kp, boxes=types.SimpleNamespace(conf=np.array([0.9]))))
    except ValueError as e:
        assert "expected" in str(e)
    else:
        raise AssertionError("expected ValueError")
