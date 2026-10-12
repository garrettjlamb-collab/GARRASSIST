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
