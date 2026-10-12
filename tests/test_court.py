import math

import numpy as np

from film.interpret.court import END_OF_POINT, FLIP_INDEX, POINT_NAMES, TEMPLATE, Mapping, fit_mapping, fold


def test_eighteen_named_points_nine_per_end():
    assert len(POINT_NAMES) == 18 and len(set(POINT_NAMES)) == 18
    assert sum(END_OF_POINT[n] == "near" for n in POINT_NAMES) == 9
    assert all(n in TEMPLATE for n in POINT_NAMES)


def test_template_is_the_tracker_court_with_the_high_school_arc():
    assert TEMPLATE["near-corner-left"] == (0.0, 0.0) and TEMPLATE["near-corner-right"] == (50.0, 0.0)
    assert TEMPLATE["near-lane-left"] == (17.0, 0.0) and TEMPLATE["near-ft-right"] == (33.0, 19.0)
    assert TEMPLATE["near-ft-top"] == (25.0, 25.0)
    x, y = TEMPLATE["near-arc-left"]
    assert y == 0.0 and math.isclose(math.hypot(x - 25.0, y - 5.25), 19.75, abs_tol=1e-6) and x < 25
    fx, fy = TEMPLATE["far-corner-left"]
    assert (fx, fy) == (50.0, 94.0)                         # the far end is the near end rotated about centre court


def test_fold_maps_the_far_end_onto_the_half_court():
    assert fold(10.0, 5.0) == (10.0, 5.0)
    assert fold(40.0, 89.0) == (10.0, 5.0)
    assert fold(25.0, 47.0) == (25.0, 47.0)


def test_flip_index_pairs_left_with_right():
    for i, name in enumerate(POINT_NAMES):
        partner = POINT_NAMES[FLIP_INDEX[i]]
        assert FLIP_INDEX[FLIP_INDEX[i]] == i
        if "left" in name:
            assert partner == name.replace("left", "right")
        elif "right" in name:
            assert partner == name.replace("right", "left")
        else:
            assert partner == name


def _camera(scale=12.0, shear=0.3, ox=300.0, oy=100.0):
    """A made-up camera: feet -> pixels with perspective, for synthetic tests. Pixel x is mirrored so the court is
    seen the right way round (as from the far end, near baseline at the top), which fit_mapping's orientation guard needs."""
    H = np.array([[scale, shear * scale, ox], [0.0, scale * 0.6, oy], [0.0, 0.004, 1.0]])

    def project(x, y):
        v = H @ np.array([x, y, 1.0])
        return (float(1000.0 - v[0] / v[2]), float(v[1] / v[2]))

    return project


def test_fit_recovers_the_mapping_from_five_points():
    project = _camera()
    names = ["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"]
    pts = {n: project(*TEMPLATE[n]) for n in names}
    m = fit_mapping(pts)
    assert isinstance(m, Mapping) and m.points_used == 5 and m.error_px < 0.5 and m.conf > 0.9 and m.end == "near"
    x, y = m.to_court(*project(10.0, 12.0))
    assert abs(x - 10.0) < 0.1 and abs(y - 12.0) < 0.1


def test_fit_needs_four_points_and_drops_an_outlier():
    project = _camera()
    names = ["near-corner-left", "near-corner-right", "near-ft-left"]
    assert fit_mapping({n: project(*TEMPLATE[n]) for n in names}) is None
    names += ["near-ft-right", "near-ft-top", "near-lane-left"]
    pts = {n: project(*TEMPLATE[n]) for n in names}
    pts["near-lane-left"] = (pts["near-lane-left"][0] + 80.0, pts["near-lane-left"][1])      # a bad click
    m = fit_mapping(pts)
    assert m is not None and m.points_used == 5 and m.error_px < 1.0


def test_far_end_points_fold_to_the_half_court():
    project = _camera()
    names = ["far-corner-left", "far-corner-right", "far-ft-left", "far-ft-right", "far-ft-top"]
    m = fit_mapping({n: project(*TEMPLATE[n]) for n in names})
    assert m.end == "far"
    x, y = m.to_half(*project(40.0, 89.0))
    assert abs(x - 10.0) < 0.2 and abs(y - 5.0) < 0.2


def test_contains_tells_court_from_bench():
    project = _camera()
    names = ["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"]
    m = fit_mapping({n: project(*TEMPLATE[n]) for n in names})
    assert m.contains(*project(25.0, 20.0)) and not m.contains(*project(-6.0, 20.0))
    assert m.contains(*project(-1.0, 20.0))                                  # within the 2 ft margin


def test_the_tolerance_is_in_pixels_not_feet():
    project = _camera()
    names = ["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top", "near-lane-left"]
    pts = {n: project(*TEMPLATE[n]) for n in names}
    px, py = pts["near-lane-left"]
    pts["near-lane-left"] = (px + 20.0, py)                  # about 1.7 ft at this scale: inside 8 ft, outside 8 px
    m = fit_mapping(pts)
    assert m is not None and m.points_used == 5 and m.error_px < 1.0


def test_degenerate_points_return_none_instead_of_raising():
    names = ["near-corner-left", "near-corner-right", "near-ft-left", "near-ft-right", "near-ft-top"]
    collinear = {n: (TEMPLATE[n][0] * 10.0, 500.0) for n in names}      # every point on one pixel row
    assert fit_mapping(collinear) is None


def _pinhole(eye=(-25.0, 47.0, 20.0), target=(25.0, 47.0, 0.0), focal=900.0, size=(1920, 1080)):
    """A real camera above the floor: here up in the stands beyond the x = 0 sideline, looking at centre court."""
    eye, target = np.array(eye), np.array(target)
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
    d = np.cross(f, r)                                         # image y grows downward
    R = np.array([r, d, f])

    def project(x, y):
        c = R @ (np.array([x, y, 0.0]) - eye)
        return (float(focal * c[0] / c[2] + size[0] / 2), float(focal * c[1] / c[2] + size[1] / 2))

    return project, (size[0] / 2, size[1] / 2 - focal * 2.0)  # a pixel well above the horizon


def test_a_mirrored_labelling_is_refused():
    project, _ = _pinhole()
    pts = {n: project(*TEMPLATE[n]) for n in POINT_NAMES}
    m = fit_mapping(pts)
    assert m is not None and m.points_used == 18
    x, y = m.to_court(*project(12.0, 30.0))
    assert abs(x - 12.0) < 0.1 and abs(y - 30.0) < 0.1
    swapped = {POINT_NAMES[FLIP_INDEX[i]]: pts[n] for i, n in enumerate(POINT_NAMES)}     # left and right names swapped
    assert fit_mapping(swapped) is None


def test_a_point_above_the_horizon_has_no_court_position():
    project, sky = _pinhole()
    m = fit_mapping({n: project(*TEMPLATE[n]) for n in POINT_NAMES})
    assert m.to_court(*sky) is None and m.to_half(*sky) is None and not m.contains(*sky)
    w_negative = Mapping(H=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.01, 1.0]]), points_used=4, error_px=0.0,
                         conf=1.0, end="near")
    assert w_negative.to_court(0.0, 200.0) is None and w_negative.to_court(0.0, 100.0) is None
    assert w_negative.to_court(0.0, 50.0) is not None
