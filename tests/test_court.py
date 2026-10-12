import math

from film.interpret.court import END_OF_POINT, FLIP_INDEX, POINT_NAMES, TEMPLATE, fold


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
