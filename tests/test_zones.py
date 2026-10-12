# tests/test_zones.py
from film.interpret.zones import ZONE_LABELS, is_three, near_zone_line, shot_type_of, zone_of


def test_three_is_a_plain_distance_test_on_the_high_school_arc():
    assert not is_three(25.0, 24.9) and is_three(25.0, 25.1)                # 19.75 ft straight out from the basket
    assert is_three(5.0, 3.0) and not is_three(8.0, 3.0)                    # the arc meets the baseline near x = 6


def test_zones_follow_the_tracker():
    assert zone_of(25.0, 6.0) == "restricted"
    assert zone_of(25.0, 15.0) == "paint"
    assert zone_of(10.0, 10.0) == "mid-left" and zone_of(40.0, 10.0) == "mid-right" and zone_of(25.0, 22.0) == "mid-center"
    assert zone_of(3.0, 5.0) == "corner-3-left" and zone_of(47.0, 5.0) == "corner-3-right"
    assert zone_of(8.0, 20.0) == "wing-3-left" and zone_of(42.0, 20.0) == "wing-3-right"
    assert zone_of(25.0, 28.0) == "top-3"
    assert set(ZONE_LABELS) == {"restricted", "paint", "mid-left", "mid-right", "mid-center",
                                "corner-3-left", "corner-3-right", "wing-3-left", "wing-3-right", "top-3"}


def test_shot_type_follows_the_tracker():
    assert shot_type_of(25.0, 6.0) == "rim" and shot_type_of(25.0, 12.0) == "floater"
    assert shot_type_of(25.0, 18.0) == "midrange" and shot_type_of(25.0, 30.0) == "three"


def test_near_zone_line_flags_dots_close_to_the_arc_or_the_paint():
    assert near_zone_line(25.0, 24.0) and near_zone_line(17.5, 10.0)
    assert not near_zone_line(25.0, 15.0) and not near_zone_line(25.0, 35.0)
