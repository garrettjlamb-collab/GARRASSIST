from pathlib import Path

import cv2
import numpy as np
import pytest

from film.perceive.sevenseg import DIGIT_PATTERNS, SEGMENT_ZONES, led_mask
from film.perceive.scoreboard import (
    RECENT_S, ScoreboardRead, clean_timeline, has_scoreboard, load_profile, load_timeline, lookup, read_clock, read_scoreboard, save_timeline,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"
EXPECTED = {
    "hudl_tactical_t60.jpg": dict(home=0, away=0, clock_s=303.0, period=1, home_fouls=0, away_fouls=0),
    "hudl_tactical_t1500.jpg": dict(home=32, away=11, clock_s=278.0, period=2, home_fouls=0, away_fouls=0),
    "hudl_tactical_t2400.jpg": dict(home=35, away=23, clock_s=150.0, period=3, home_fouls=0, away_fouls=0),
    "hudl_tactical_t4800.jpg": dict(home=0, away=0, clock_s=0.0, period=1, home_fouls=0, away_fouls=0),
}


@pytest.mark.parametrize("name,expected", EXPECTED.items())
def test_reads_real_frames(name, expected):
    profile = load_profile("hudl-tactical")
    frame = cv2.imread(str(FIXTURES / name))
    read = read_scoreboard(frame, profile, t=1.0)
    for key, value in expected.items():
        assert getattr(read, key) == value, (name, key, getattr(read, key))
    assert read.conf == 1.0


def test_has_scoreboard():
    assert has_scoreboard("hudl-tactical")
    assert not has_scoreboard("handheld")


def R(t, period=1, clock=None, home=0, away=0, conf=1.0):
    return ScoreboardRead(t=t, period=period, clock_s=clock, home=home, away=away, home_fouls=0, away_fouls=0, conf=conf)


def test_clean_drops_clock_going_up_and_score_going_down():
    reads = [R(0, clock=480), R(1, clock=479), R(2, clock=488), R(3, clock=477), R(4, clock=476, home=2), R(5, clock=475, home=0), R(6, clock=474, home=2)]
    kept = clean_timeline(reads)
    assert [r.t for r in kept] == [0, 1, 3, 4, 6]


def test_clean_drops_low_confidence_and_period_going_back():
    reads = [R(0, period=2, clock=100), R(1, period=2, clock=99, conf=0.25), R(2, period=1, clock=98), R(3, period=2, clock=97)]
    assert [r.t for r in clean_timeline(reads)] == [0, 3]


def test_clean_accepts_big_jump_after_a_gap():
    reads = [R(0, clock=100, home=10), R(30, clock=70, home=18)]
    assert len(clean_timeline(reads)) == 2


def test_lookup_and_roundtrip(tmp_path):
    reads = [R(0, clock=100), R(1, clock=99), R(2, clock=98)]
    assert lookup(reads, 1.5).t == 1
    assert lookup(reads, -1) is None
    assert lookup([], 5) is None
    path = tmp_path / "scoreboard.jsonl"
    save_timeline(reads, path)
    assert load_timeline(path) == reads


def _kept_times(reads):
    return [r.t for r in clean_timeline(reads)]


def test_score_rise_of_four_within_window_is_dropped():
    assert _kept_times([R(0, clock=100, home=10), R(1, clock=99, home=14)]) == [0]


def test_score_rise_of_three_within_window_is_kept():
    assert _kept_times([R(0, clock=100, home=10), R(1, clock=99, home=13)]) == [0, 1]


def test_score_rise_of_eight_after_a_gap_is_kept():
    assert _kept_times([R(0, clock=100, home=10), R(RECENT_S + 1, clock=90, home=18)]) == [0, RECENT_S + 1]


def test_score_fall_within_window_is_dropped():
    assert _kept_times([R(0, clock=100, home=10), R(1, clock=99, home=8)]) == [0]


def test_score_fall_after_a_gap_is_kept():
    assert _kept_times([R(0, clock=100, home=10), R(RECENT_S + 1, clock=90, home=8)]) == [0, RECENT_S + 1]


def test_clock_rise_of_under_half_a_second_is_kept():
    assert _kept_times([R(0, clock=100.0), R(1, clock=100.4)]) == [0, 1]


def test_clock_rise_of_over_half_a_second_is_dropped():
    assert _kept_times([R(0, clock=100.0), R(1, clock=100.6)]) == [0]


def test_clock_rise_after_a_gap_is_kept():
    assert _kept_times([R(0, clock=100.0), R(RECENT_S + 1, clock=150.0)]) == [0, RECENT_S + 1]


def test_violation_exactly_recent_s_after_last_kept_is_dropped():
    assert _kept_times([R(0, clock=100, home=10), R(RECENT_S, clock=90, home=20)]) == [0]


def test_violation_just_past_recent_s_is_kept():
    t = RECENT_S + 0.5
    assert _kept_times([R(0, clock=100, home=10), R(t, clock=90, home=20)]) == [0, t]


def test_a_misread_period_does_not_poison_the_timeline():
    reads = [R(0, period=1, clock=500), R(1, period=4, clock=499, conf=0.75)]
    reads += [R(t, period=1, clock=500 - t) for t in range(2, 15)]
    kept = _kept_times(reads)
    assert 1 in kept                                     # nothing yet contradicts the bad read
    assert not any(2 <= t <= 11 for t in kept)           # within 10 s of it, period 1 reads are dropped
    assert [t for t in kept if t >= 12] == [12, 13, 14]  # then the timeline recovers


AMBER = (47, 160, 211)
CLOCK_FIELD = {
    "cells": [[5, 5, 23, 41], [30, 5, 23, 41], [70, 5, 23, 41], [95, 5, 23, 41]],
    "sep_top": [56, 10, 65, 17],
    "sep_bottom": [57, 34, 64, 38],
}


def _clock_mask(digits, top, bottom):
    img = np.zeros((60, 130, 3), np.uint8)
    for d, cell in zip(digits, CLOCK_FIELD["cells"]):
        if d is None:
            continue
        pattern = d if isinstance(d, str) else next(p for p, v in DIGIT_PATTERNS.items() if v == d)
        x, y, w, h = cell
        for s in pattern:
            zx1, zy1, zx2, zy2 = SEGMENT_ZONES[s]
            cv2.rectangle(img, (int(x + zx1 * w), int(y + zy1 * h)), (int(x + zx2 * w), int(y + zy2 * h)), AMBER, -1)
    for on, box in ((top, CLOCK_FIELD["sep_top"]), (bottom, CLOCK_FIELD["sep_bottom"])):
        if on:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), AMBER, -1)
    return led_mask(img, "amber")


def test_read_clock_minutes_and_seconds_with_both_dots():
    assert read_clock(_clock_mask([0, 4, 3, 8], True, True), CLOCK_FIELD) == 278.0


def test_read_clock_seconds_and_tenths_with_bottom_dot_only():
    assert read_clock(_clock_mask([5, 9, 9, None], False, True), CLOCK_FIELD) == pytest.approx(59.9)


def test_read_clock_with_no_dots_is_none():
    assert read_clock(_clock_mask([0, 4, 3, 8], False, False), CLOCK_FIELD) is None


def test_read_clock_with_an_unrecognised_digit_is_none():
    assert read_clock(_clock_mask([0, 4, "ab", 8], True, True), CLOCK_FIELD) is None


def test_a_profile_may_have_no_clock_and_its_own_lit_threshold():
    """The Marcos de Niza Hudl skin: LED-dot digits that fill a segment zone only sparsely, and a clock font
    the segment zones cannot read. Scores and period are enough for made-or-missed."""
    import cv2
    from film.perceive.scoreboard import load_profile, read_scoreboard

    frame = cv2.imread("fixtures/frames/hudl_mdn_40_71_p4.jpg")
    read = read_scoreboard(frame, load_profile("hudl-mdn"), t=3984.0)
    assert (read.home, read.away, read.period) == (40, 71, 4)
    assert read.clock_s is None and read.home_fouls is None and read.conf == 0.75


def test_every_profile_file_is_a_choice_for_ingest():
    from film.ingest.profile import PROFILES

    assert "hudl-mdn" in PROFILES and "hudl-tactical" in PROFILES and "handheld" in PROFILES
