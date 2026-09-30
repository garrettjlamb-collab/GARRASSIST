from pathlib import Path

import cv2
import pytest

from film.perceive.scoreboard import (
    ScoreboardRead, clean_timeline, has_scoreboard, load_profile, load_timeline, lookup, read_scoreboard, save_timeline,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"
EXPECTED = {
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
