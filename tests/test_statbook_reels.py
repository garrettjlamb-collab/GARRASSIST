from film.export.statbook import select, top_lineups
from film.interpret.phases import _score_kept
from film.interpret.run import without_frozen_score
from film.perceive.glyphs import clock_digits
from film.perceive.scoreboard import ScoreboardRead, _clock_from, repair_clock, resolve_clock


def _play(seq, type_, team="us", player="3", made=None, pts=None, on=("2", "3", "4", "13", "15"), period=1):
    return {"seq": seq, "type": type_, "team": team, "player": player, "name": None, "made": made, "pts": pts,
            "on_court": list(on), "period": period, "film_t": 100.0 + seq, "label": "x"}


PLAYS = [
    _play(1, "shot", made=True, pts=3),
    _play(2, "shot", team="them", player=None, made=True, pts=3),
    _play(3, "shot", team="them", player=None, made=False, pts=2),
    _play(4, "to", player="4", on=("0", "3", "4", "10", "11")),
    _play(5, "reb", player="3", on=("0", "3", "4", "10", "11"), period=2),
    _play(6, "period"),                                         # never a clip
]


def test_select_by_lineup_teammates_made_and_points():
    assert [p["seq"] for p in select(PLAYS, lineup="15 13 4 3 2")] == [1, 2, 3]          # order does not matter
    assert [p["seq"] for p in select(PLAYS, with_players="3,4")] == [1, 2, 3, 4, 5]
    assert [p["seq"] for p in select(PLAYS, with_players="0 10")] == [4, 5]
    assert [p["seq"] for p in select(PLAYS, team="them", made=True, pts=3)] == [2]
    assert [p["seq"] for p in select(PLAYS, period=2)] == [5]
    assert [p["seq"] for p in select(PLAYS, player="3")] == [1, 5]


def test_top_lineups_counts_plays_per_five():
    assert top_lineups(PLAYS, n=2, min_plays=1) == [("2 3 4 13 15", 3), ("0 3 4 10 11", 2)]


def test_clock_cells_from_a_value():
    assert clock_digits("7:58") == [None, 7, 5, 8]
    assert clock_digits("05:00") == [0, 5, 0, 0]
    assert _clock_from([None, 7, 5, 8]) == 478.0
    assert _clock_from([None, 7, 6, 8]) is None                    # no 68 seconds
    assert _clock_from([None, -1, 5, 8]) is None


def _r(t, clock, alts=None):
    return ScoreboardRead(t=t, period=1, clock_s=clock, home=0, away=0, home_fouls=None, away_fouls=None,
                          conf=1.0, clock_alts=alts)


def test_an_unsure_minutes_digit_takes_the_value_the_countdown_allows():
    reads = [_r(0, 268.0), _r(1, None, [88.0, 267.0]), _r(2, 266.0)]
    assert [r.clock_s for r in resolve_clock(reads)] == [268.0, 267.0, 266.0]


def test_a_minute_misread_by_more_than_one_is_repaired_right_after_a_sure_read():
    reads = [_r(0, 301.0), _r(2, 179.0), _r(40, 141.0)]              # 4:59 read as 2:59; much later, left alone
    assert [r.clock_s for r in repair_clock(reads)] == [301.0, 299.0, 141.0]


def test_a_score_that_never_moved_is_not_evidence():
    frozen = [_r(t, 400.0 - t) for t in range(5)]
    assert not _score_kept(frozen)
    assert all(r.home is None and r.away is None for r in without_frozen_score(frozen))
    kept = frozen[:3] + [ScoreboardRead(t=9, period=1, clock_s=390.0, home=2, away=0, home_fouls=None,
                                        away_fouls=None, conf=1.0)]
    assert _score_kept(kept) and without_frozen_score(kept) == kept


def test_estimated_plays_reach_a_reel_only_from_a_live_period():
    from film.export.statbook import reel_plays

    result = {"periods": [{"period": 1, "mode": "rewatch"}, {"period": 2, "mode": "live"}],
              "plays": [{"seq": 1, "period": 1, "how": "shot"}, {"seq": 2, "period": 1, "how": "fit"},
                        {"seq": 3, "period": 2, "how": "fit"}, {"seq": 4, "period": 2, "how": "follow"}]}
    assert [p["seq"] for p in reel_plays(result)] == [1, 3, 4]
