from film.interpret.arrivals import Arrival
from film.interpret.events import (
    Event, build_shot_events, event_id, load_events, save_events, score_change, scoreboard_only_events,
)
from film.perceive.scoreboard import ScoreboardRead

RIM = (900.0, 500.0, 950.0, 530.0)


def R(t, home=0, away=0, period=2, clock=300.0):
    return ScoreboardRead(t=t, period=period, clock_s=clock - t, home=home, away=away, home_fouls=0, away_fouls=0, conf=1.0)


def timeline_with_home_score_at(t_score, points=2, length=60):
    return [R(t, home=(points if t >= t_score else 0)) for t in range(length)]


def A(t, source="ball", rim_id=1):
    return Arrival(t=t, frame=int(t * 10), rim_id=rim_id, rim_box=RIM, source=source, conf=0.8 if source == "ball" else 0.4)


def test_event_id_format():
    assert event_id(1500.44, 2) == "shot-015004-r2"


def test_score_change_window():
    tl = timeline_with_home_score_at(14)
    assert score_change(tl, 10.0, 6.0) == (2, 0)
    assert score_change(tl, 10.0, 2.0) == (0, 0)
    assert score_change([], 10.0, 6.0) is None


def test_agreement_gives_high_confidence_and_team():
    tl = timeline_with_home_score_at(13)
    [e] = build_shot_events([A(10.0)], tl, [("made", 0.97)], "g")
    assert e.type == "shot" and e.made is True and e.team == "home" and e.confidence == 0.95 and e.flags == []
    assert e.quarter == 2 and e.clock_s == 290.0
    assert e.clip == {"start": 2.0, "end": 13.0}      # 8 s of build-up, 3 s for the finish
    assert e.signals["score_delta"] == [2, 0] and e.signals["points"] == 2
    [m] = build_shot_events([A(30.0)], tl, [("missed", 0.9)], "g")
    assert m.made is False and m.team is None and m.confidence == 0.9


def test_disagreement_takes_the_scoreboard_and_flags():
    tl = timeline_with_home_score_at(13)
    [e] = build_shot_events([A(10.0)], tl, [("missed", 0.8)], "g")
    assert e.made is True and e.confidence == 0.5 and "clf-scoreboard-disagree" in e.flags
    [e2] = build_shot_events([A(30.0)], tl, [("made", 0.8)], "g")
    assert e2.made is False and e2.confidence == 0.5


def test_without_scoreboard_or_classifier():
    [e] = build_shot_events([A(10.0)], [], [("made", 0.8)], "g")
    assert e.made is True and e.confidence == 0.56 and "no-scoreboard" in e.flags and e.quarter is None
    [f] = build_shot_events([A(10.0)], timeline_with_home_score_at(13), [None], "g")
    assert f.made is True and f.confidence == 0.6 and "no-classifier" in f.flags
    [g] = build_shot_events([A(10.0)], [], [None], "g")
    assert g.made is None and g.confidence == 0.3


def test_motion_only_and_none_class():
    tl = timeline_with_home_score_at(13)
    [e] = build_shot_events([A(10.0, source="motion")], tl, [("made", 0.9)], "g")
    assert e.confidence == 0.5 and "motion-only" in e.flags
    [u] = build_shot_events([A(10.0, source="motion")], tl, [("none", 0.9)], "g")
    assert u.type == "unknown" and u.made is None
    [s] = build_shot_events([A(10.0)], tl, [("none", 0.9)], "g")
    assert s.type == "shot" and s.made is True and s.confidence == 0.4 and "clf-none" in s.flags


def test_free_throw_flag_and_putback_window():
    tl = [R(t, home=(1 if t >= 13 else 0)) for t in range(60)]
    [e] = build_shot_events([A(10.0)], tl, [("made", 0.9)], "g")
    assert "free-throw?" in e.flags and e.signals["points"] == 1
    tl2 = timeline_with_home_score_at(15)
    miss, putback = build_shot_events([A(10.0), A(12.0)], tl2, [("missed", 0.9), ("made", 0.9)], "g")
    assert miss.made is False and miss.confidence == 0.9
    assert putback.made is True and putback.confidence == 0.95


def test_save_and_load_roundtrip(tmp_path):
    events = build_shot_events([A(10.0)], timeline_with_home_score_at(13), [("made", 0.9)], "g")
    save_events(tmp_path, events)
    assert load_events(tmp_path) == events
    assert load_events(tmp_path / "missing") == []


def test_ambiguous_scoreboard_is_no_evidence_not_a_miss():
    both = [R(t, home=(12 if t >= 13 else 10), away=(10 if t >= 13 else 8)) for t in range(60)]
    [a] = build_shot_events([A(10.0)], both, [("made", 0.97)], "g")
    assert a.made is True and a.confidence == round(0.7 * 0.97, 2) and a.team is None
    assert "scoreboard-ambiguous" in a.flags and "clf-scoreboard-disagree" not in a.flags
    assert a.signals["scoreboard"] == "ambiguous"
    [b] = build_shot_events([A(10.0)], timeline_with_home_score_at(13, points=5), [("missed", 0.9)], "g")
    assert b.made is False and b.confidence == round(0.7 * 0.9, 2) and "scoreboard-ambiguous" in b.flags
    [c] = build_shot_events([A(10.0)], both, [None], "g")
    assert c.made is None and c.confidence == 0.3
    assert "no-classifier" in c.flags and "scoreboard-ambiguous" in c.flags


def test_event_from_json_ignores_unknown_keys():
    e = Event(id="shot-000010-r1", type="shot", t=1.0, rim_id=1, quarter=1, clock_s=400.0, team=None, made=None,
              confidence=0.3, clip={"start": 0.0, "end": 2.5}, signals={})
    assert Event.from_json(e.to_json() | {"added_later": 1}) == e


def test_a_rise_with_no_arrival_becomes_a_scoreboard_only_event():
    tl = [R(t, home=0, away=(2 if t >= 20 else 0)) for t in range(60)]
    [e] = scoreboard_only_events(tl, [])
    assert e.id == "shot-000180-sb" and e.type == "shot" and e.t == 18.0 and e.rim_id == -1
    assert e.made is True and e.team == "away" and e.confidence == 0.5 and e.needs_location is True
    assert e.flags == ["scoreboard-only"]
    assert e.signals == {"source": "scoreboard", "score_delta": [0, 2], "scoreboard": "made", "points": 2}
    assert e.clip == {"start": 10.0, "end": 21.0}
    assert e.quarter == 2 and e.clock_s == 282.0


def test_a_one_point_rise_is_flagged_as_a_possible_free_throw_and_t_never_goes_negative():
    tl = [R(t, home=(1 if t >= 1 else 0)) for t in range(10)]
    [e] = scoreboard_only_events(tl, [])
    assert e.t == 0.0 and e.team == "home" and e.signals["points"] == 1
    assert e.flags == ["scoreboard-only", "free-throw?"] and e.clip == {"start": 0.0, "end": 3.0}


def test_a_rise_already_explained_by_an_arrival_gives_no_scoreboard_only_event():
    tl = timeline_with_home_score_at(20)
    events = build_shot_events([A(17.0, rim_id=2)], tl, [None], "g")
    assert scoreboard_only_events(tl, events) == []
    just_after = build_shot_events([A(20.5)], tl, [None], "g")    # within SCORE_LEAD_S after: it was credited the rise
    assert just_after[0].signals["points"] == 2
    assert scoreboard_only_events(tl, just_after) == []
    later = build_shot_events([A(21.0)], tl, [None], "g")         # a full lead after the rise: explains nothing
    assert later[0].signals["points"] == 0
    assert len(scoreboard_only_events(tl, later)) == 1
    early = build_shot_events([A(12.0)], tl, [None], "g")         # 8 s before: too early
    assert len(scoreboard_only_events(tl, early)) == 1


def test_jumps_that_are_not_one_basket_give_no_scoreboard_only_event():
    four = timeline_with_home_score_at(20, points=4)
    assert scoreboard_only_events(four, []) == []
    both = [R(t, home=(2 if t >= 20 else 0), away=(1 if t >= 20 else 0)) for t in range(60)]
    assert scoreboard_only_events(both, []) == []
    down = [R(t, home=(0 if t >= 20 else 2)) for t in range(60)]
    assert scoreboard_only_events(down, []) == []


def test_a_blank_read_does_not_hide_a_rise():
    blank = [R(t, home=(None if t == 20 else (2 if t > 20 else 0))) for t in range(60)]
    [e] = scoreboard_only_events(blank, [])
    assert e.t == 19.0 and e.signals["score_delta"] == [2, 0]


def test_arrivals_at_different_rims_do_not_share_one_basket():
    tl = timeline_with_home_score_at(15)
    first, second = build_shot_events([A(10.0, rim_id=1), A(13.0, rim_id=2)], tl, [None, None], "g")
    assert first.signals["points"] == 0 and first.made is False
    assert second.signals["points"] == 2 and second.made is True


def test_a_dropout_across_the_arrival_is_no_evidence():
    gap = [R(t) for t in range(11)] + [R(t, home=2) for t in range(30, 40)]
    assert score_change(gap, 10.5, 6.0) is None                 # the latest read is from before the arrival
    assert score_change(gap, 8.0, 1.5) == (0, 0)                 # the after-read (t=9) is later than the arrival
    [e] = build_shot_events([A(10.5)], gap, [None], "g")
    assert e.made is None and e.signals["scoreboard"] == "unknown" and "no-scoreboard" in e.flags


def test_dead_time_marks_arrivals_as_warmups_but_keeps_late_posted_baskets():
    from film.interpret.events import mark_dead_time
    from film.interpret.phases import GamePhases
    phases = GamePhases(quarter_length_s=480.0, live=[(100.0, 200.0)], known=True, game_over_t=300.0)
    live, warm = build_shot_events([A(150.0), A(250.0)], [], [None, None], "g")
    tl = [R(t, home=(2 if t >= 255 else 0)) for t in range(240, 260)] + [R(t, home=(2 if t < 320 else 0 if t < 325 else 2)) for t in range(305, 330)]   # reset, then the next game scores
    late, after = scoreboard_only_events(tl, [])
    marked = mark_dead_time([live, warm, late, after], phases)
    assert marked == 2
    assert "warmup" not in live.flags and live.signals.get("phase") == "live"
    assert "warmup" in warm.flags and warm.signals["phase"] == "warmup"
    assert "warmup" not in late.flags                    # a basket posted during a break is still this game's
    assert "warmup" in after.flags                       # after the board reset it belongs to the next game
