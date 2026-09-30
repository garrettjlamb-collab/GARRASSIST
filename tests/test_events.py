from film.interpret.arrivals import Arrival
from film.interpret.events import Event, build_shot_events, event_id, load_events, save_events, score_change
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
    assert e.clip == {"start": 7.0, "end": 11.5}
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
