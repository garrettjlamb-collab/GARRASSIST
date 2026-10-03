import pytest

from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import Event, load_events, mark_dead_time, save_events
from film.interpret.quarters import apply_quarters, load_quarters, phases_from_quarters, quarter_of, save_quarters


def _event(t, flags=None, source="ball"):
    return Event(id=f"shot-{int(t * 10):06d}-r1", type="shot", t=t, rim_id=1, quarter=None, clock_s=None, team=None,
                 made=None, confidence=0.5, clip={"start": t - 1, "end": t + 1}, signals={"source": source}, flags=list(flags or []))


def test_quarter_of_names_the_quarter_a_moment_falls_in():
    marks = [(100.0, 600.0), (700.0, 1200.0), (1800.0, 2300.0), (2400.0, 2900.0)]
    assert [quarter_of(marks, t) for t in (50, 100, 599.9, 650, 700, 2500, 3000)] == [None, 1, 1, None, 2, 4, None]


def test_quarter_marks_decide_what_is_warmup():
    phases = phases_from_quarters([(10.0, 20.0)])
    events = [_event(5.0), _event(15.0), _event(25.0)]
    assert mark_dead_time(events, phases) == 2
    assert [("warmup" in e.flags) for e in events] == [True, False, True]


def test_a_mark_moved_later_clears_a_warmup_flag_it_no_longer_earns():
    events = [_event(15.0, flags=["warmup", "motion-only"])]
    mark_dead_time(events, phases_from_quarters([(10.0, 20.0)]))
    assert events[0].flags == ["motion-only"] and events[0].signals["phase"] == "live"


def test_save_refuses_marks_that_cannot_be_quarters(tmp_path):
    with pytest.raises(ValueError, match="before"):
        save_quarters(tmp_path, [(20.0, 10.0)])
    with pytest.raises(ValueError, match="overlap"):
        save_quarters(tmp_path, [(10.0, 30.0), (20.0, 40.0)])
    save_quarters(tmp_path, [(50.0, 60.0), (10.0, 20.0)])
    assert load_quarters(tmp_path) == [(10.0, 20.0), (50.0, 60.0)]                   # kept in film order
    assert load_quarters(tmp_path / "nowhere") == []


def test_apply_quarters_rewrites_the_cards_without_rerunning_anything(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="nfhs-auto")
    gd = game_dir(m.game_id)
    save_events(gd, [_event(0.5), _event(1.5), _event(2.5)])
    save_quarters(gd, [(1.0, 2.0), (2.2, 2.8)])
    assert apply_quarters(gd) == {"events": 3, "warmups": 1, "quarters": 2}
    events = load_events(gd)
    assert [(e.quarter, "warmup" in e.flags) for e in events] == [(None, True), (1, False), (2, False)]
