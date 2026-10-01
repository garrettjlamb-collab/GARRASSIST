import pytest

from film.review.decisions import apply_decisions, load_decisions, save_decision


def test_save_and_apply(tmp_path):
    assert load_decisions(tmp_path) == {}
    entry = save_decision(tmp_path, "shot-1", {"status": "confirmed", "made": False})
    assert entry["status"] == "confirmed" and entry["made"] is False and "decided_at" in entry
    save_decision(tmp_path, "shot-2", {"status": "rejected"})
    decisions = load_decisions(tmp_path)
    assert set(decisions) == {"shot-1", "shot-2"}
    events = [{"id": "shot-1", "made": True, "team": "home"}, {"id": "shot-2", "made": None, "team": None}, {"id": "shot-3", "made": True, "team": None}]
    merged = apply_decisions(events, decisions)
    assert merged[0]["made"] is False and merged[0]["team"] == "home" and merged[0]["decision"]["status"] == "confirmed"
    assert merged[1]["decision"]["status"] == "rejected"
    assert merged[2]["decision"] is None and merged[2]["made"] is True


def test_bad_status_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_decision(tmp_path, "shot-1", {"status": "maybe"})


def test_save_is_atomic_and_threadsafe(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(20) as pool:
        list(pool.map(lambda i: save_decision(tmp_path, f"shot-{i}", {"status": "unsure"}), range(20)))
    assert set(load_decisions(tmp_path)) == {f"shot-{i}" for i in range(20)}
    assert not list(tmp_path.glob("*.tmp"))
    import json
    json.loads((tmp_path / "review.json").read_text())


def _ev(eid, t, rim_id=1, source="motion", verdict="missed"):
    return {"id": eid, "t": t, "rim_id": rim_id, "signals": {"source": source, "scoreboard": verdict}}


def test_decisions_follow_a_shot_whose_time_shifted(tmp_path):
    from film.review.decisions import carry_over_decisions
    old = [_ev("shot-000100-r1", 10.0), _ev("shot-000300-r1", 30.0), _ev("shot-000500-r2", 50.0, rim_id=2), _ev("shot-000700-r1", 70.0)]
    new = [_ev("shot-000108-r1", 10.8, source="ball"),        # same rim, 0.8 s later: the confirmed shot moved here
           _ev("shot-000300-r1", 30.0),                       # unchanged id keeps its decision untouched
           _ev("shot-000515-r1", 51.5),                       # other rim: not the same shot
           _ev("shot-000715-r1", 71.5, source="ball")]        # 1.5 s from a rejection: too far to inherit a rejection
    save_decision(tmp_path, "shot-000100-r1", {"status": "confirmed", "made": True})
    save_decision(tmp_path, "shot-000300-r1", {"status": "unsure"})
    save_decision(tmp_path, "shot-000500-r2", {"status": "confirmed", "made": False})
    save_decision(tmp_path, "shot-000700-r1", {"status": "rejected"})
    counts = carry_over_decisions(tmp_path, old, new)
    assert counts == {"kept": 1, "moved": 1, "orphaned": 2}
    d = load_decisions(tmp_path)
    assert d["shot-000108-r1"]["status"] == "confirmed" and d["shot-000108-r1"]["made"] is True
    assert d["shot-000108-r1"]["moved_from"] == "shot-000100-r1" and "shot-000100-r1" not in d
    assert d["shot-000300-r1"]["status"] == "unsure"
    assert "shot-000515-r1" not in d and "shot-000715-r1" not in d
    assert d["shot-000500-r2"]["status"] == "confirmed" and d["shot-000700-r1"]["status"] == "rejected"   # orphans are kept, never deleted


def test_a_scoreboard_card_hands_its_decision_to_the_shot_that_now_explains_the_basket(tmp_path):
    from film.review.decisions import carry_over_decisions
    old = [_ev("shot-000180-sb", 18.0, rim_id=-1, source="scoreboard", verdict="made")]
    new = [_ev("shot-000148-r2", 14.8, rim_id=2, source="ball", verdict="made"),       # the ball arrival that now carries the points
           _ev("shot-000170-r2", 17.0, rim_id=2, source="motion", verdict="missed")]   # closer in time, but not the basket
    save_decision(tmp_path, "shot-000180-sb", {"status": "confirmed", "made": True, "team": "home"})
    assert carry_over_decisions(tmp_path, old, new) == {"kept": 0, "moved": 1, "orphaned": 0}
    d = load_decisions(tmp_path)
    assert d["shot-000148-r2"]["made"] is True and d["shot-000148-r2"]["team"] == "home"


def test_two_old_decisions_never_land_on_one_new_event_and_existing_ones_are_not_overwritten(tmp_path):
    from film.review.decisions import carry_over_decisions
    old = [_ev("shot-000100-r1", 10.0), _ev("shot-000112-r1", 11.2), _ev("shot-000400-r1", 40.0)]
    new = [_ev("shot-000105-r1", 10.5), _ev("shot-000405-r1", 40.5)]
    save_decision(tmp_path, "shot-000100-r1", {"status": "confirmed", "made": True})
    save_decision(tmp_path, "shot-000112-r1", {"status": "confirmed", "made": False})
    save_decision(tmp_path, "shot-000400-r1", {"status": "confirmed", "made": True})
    save_decision(tmp_path, "shot-000405-r1", {"status": "unsure"})                    # already decided under the new id
    counts = carry_over_decisions(tmp_path, old, new)
    assert counts == {"kept": 1, "moved": 1, "orphaned": 2}
    d = load_decisions(tmp_path)
    assert d["shot-000105-r1"]["moved_from"] == "shot-000100-r1"                       # the nearer of the two wins
    assert d["shot-000405-r1"]["status"] == "unsure"
