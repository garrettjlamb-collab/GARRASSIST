import pytest

from film.common.paths import game_dir
from film.export.clips import clip_path, cut_clip
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.review import dotreview
from film.review.decisions import apply_decisions, load_decisions, save_decision


@pytest.fixture
def game(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    made = [("low", 30.0, 20.0, 0.3, "flight"), ("mid", 31.0, 21.0, 0.6, "pose"), ("sure", 32.0, 22.0, 0.95, "pose"), ("none", None, None, None, "none"), ("ft", 25.0, 19.0, 0.9, "free-throw")]
    events = []
    for i, (name, x, y, conf, method) in enumerate(made):
        e = Event(id=f"shot-{name}", type="shot", t=1.0 + i * 0.1, rim_id=1, quarter=1, clock_s=100.0, team=None, made=True, confidence=0.9, clip={"start": 0.5, "end": 1.5},
                  signals={"phase": "live", "location": {"method": method}}, flags=[], x=x, y=y, location_conf=conf)
        events.append(e)
        cut_clip(synthetic_video, 0.5, 1.5, clip_path(gd, e.id))
        save_decision(gd, e.id, {"status": "confirmed", "made": True, "team": "home"})
    save_events(gd, events)
    monkeypatch.setattr(dotreview, "is_blind", lambda g, e: False)
    return m.game_id, gd


def test_the_worst_dots_come_first_then_no_dot_then_the_sure_ones_and_never_a_free_throw(game):
    assert [r["event_id"] for r in dotreview.queue()] == ["shot-low", "shot-mid", "shot-none", "shot-sure"]
    card = dotreview.next_shot()
    assert card["event_id"] == "shot-low" and card["x"] == 30.0 and card["conf"] == 0.3
    assert (card["doubtful"], card["no_dot"], card["bulk"], card["checks"]) == (2, 1, 1, 0)


def test_the_hidden_checks_of_the_sure_dots_come_before_everything(game, monkeypatch):
    monkeypatch.setattr(dotreview, "is_blind", lambda g, e: e == "shot-sure")
    assert [r["event_id"] for r in dotreview.queue()] == ["shot-sure", "shot-low", "shot-mid", "shot-none"]
    assert dotreview.next_shot()["checks"] == 1


def test_a_hidden_card_carries_nothing_about_the_machines_dot(game, monkeypatch):
    monkeypatch.setattr(dotreview, "is_blind", lambda g, e: True)
    card = dotreview.next_shot()
    assert card["blind"] is True and not {"x", "y", "conf", "zone", "method"} & set(card)
    assert card["event_id"] == "shot-sure"


def test_keeping_a_dot_saves_it_as_approved_and_keeps_his_other_answers(game):
    game_id, gd = game
    dotreview.answer(game_id, "shot-low")
    d = load_decisions(gd)["shot-low"]
    assert (d["x"], d["y"], d["dot_source"], d["status"], d["made"], d["team"]) == (30.0, 20.0, "approved", "confirmed", True, "home")


def test_moving_a_dot_saves_it_as_his(game):
    game_id, gd = game
    dotreview.answer(game_id, "shot-low", 40.0, 30.0)
    assert load_decisions(gd)["shot-low"]["dot_source"] == "coach"
    assert dotreview.read_answers()[0]["machine_x"] == 30.0


def test_a_hidden_or_empty_card_cannot_be_kept_without_a_dot(game, monkeypatch):
    game_id, gd = game
    with pytest.raises(ValueError):
        dotreview.answer(game_id, "shot-none")
    monkeypatch.setattr(dotreview, "is_blind", lambda g, e: True)
    with pytest.raises(ValueError):
        dotreview.answer(game_id, "shot-low")


def test_bulk_keeps_only_the_sure_dots_and_leaves_their_hidden_checks(game, monkeypatch):
    game_id, gd = game
    assert dotreview.bulk_approve(log=lambda m: None) == 1
    assert load_decisions(gd)["shot-sure"]["dot_source"] == "bulk"
    assert "x" not in load_decisions(gd)["shot-mid"]
    assert "shot-sure" not in [r["event_id"] for r in dotreview.queue()]


def test_a_sure_dot_hidden_as_a_check_is_not_kept_in_bulk(game, monkeypatch):
    game_id, gd = game
    monkeypatch.setattr(dotreview, "is_blind", lambda g, e: e == "shot-sure")
    assert dotreview.bulk_approve(log=lambda m: None) == 0
    assert dotreview.next_shot()["checks"] == 1


def test_the_score_only_counts_hidden_cards_as_measuring_the_dots(game, monkeypatch):
    game_id, gd = game
    dotreview.answer(game_id, "shot-low")                     # kept, looking at it
    dotreview.answer(game_id, "shot-mid", 41.0, 21.0)         # moved 10 ft, looking at it
    monkeypatch.setattr(dotreview, "is_blind", lambda g, e: True)
    dotreview.answer(game_id, "shot-sure", 35.0, 26.0)        # hidden: 5 ft off
    got = dotreview.score()
    assert got["hidden"]["all"]["n"] == 1 and got["hidden"]["all"]["median_ft"] == 5.0 and got["hidden"]["conf_under_0.8"] is None
    assert got["shown_not_independent"]["approved"] == 1 and got["shown_not_independent"]["moved"] == 1


def test_a_skipped_shot_leaves_the_queue_and_keeps_no_dot(game):
    game_id, gd = game
    dotreview.skip(game_id, "shot-low")
    assert "shot-low" not in [r["event_id"] for r in dotreview.queue()]
    assert "x" not in load_decisions(gd)["shot-low"]


def test_a_shot_not_in_the_queue_is_refused(game):
    game_id, _ = game
    with pytest.raises(KeyError):
        dotreview.answer(game_id, "shot-ft", 1.0, 1.0)


def test_a_dot_he_only_kept_does_not_count_as_measuring_the_machine(game):
    from film.export.evaluate import evaluate

    game_id, gd = game
    dotreview.answer(game_id, "shot-low")
    dotreview.answer(game_id, "shot-mid", 41.0, 21.0)
    got = evaluate(game_id)
    assert got["dot_placed_by_coach"] == 1 and got["median_dot_error_ft"] == pytest.approx(10.0)


def test_a_later_decision_keeps_where_the_dot_came_from(game):
    game_id, gd = game
    dotreview.answer(game_id, "shot-low")
    save_decision(gd, "shot-low", {"status": "confirmed", "made": False})
    d = load_decisions(gd)["shot-low"]
    assert d["dot_source"] == "approved" and d["x"] == 30.0


def _trust(gd, **by_id):
    from film.interpret.events import load_events

    events = load_events(gd)
    for e in events:
        if e.id in by_id:
            e.signals["trust"] = by_id[e.id]
    save_events(gd, events)


def test_the_trust_model_overrides_the_machines_confidence(game):
    game_id, gd = game
    _trust(gd, **{"shot-sure": {"prob": 0.3, "agree": False, "trusted": False, "pick": [1, 1]}, "shot-low": {"prob": 0.9, "agree": True, "trusted": True, "pick": [30, 20]}})
    order = [r["event_id"] for r in dotreview.queue()]
    assert order.index("shot-sure") < order.index("shot-none") < order.index("shot-low")
    assert dotreview.next_shot()["event_id"] == "shot-sure"


def test_one_trusted_dot_in_five_stays_hidden_and_is_not_kept_in_bulk(game, monkeypatch):
    game_id, gd = game
    _trust(gd, **{"shot-sure": {"prob": 0.9, "agree": True, "trusted": True, "pick": [32, 22]}})
    monkeypatch.setattr(dotreview, "is_trust_check", lambda g, e: True)
    assert dotreview.next_shot()["event_id"] == "shot-sure" and dotreview.next_shot()["blind"] is True
    assert dotreview.bulk_approve(log=lambda m: None) == 0


def test_no_shot_stops_it_being_a_shot_and_keeps_what_else_he_said(game):
    game_id, gd = game
    dotreview.no_shot(game_id, "shot-none")
    d = load_decisions(gd)["shot-none"]
    assert d["status"] == "rejected" and d["team"] == "home" and "x" not in d
    assert "shot-none" not in [r["event_id"] for r in dotreview.queue()]
    assert dotreview.read_answers()[-1]["no_shot"] is True


def test_skip_leaves_it_a_shot(game):
    game_id, gd = game
    dotreview.skip(game_id, "shot-none")
    assert load_decisions(gd)["shot-none"]["status"] == "confirmed"


def test_a_game_with_no_court_has_all_its_shots_queued_with_no_dot_and_never_counts_as_measuring(game):
    game_id, gd = game
    card = next(c for c in dotreview.queue() if c["event_id"] == "shot-none")
    assert card["x"] is None and dotreview.next_shot()["no_dot"] == 1
    dotreview.answer(game_id, "shot-none", 12.0, 20.0)
    assert load_decisions(gd)["shot-none"]["dot_source"] == "coach"
    assert dotreview.score()["hidden"]["all"] is None and dotreview.score()["no_machine_dot_placed"] == 1
