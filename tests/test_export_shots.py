import csv
import json

from film.common.paths import game_dir
from film.export.clips import clip_path, cut_clip
from film.export.shots import export_reel, export_shots
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.review.decisions import save_decision


def _game(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i * 0.4, rim_id=1, quarter=2, clock_s=125.0 - i, team="home",
                    made=True, confidence=0.9, clip={"start": 0.5, "end": 1.5}, signals={}, flags=["x"]) for i in range(3)]
    save_events(gd, events)
    save_decision(gd, "shot-000000-r1", {"status": "confirmed", "made": False})
    save_decision(gd, "shot-000001-r1", {"status": "rejected"})
    save_decision(gd, "shot-000002-r1", {"status": "unsure"})
    return m.game_id, gd


def test_export_shots(synthetic_video, tmp_path, monkeypatch):
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    csv_path, json_path = export_shots(game_id)
    rows = list(csv.DictReader(csv_path.open()))
    assert [r["id"] for r in rows] == ["shot-000000-r1", "shot-000002-r1"]
    assert rows[0]["made"] == "False" and rows[0]["status"] == "confirmed" and rows[1]["status"] == "unsure"
    data = json.loads(json_path.read_text())
    assert data["format"] == "film-shot-times" and len(data["shots"]) == 2 and data["shots"][0]["made"] is False


def test_export_reel_uses_confirmed_only(synthetic_video, tmp_path, monkeypatch):
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    for i in range(3):
        cut_clip(synthetic_video, 0.5, 1.5, clip_path(gd, f"shot-{i:06d}-r1"))
    reel = export_reel(game_id)
    assert reel is not None and reel.exists() and reel.name.endswith(".reel.mp4")


def test_a_confirmed_unknown_event_is_a_shot(synthetic_video, tmp_path, monkeypatch):
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    from film.interpret.events import load_events

    unknown = Event(id="shot-000030-r2", type="unknown", t=3.0, rim_id=2, quarter=2, clock_s=100.0, team=None,
                    made=None, confidence=0.2, clip={"start": 0.0, "end": 3.0}, signals={}, flags=["clf-none"])
    save_events(gd, load_events(gd) + [unknown])
    save_decision(gd, "shot-000030-r2", {"status": "confirmed", "made": True})
    rows = list(csv.DictReader(export_shots(game_id)[0].open()))
    assert [r["id"] for r in rows] == ["shot-000000-r1", "shot-000002-r1", "shot-000030-r2"]
    assert rows[2]["made"] == "True" and rows[2]["status"] == "confirmed"


def test_warmup_cards_never_reach_the_export(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i, rim_id=1, quarter=1, clock_s=None, team=None,
                    made=True, confidence=0.5, clip={"start": 0.0, "end": 1.0}, signals={},
                    flags=(["warmup"] if i == 0 else [])) for i in range(2)]
    save_events(gd, events)
    for e in events:
        save_decision(gd, e.id, {"status": "confirmed", "made": True})
    _, json_path = export_shots(m.game_id)
    assert [s["id"] for s in json.loads(json_path.read_text())["shots"]] == ["shot-000001-r1"]


def test_the_export_says_which_shots_had_a_foul_and_which_do_not_count(synthetic_video, tmp_path, monkeypatch):
    """A shot after the whistle is true made-or-missed data for the classifier and not an attempt in the stats."""
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    save_decision(gd, "shot-000000-r1", {"status": "confirmed", "made": True, "foul": "before_shot"})
    save_decision(gd, "shot-000001-r1", {"status": "confirmed", "made": False, "foul": "on_shot"})
    save_decision(gd, "shot-000002-r1", {"status": "confirmed", "made": True})
    csv_path, json_path = export_shots(game_id)
    rows = list(csv.DictReader(csv_path.open()))
    assert [(r["foul"], r["counts"]) for r in rows] == [("before_shot", "False"), ("on_shot", "True"), ("", "True")]
    shots = json.loads(json_path.read_text())["shots"]
    assert [(s["foul"], s["counts"]) for s in shots] == [("before_shot", False), ("on_shot", True), (None, True)]


def test_the_export_carries_the_dot_and_zone(synthetic_video, tmp_path, monkeypatch):
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    save_decision(gd, "shot-000000-r1", {"status": "confirmed", "made": False, "x": 3.0, "y": 4.0})
    csv_path, json_path = export_shots(game_id)
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["x"] == "3.0" and rows[0]["zone"] == "corner-3-left" and rows[0]["shot_type"] == "three"
    assert rows[1]["x"] == "" and rows[1]["zone"] == ""
    shots = json.loads(json_path.read_text())["shots"]
    assert shots[0]["x"] == 3.0 and shots[1]["x"] is None
