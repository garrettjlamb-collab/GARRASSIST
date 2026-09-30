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
