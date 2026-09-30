import numpy as np
from fastapi.testclient import TestClient

from film.common.paths import game_dir
from film.export.clips import clip_path, cut_clip
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.perceive.detectors import Box
from film.review.app import create_app
from film.train import datasets as ds


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def _game(synthetic_video):
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i * 0.5, rim_id=1, quarter=1, clock_s=100.0, team=None,
                    made=bool(i), confidence=0.4 + 0.5 * i, clip={"start": 0.5 + i * 0.5, "end": 1.5 + i * 0.5}, signals={}, flags=[])
              for i in range(2)]
    save_events(gd, events)
    cut_clip(synthetic_video, 0.5, 1.5, clip_path(gd, events[0].id))
    return m.game_id, gd


def test_games_events_and_decisions(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    client = TestClient(create_app())
    games = client.get("/api/games").json()
    assert games[0]["game_id"] == game_id and games[0]["events"] == 2 and games[0]["decided"] == 0
    events = client.get(f"/api/games/{game_id}/events").json()
    assert [e["id"] for e in events] == ["shot-000000-r1", "shot-000001-r1"] and events[0]["decision"] is None
    r = client.post(f"/api/games/{game_id}/decisions/shot-000001-r1", json={"status": "confirmed", "made": False})
    assert r.status_code == 200 and r.json()["status"] == "confirmed"
    events = client.get(f"/api/games/{game_id}/events").json()
    assert events[1]["made"] is False and events[1]["decision"]["status"] == "confirmed"
    assert client.get("/api/games").json()[0]["decided"] == 1
    assert client.post(f"/api/games/{game_id}/decisions/nope", json={"status": "confirmed"}).status_code == 404
    assert client.post(f"/api/games/{game_id}/decisions/shot-000000-r1", json={"status": "maybe"}).status_code == 400
    assert client.get("/api/games/missing/events").status_code == 404


def test_clip_and_ranged_video(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    client = TestClient(create_app())
    assert client.get(f"/clips/{game_id}/shot-000000-r1.mp4").status_code == 200
    assert client.get(f"/clips/{game_id}/shot-000001-r1.mp4").status_code == 404
    whole = client.get(f"/video/{game_id}")
    assert whole.status_code == 200 and whole.headers["accept-ranges"] == "bytes"
    part = client.get(f"/video/{game_id}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    assert part.headers["content-range"].startswith("bytes 0-99/")


def test_label_flow(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    assert client.get("/api/label/rim/next").json() == {"done": True, "remaining": 0}
    img = np.zeros((100, 200, 3), np.uint8)
    ds.write_pending("rim", "g_1", img, [Box(10, 20, 30, 40, 0.5, "rim")])
    ds.write_pending("rim", "g_2", img, [])
    nxt = client.get("/api/label/rim/next").json()
    assert nxt["image_id"] == "g_1" and nxt["remaining"] == 2 and nxt["boxes"][0]["cls"] == "rim"
    assert client.get(nxt["image_url"]).status_code == 200
    r = client.post("/api/label/rim/g_1/boxes", json={"boxes": [{"cls": "rim", "x1": 10, "y1": 20, "x2": 30, "y2": 40}]})
    assert r.status_code == 200 and r.json()["split"] == ds.split_for_id("g_1")
    assert client.post("/api/label/rim/g_2/skip").status_code == 200
    assert client.get("/api/label/rim/next").json()["done"] is True
    ds.write_pending("rimclf", "m_1", np.zeros((288, 288, 3), np.uint8), [], extra={"suggested": "missed"})
    assert client.get("/api/label/rimclf/next").json()["suggested"] == "missed"
    assert client.post("/api/label/rimclf/m_1/class", json={"cls": "hit"}).status_code == 400
    assert client.post("/api/label/rimclf/m_1/class", json={"cls": "missed"}).json()["split"] == "val"  # the first image of a class seeds val (store_class_image)
    assert client.get("/").status_code == 200
