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


def test_video_ranges(synthetic_video, tmp_path, monkeypatch):
    from pathlib import Path

    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    client = TestClient(create_app())
    data = Path(synthetic_video).read_bytes()
    size = len(data)
    url = f"/video/{game_id}"
    tail = client.get(url, headers={"Range": "bytes=-100"})
    assert tail.status_code == 206 and tail.content == data[-100:]
    assert tail.headers["content-range"] == f"bytes {size - 100}-{size - 1}/{size}"
    rest = client.get(url, headers={"Range": "bytes=100-"})
    assert rest.status_code == 206 and rest.content == data[100:]
    clamped = client.get(url, headers={"Range": f"bytes=0-{size + 500}"})
    assert clamped.status_code == 206 and clamped.content == data
    for bad in ("bytes=-", "bytes=900-100", f"bytes={size}-", "bytes=abc"):
        r = client.get(url, headers={"Range": bad})
        assert r.status_code == 416 and r.headers["content-range"] == f"bytes */{size}", bad


def test_label_validation(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    assert client.get("/api/label/nope/next").status_code == 404
    assert client.post("/api/label/nope/x/skip").status_code == 404
    ds.write_pending("rim", "g_1", np.zeros((100, 200, 3), np.uint8), [])
    assert client.post("/api/label/rim/g_1/class", json={"cls": "made"}).status_code == 400
    bad = {"boxes": [{"cls": "ball", "x1": 1, "y1": 1, "x2": 5, "y2": 5}]}
    assert client.post("/api/label/rim/g_1/boxes", json=bad).status_code == 400
    ok = {"boxes": [{"cls": "rim", "x1": 1, "y1": 1, "x2": 5, "y2": 5}]}
    assert client.post("/api/label/rim/missing/boxes", json=ok).status_code == 404
    assert client.post("/api/label/rimclf/missing/class", json={"cls": "made"}).status_code == 404
    assert client.post("/api/label/rim/missing/skip").status_code == 404
    assert not (tmp_path / "datasets" / "nope").exists()


def test_one_broken_game_does_not_blank_the_list(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, _ = _game(synthetic_video)
    broken = tmp_path / "games" / "2026-01-02-broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{not json")
    client = TestClient(create_app())
    r = client.get("/api/games")
    assert r.status_code == 200
    by_id = {g["game_id"]: g for g in r.json()}
    assert by_id[game_id]["events"] == 2 and "error" not in by_id[game_id]
    assert set(by_id["2026-01-02-broken"]) == {"game_id", "error"} and by_id["2026-01-02-broken"]["error"]


def test_an_explicit_null_team_persists_and_an_omitted_one_does_not(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i, rim_id=1, quarter=1, clock_s=100.0, team="home",
                    made=True, confidence=0.9, clip={"start": 0.5, "end": 1.5}, signals={}, flags=[]) for i in range(2)]
    save_events(gd, events)
    client = TestClient(create_app())
    r = client.post(f"/api/games/{game_id}/decisions/shot-000000-r1", json={"status": "confirmed", "made": True, "team": None})
    assert r.status_code == 200 and "team" in r.json() and r.json()["team"] is None
    client.post(f"/api/games/{game_id}/decisions/shot-000001-r1", json={"status": "confirmed"})
    first, second = client.get(f"/api/games/{game_id}/events").json()
    assert first["team"] is None                    # the coach cleared it
    assert second["team"] == "home" and "team" not in second["decision"]


def test_quarter_marks_are_read_saved_and_applied_to_the_cards(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)                       # events at 1.0 and 1.5 s
    client = TestClient(create_app())
    assert client.get(f"/api/games/{game_id}/quarters").json() == {"quarters": [], "scoreboard": False}
    assert client.get("/api/games").json()[0]["scoreboard"] is False
    r = client.post(f"/api/games/{game_id}/quarters", json={"quarters": [[1.2, 2.0]]})
    assert r.status_code == 200 and r.json() == {"events": 2, "warmups": 1, "quarters": 1}
    events = {e["id"]: e for e in client.get(f"/api/games/{game_id}/events").json()}
    assert "warmup" in events["shot-000000-r1"]["flags"] and events["shot-000001-r1"]["quarter"] == 1
    assert client.get(f"/api/games/{game_id}/quarters").json()["quarters"] == [[1.2, 2.0]]
    bad = client.post(f"/api/games/{game_id}/quarters", json={"quarters": [[2.0, 1.0]]})
    assert bad.status_code == 400 and "before" in bad.json()["detail"]


def test_a_foul_on_the_play_is_saved_with_the_decision_and_shown_again(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    client = TestClient(create_app())
    r = client.post(f"/api/games/{game_id}/decisions/shot-000001-r1", json={"status": "confirmed", "made": False, "foul": "on_shot"})
    assert r.status_code == 200 and r.json()["foul"] == "on_shot"
    events = {e["id"]: e for e in client.get(f"/api/games/{game_id}/events").json()}
    assert events["shot-000001-r1"]["foul"] == "on_shot" and events["shot-000000-r1"]["foul"] is None
    bad = client.post(f"/api/games/{game_id}/decisions/shot-000001-r1", json={"status": "confirmed", "foul": "flagrant"})
    assert bad.status_code == 400


def test_the_game_list_shows_titles(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    client = TestClient(create_app())
    assert client.get("/api/games").json()[0]["title"] == game_id
    from film.common.manifest import Manifest
    m = Manifest.load(gd); m.title = "1 · Review in full: a game"; m.save(gd)
    assert client.get("/api/games").json()[0]["title"] == "1 · Review in full: a game"


def test_court_points_are_accepted_over_the_api(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    ds.write_pending("court", "c1", np.zeros((100, 200, 3), np.uint8), [], extra={"t": 1.0, "points": {}})
    nxt = client.get("/api/label/court/next").json()
    assert nxt["image_id"] == "c1" and nxt["points"] == {}
    r = client.post("/api/label/court/c1/points", json={"points": {"near-corner-left": [20, 80], "near-corner-right": [180, 80],
                                                                   "near-ft-top": [100, 30], "near-ft-left": [70, 50]}})
    assert r.status_code == 200 and r.json()["split"] in ("train", "val")
    assert client.get("/api/label/court/next").json()["done"] is True
    bad = client.post("/api/label/court/c1/points", json={"points": {"nowhere": [1, 2]}})
    assert bad.status_code == 400
