from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.perceive.detectors import Box, FakeDetector
from film.train import datasets as ds
from film.train.prelabel import prelabel_balls, prelabel_rims, sample_times


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def test_sample_times():
    assert sample_times(100, 3) == [30.0, 50.0, 70.0]
    assert sample_times(50, 3) == []


def test_prelabel_rims_writes_pending(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    det = FakeDetector(fn=lambda img, i: [Box(100, 60, 150, 90, 0.8, "rim"), Box(0, 0, 9, 9, 0.9, "person")])
    n = prelabel_rims(m.game_id, n=4, detector=det, margin_s=0.5, log=lambda *a: None)
    assert n == 4
    ids = ds.list_pending("rim")
    assert len(ids) == 4 and all(i.startswith(m.game_id) for i in ids)
    assert [b["cls"] for b in ds.read_pending("rim", ids[0])["boxes"]] == ["rim"]


def test_prelabel_balls_samples_around_arrivals(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    ev = Event(id="shot-000010-r1", type="shot", t=1.0, rim_id=1, quarter=None, clock_s=None, team=None, made=None,
               confidence=0.3, clip={"start": 0.0, "end": 2.5}, signals={}, flags=[])
    save_events(game_dir(m.game_id), [ev])
    rims = FakeDetector(fn=lambda img, i: [Box(100, 60, 150, 90, 0.8, "rim")])
    balls = FakeDetector(fn=lambda img, i: [Box(5, 5, 15, 15, 0.6, "ball")])
    n = prelabel_balls(m.game_id, n=10, rim_detector=rims, ball_detector=balls, margin_s=0.5, log=lambda *a: None)
    assert n == 5                                              # five offsets around the one arrival
    data = ds.read_pending("ball", ds.list_pending("ball")[0])
    assert data["width"] == 240 and data["height"] == 240        # the rim crop, not the full frame
    assert data["boxes"][0]["cls"] == "ball"
