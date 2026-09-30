import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import load_events
from film.interpret.run import interpret
from film.perceive.tracks import replace_kind, rows_to_frame
from film.train.rimclf import FakeClassifier


def _tracks(gd):
    rim = (100.0, 60.0, 150.0, 90.0)
    rim_rows = [{"frame": f, "t": f / 10, "kind": "rim", "track_id": 1, "ref_id": None, "x1": rim[0], "y1": rim[1],
                 "x2": rim[2], "y2": rim[3], "conf": 0.9, "on_court": None, "court_x": None, "court_y": None,
                 "team": None, "model_version": "x"} for f in range(30)]
    ys = np.linspace(20, 100, 12)
    ball_rows = [{"frame": 4 + i, "t": (4 + i) / 10, "kind": "ball", "track_id": None, "ref_id": 1, "x1": 119.0,
                  "y1": float(y) - 6, "x2": 131.0, "y2": float(y) + 6, "conf": 0.7, "on_court": None,
                  "court_x": None, "court_y": None, "team": None, "model_version": "x"} for i, y in enumerate(ys)]
    replace_kind(gd, rows_to_frame(rim_rows), "rim")
    replace_kind(gd, rows_to_frame(ball_rows), "ball")


def test_interpret_writes_events(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    gd = game_dir(m.game_id)
    _tracks(gd)
    events = interpret(m.game_id, classifier=FakeClassifier([("made", 0.9)]), log=lambda *a: None)
    assert len(events) == 1
    e = events[0]
    assert e.type == "shot" and e.made is True and e.confidence == 0.63 and "no-scoreboard" in e.flags
    assert load_events(gd) == events
    saved = Manifest.load(gd)
    assert saved.stages["interpret"]["status"] == "done" and saved.stages["interpret"]["shots"] == 1
    assert saved.model_versions["rimclf"] == "rimclf@fake"


def test_interpret_without_classifier(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    _tracks(game_dir(m.game_id))
    [e] = interpret(m.game_id, use_classifier=False, log=lambda *a: None)
    assert e.made is None and e.confidence == 0.3 and "no-classifier" in e.flags
