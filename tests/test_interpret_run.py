import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import load_events
from film.interpret.run import interpret
from film.perceive.detectors import Box, FakeDetector
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
    events = interpret(m.game_id, classifier=FakeClassifier([("made", 0.9)]), ball_detector=FakeDetector(), log=lambda *a: None)
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


def test_interpret_adds_scoreboard_only_events_without_calling_the_classifier(synthetic_video, tmp_path, monkeypatch):
    from film.perceive.scoreboard import ScoreboardRead, save_timeline

    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    gd = game_dir(m.game_id)
    _tracks(gd)
    reads = [ScoreboardRead(t=float(t), period=1, clock_s=400.0 - t, home=(3 if t >= 30 else 0), away=0,
                            home_fouls=0, away_fouls=0, conf=1.0) for t in range(60)]
    save_timeline(reads, gd / "scoreboard.jsonl")
    clf = FakeClassifier([("missed", 0.9)])
    lines = []
    events = interpret(m.game_id, classifier=clf, ball_detector=FakeDetector(), log=lines.append)
    assert [e.rim_id for e in events] == [1, -1]
    assert clf.calls == 1
    sb = events[1]
    assert sb.id == "shot-000280-sb" and sb.team == "home" and sb.signals["points"] == 3
    assert "1 scoreboard-only" in lines[-1]
    assert Manifest.load(gd).stages["interpret"]["scoreboard_only"] == 1


def test_a_rerun_carries_the_coachs_decisions_onto_the_redetected_shot(synthetic_video, tmp_path, monkeypatch):
    from film.interpret.events import Event, save_events
    from film.review.decisions import load_decisions, save_decision
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    gd = game_dir(m.game_id)
    _tracks(gd)
    [fresh] = interpret(m.game_id, use_classifier=False, log=lambda *a: None)
    # Pretend the previous run had put this shot half a second earlier, under another id, and the coach confirmed it.
    old = Event.from_json({**fresh.to_json(), "id": "shot-earlier-r1", "t": round(fresh.t - 0.5, 2)})
    save_events(gd, [old])
    save_decision(gd, old.id, {"status": "confirmed", "made": True})
    lines = []
    [again] = interpret(m.game_id, use_classifier=False, log=lines.append)
    decisions = load_decisions(gd)
    assert again.id == fresh.id and decisions[fresh.id]["made"] is True and decisions[fresh.id]["moved_from"] == old.id
    assert Manifest.load(gd).stages["interpret"]["decisions"] == {"kept": 0, "moved": 1, "orphaned": 0}
    assert any("1 moved" in line for line in lines)


def test_the_classifier_is_given_the_balls_path_and_the_path_is_read_from_film_once(synthetic_video, tmp_path, monkeypatch):
    from film.interpret.ballpath import FEATURES

    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    gd = game_dir(m.game_id)
    _tracks(gd)
    given = []

    class Recording(FakeClassifier):
        def predict(self, features):
            given.append(list(features))
            return super().predict(features)

    det = FakeDetector(per_call=[[Box(100.0, 100.0, 110.0, 110.0, 0.7, "ball")]], name="ball@fake")
    interpret(m.game_id, classifier=Recording([("made", 0.9)]), ball_detector=det, log=lambda *a: None)
    assert len(given) == 1 and len(given[0]) == len(FEATURES)
    assert dict(zip(FEATURES, given[0]))["seen"] == 1.0 and dict(zip(FEATURES, given[0]))["peak_conf"] == 0.7
    calls = det.calls
    assert calls > 5 and (gd / "ballpaths.json").exists()
    interpret(m.game_id, classifier=Recording([("made", 0.9)]), ball_detector=det, log=lambda *a: None)
    assert det.calls == calls and given[1] == given[0]                       # a re-run reuses the stored path
    other = FakeDetector(name="ball@newer")
    interpret(m.game_id, classifier=Recording([("made", 0.9)]), ball_detector=other, log=lambda *a: None)
    assert other.calls == calls and dict(zip(FEATURES, given[2]))["seen"] == 0.0   # a new detector reads them again


def test_quarter_marks_set_warmups_and_quarters_on_a_fresh_interpret(synthetic_video, tmp_path, monkeypatch):
    from film.interpret.quarters import save_quarters

    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    gd = game_dir(m.game_id)
    _tracks(gd)
    [e] = interpret(m.game_id, use_classifier=False, log=lambda *a: None)
    assert "warmup" not in e.flags and e.quarter is None
    save_quarters(gd, [(e.t + 1, e.t + 2)])
    [e] = interpret(m.game_id, use_classifier=False, log=lambda *a: None)
    assert "warmup" in e.flags
    save_quarters(gd, [(0.0, e.t - 0.5), (e.t - 0.1, e.t + 1)])
    [e] = interpret(m.game_id, use_classifier=False, log=lambda *a: None)
    assert "warmup" not in e.flags and e.quarter == 2


def test_hoop_offset_is_shared_by_every_track_of_the_same_basket(monkeypatch):
    """Ten short tracks of one basket each see one arrival: alone none can place the hoop, together they can."""
    import pandas as pd
    import film.interpret.run as R
    from film.interpret.arrivals import Arrival

    path = {"rim_w": 50.0, "frames": [{"dt": 0.0, "balls": [[0.5, 0.0, 0.8]]}, {"dt": 0.3, "balls": [[0.5, 0.6, 0.8]]}]}
    arrivals = [Arrival(t=float(i), frame=i * 10, rim_id=10 + i, rim_box=(100.0, 60.0, 150.0, 90.0), source="ball", conf=0.8) for i in range(10)]
    rims = pd.DataFrame([{"frame": a.frame, "track_id": a.rim_id, "x1": 100.0, "y1": 60.0, "x2": 150.0, "y2": 90.0} for a in arrivals])
    monkeypatch.setattr(R, "ball_paths", lambda gd, m, arr, r, det: [path] * len(arr))
    given = []

    class Recording(FakeClassifier):
        def predict(self, features):
            given.append(dict(zip(R.FEATURES, features)))
            return super().predict(features)

    manifest = type("M", (), {"width": 1920})()
    R.classify_arrivals(None, manifest, arrivals, rims, Recording(), None)
    assert len(given) == 10 and all(f["column_frames"] > 0 and f["sideways"] < 0.05 for f in given)


class FakePictures:
    """A scripted picture classifier: the same class probabilities for every view; counts the views it saw."""

    def __init__(self, scores=(0.9, 0.05, 0.05), version="picclf@fake"):
        self.row, self.version, self.seen = list(scores), version, 0

    def scores(self, views):
        views = list(views)
        self.seen += len(views)
        return np.array([self.row] * len(views), dtype=np.float32)


def test_the_picture_classifier_gives_a_second_opinion_and_its_answers_are_kept(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    gd = game_dir(m.game_id)
    _tracks(gd)
    pics = FakePictures()
    kwargs = dict(classifier=FakeClassifier([("made", 0.9)]), ball_detector=FakeDetector(), log=lambda *a: None)
    [e] = interpret(m.game_id, picture_classifier=pics, **kwargs)
    assert pics.seen == 1 and e.made is True and e.signals["agreement"] == "strong" and e.confidence >= 0.9
    assert e.signals["clf_pictures"][0] == "made" and e.signals["clf_path"][0] == "made"
    assert Manifest.load(gd).model_versions["picclf"] == "picclf@fake"
    interpret(m.game_id, picture_classifier=pics, **kwargs)
    assert pics.seen == 1                                                     # a re-run reuses the stored answer
    newer = FakePictures(scores=(0.05, 0.9, 0.05), version="picclf@newer")
    [e] = interpret(m.game_id, picture_classifier=newer, **kwargs)
    assert newer.seen == 1 and e.signals["agreement"] == "split" and "classifiers-disagree" in e.flags


def test_without_a_picture_classifier_the_path_classifier_carries_on_alone(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    _tracks(game_dir(m.game_id))
    [e] = interpret(m.game_id, classifier=FakeClassifier([("made", 0.9)]), ball_detector=FakeDetector(), log=lambda *a: None)
    assert e.confidence == 0.63 and e.signals["agreement"] is None and e.signals["clf_pictures"] is None
