import pandas as pd

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.perceive.detectors import Box, FakeDetector
from film.perceive.run import perceive
from film.perceive.tracks import read_tracks


def test_perceive_writes_tracks_motion_and_marks_stage(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    rim = Box(100, 60, 150, 90, 0.9, "rim")
    rims = FakeDetector(fn=lambda img, i: [rim], name="rim@fake")
    balls = FakeDetector(fn=lambda img, i: [Box(10, 10, 20, 20, 0.7, "ball")] if i % 3 == 0 else [], name="ball@fake")
    info = perceive(m.game_id, device="cpu", rim_detector=rims, ball_detector=balls, log=lambda *a: None)
    gd = game_dir(m.game_id)
    rim_rows = read_tracks(gd, "rim")
    ball_rows = read_tracks(gd, "ball")
    assert 28 <= len(rim_rows) <= 31
    assert set(rim_rows["track_id"]) == {3}          # no template side: a tracker id, above the fixed 1 and 2
    assert 9 <= len(ball_rows) <= 11
    assert (ball_rows["ref_id"] == 3).all()
    assert ball_rows["x1"].iloc[0] == 15 and ball_rows["y1"].iloc[0] == 10   # crop origin is (5, 0), see note
    motion = pd.read_parquet(gd / "motion.parquet")
    assert list(motion.columns) == ["frame", "t", "rim_id", "energy", "energy_global"]
    assert len(motion) == len(rim_rows)
    assert not (gd / "scoreboard.jsonl").exists()
    saved = Manifest.load(gd)
    assert saved.stages["perceive"]["status"] == "done" and saved.stages["perceive"]["limit_s"] is None
    assert saved.model_versions == {"rim": "rim@fake", "ball": "ball@fake"}
    assert info["frames"] == len(rim_rows)


def test_perceive_warns_loudly_when_rims_are_rare(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    rim = Box(100, 60, 150, 90, 0.9, "rim")
    other = Box(250, 60, 300, 90, 0.8, "rim")
    rims = FakeDetector(fn=lambda img, i: [rim, other] if i < 2 else [], name="rim@fake")   # 4 rows, 2 frames
    lines = []
    perceive(m.game_id, device="cpu", rim_detector=rims, ball_detector=FakeDetector(name="ball@fake"), log=lines.append)
    warning = [line for line in lines if "WARNING" in line]
    assert len(warning) == 1 and "7%" in warning[0] and "template" in warning[0]
    quiet = []
    perceive(m.game_id, device="cpu", rim_detector=FakeDetector(fn=lambda img, i: [rim], name="rim@fake"),
             ball_detector=FakeDetector(name="ball@fake"), log=quiet.append)
    assert not [line for line in quiet if "WARNING" in line]


def test_perceive_of_no_frames_does_not_warn(synthetic_video, tmp_path, monkeypatch):
    import film.perceive.run as run_mod

    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    monkeypatch.setattr(run_mod, "iter_frames", lambda *a, **kw: iter(()))
    lines = []
    info = perceive(m.game_id, device="cpu", rim_detector=FakeDetector(name="rim@fake"),
                    ball_detector=FakeDetector(name="ball@fake"), log=lines.append)
    assert info["frames"] == 0 and not [line for line in lines if "WARNING" in line]
