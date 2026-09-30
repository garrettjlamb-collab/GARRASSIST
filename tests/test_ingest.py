from film.ingest.run import ingest


def test_ingest_writes_manifest(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    assert m.game_id == "2026-01-01-synthetic"
    assert m.profile == "test"
    assert m.width == 320 and m.height == 240
    assert 2.8 <= m.duration_s <= 3.2
    assert m.fps == 10.0
    assert (tmp_path / "games" / m.game_id / "manifest.json").exists()
    assert m.stages["ingest"]["status"] == "done"


def test_ingest_profile_unknown_for_short_video(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01")
    assert m.profile == "unknown"
    assert m.stages["ingest"]["samples"] == 0
