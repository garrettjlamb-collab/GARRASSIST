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


def test_reingest_keeps_later_stages_only_when_nothing_changed(synthetic_video, tmp_path, monkeypatch):
    from film.common.manifest import Manifest

    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="hudl-tactical", fps=10.0)
    m.mark("perceive", "done")
    m.model_versions["rim"] = "rim@template"
    m.save(tmp_path / "games" / m.game_id)
    again = ingest(synthetic_video, date="2026-01-01", profile="hudl-tactical", fps=10.0)
    assert again.stages["perceive"]["status"] == "done" and again.model_versions == {"rim": "rim@template"}
    other = ingest(synthetic_video, date="2026-01-01", profile="handheld", fps=10.0)
    assert set(other.stages) == {"ingest"} and other.model_versions == {}
    assert set(Manifest.load(tmp_path / "games" / m.game_id).stages) == {"ingest"}


def test_a_new_prefix_on_the_file_name_is_not_part_of_the_game_id(synthetic_video, tmp_path, monkeypatch):
    import shutil
    from film.ingest.run import ingest

    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    film = tmp_path / "NEW Marcos de Niza vs Tempe - Angle 1.mp4"
    shutil.copy(synthetic_video, film)
    assert ingest(film, date="2026-01-01", profile="test").game_id == "2026-01-01-marcos-de-niza-vs-tempe-angle-1"
