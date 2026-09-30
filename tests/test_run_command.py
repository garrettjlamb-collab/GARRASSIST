import types

from film.cli import main


def test_run_chains_the_stages(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    calls = []
    import film.export.clips as clips
    import film.interpret.run as interpret_mod
    import film.perceive.run as perceive_mod

    monkeypatch.setattr(perceive_mod, "perceive", lambda game_id, **kw: calls.append(("perceive", game_id, kw.get("limit_s"))))
    monkeypatch.setattr(interpret_mod, "interpret", lambda game_id, **kw: calls.append(("interpret", game_id)) or [])
    monkeypatch.setattr(clips, "clips_for_game", lambda game_id, **kw: calls.append(("clips", game_id)) or 0)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld", "--limit-s", "2"]) == 0
    assert calls == [("perceive", "2026-01-01-synthetic", 2.0), ("interpret", "2026-01-01-synthetic"), ("clips", "2026-01-01-synthetic")]
