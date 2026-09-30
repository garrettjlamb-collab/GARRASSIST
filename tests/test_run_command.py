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


def _stub_stages(monkeypatch, calls):
    import film.export.clips as clips
    import film.interpret.run as interpret_mod
    import film.perceive.run as perceive_mod

    monkeypatch.setattr(perceive_mod, "perceive", lambda game_id, **kw: calls.append("perceive"))
    monkeypatch.setattr(interpret_mod, "interpret", lambda game_id, **kw: calls.append("interpret") or [])
    monkeypatch.setattr(clips, "clips_for_game", lambda game_id, **kw: calls.append("clips") or 0)


def test_run_refuses_an_unknown_profile_without_perceiving(synthetic_video, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    calls = []
    _stub_stages(monkeypatch, calls)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01"]) == 2    # a 3 s clip: no overlay, so unknown
    assert calls == []
    err = capsys.readouterr().err
    assert "--profile" in err and "hudl-tactical" in err


def test_perceive_refuses_an_unknown_profile(synthetic_video, tmp_path, monkeypatch, capsys):
    from film.ingest.run import ingest

    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    calls = []
    _stub_stages(monkeypatch, calls)
    m = ingest(synthetic_video, date="2026-01-01")
    assert m.profile == "unknown"
    assert main(["perceive", m.game_id]) == 2
    assert calls == [] and "--profile" in capsys.readouterr().err
