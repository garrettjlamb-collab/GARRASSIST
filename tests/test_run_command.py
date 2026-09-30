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


def _done_game(synthetic_video, tmp_path, monkeypatch):
    from film.common.paths import game_dir
    from film.ingest.run import ingest

    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="handheld")
    m.mark("perceive", "done")
    m.mark("interpret", "done")
    gd = game_dir(m.game_id)
    m.save(gd)
    return gd


def test_run_skips_stages_already_done(synthetic_video, tmp_path, monkeypatch):
    _done_game(synthetic_video, tmp_path, monkeypatch)
    calls = []
    _stub_stages(monkeypatch, calls)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld"]) == 0
    assert calls == ["clips"]


def test_run_force_reruns_every_stage(synthetic_video, tmp_path, monkeypatch):
    _done_game(synthetic_video, tmp_path, monkeypatch)
    calls = []
    _stub_stages(monkeypatch, calls)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld", "--force"]) == 0
    assert calls == ["perceive", "interpret", "clips"]


def test_run_prunes_stale_clips_and_counts_orphaned_decisions(synthetic_video, tmp_path, monkeypatch, capsys):
    from film.interpret.events import Event, save_events
    from film.review.decisions import save_decision

    gd = _done_game(synthetic_video, tmp_path, monkeypatch)
    save_events(gd, [Event(id="shot-000010-r1", type="shot", t=1.0, rim_id=1, quarter=None, clock_s=None, team=None,
                           made=None, confidence=0.3, clip={"start": 0.0, "end": 2.5}, signals={})])
    clips = gd / "clips"
    clips.mkdir()
    for name in ("shot-000010-r1.mp4", "shot-000099-r7.mp4", "shot-000099-r7.part.mp4", "shot-000050-r3.part.mp4"):
        (clips / name).write_bytes(b"x")
    save_decision(gd, "shot-000010-r1", {"status": "confirmed"})
    save_decision(gd, "shot-000099-r7", {"status": "rejected"})
    calls = []
    _stub_stages(monkeypatch, calls)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld"]) == 0
    assert sorted(p.name for p in clips.iterdir()) == ["shot-000010-r1.mp4"]
    out = capsys.readouterr().out
    assert "pruned 3 stale clip files" in out and "1 review decision no longer matches an event" in out
    assert (gd / "review.json").exists() and "shot-000099-r7" in (gd / "review.json").read_text()


def test_a_fresh_perceive_always_gets_a_fresh_interpret(synthetic_video, tmp_path, monkeypatch):
    from film.common.manifest import Manifest

    gd = _done_game(synthetic_video, tmp_path, monkeypatch)
    m = Manifest.load(gd)
    m.stages["perceive"]["status"] = "failed"
    m.save(gd)
    calls = []
    _stub_stages(monkeypatch, calls)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld"]) == 0
    assert calls == ["perceive", "interpret", "clips"]


def test_a_stage_done_under_a_limit_is_rerun_by_a_full_run(synthetic_video, tmp_path, monkeypatch, capsys):
    from film.common.manifest import Manifest

    gd = _done_game(synthetic_video, tmp_path, monkeypatch)
    m = Manifest.load(gd)
    m.stages["perceive"]["limit_s"] = 2.0
    m.save(gd)
    calls = []
    _stub_stages(monkeypatch, calls)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld"]) == 0
    assert calls == ["perceive", "interpret", "clips"]
    assert "--limit-s 2" in capsys.readouterr().out
    calls.clear()
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld", "--limit-s", "2"]) == 0
    assert calls == ["clips"]                               # the same limit again: nothing new to do


def test_a_full_perceive_is_skipped_by_any_later_run(synthetic_video, tmp_path, monkeypatch):
    from film.common.manifest import Manifest

    gd = _done_game(synthetic_video, tmp_path, monkeypatch)
    m = Manifest.load(gd)
    m.stages["perceive"]["limit_s"] = None
    m.save(gd)
    calls = []
    _stub_stages(monkeypatch, calls)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld"]) == 0
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "handheld", "--limit-s", "2"]) == 0
    assert calls == ["clips", "clips"]
