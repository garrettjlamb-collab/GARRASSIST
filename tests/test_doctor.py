from film.cli import main


def test_doctor_runs(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "device" in out and "ffmpeg" in out


def test_doctor_lists_where_each_games_film_is(synthetic_video, tmp_path, cloud_env, capsys):
    import shutil

    from film.ingest.run import ingest

    here = shutil.copy(synthetic_video, tmp_path / "here.mp4")
    away = shutil.copy(synthetic_video, tmp_path / "away.mp4")
    gone = shutil.copy(synthetic_video, tmp_path / "gone.mp4")
    for film in (here, away, gone):
        ingest(film, date="2026-01-01", profile="test")
    cloud_env.offload(away)
    gone.unlink()
    assert main(["doctor"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("film ")]
    assert lines == ["film cloud    2026-01-01-away", "film missing  2026-01-01-gone", "film local    2026-01-01-here"]
