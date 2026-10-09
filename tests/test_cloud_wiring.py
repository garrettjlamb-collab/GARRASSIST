import shutil
from collections import namedtuple

from film.cli import main
from film.common import cloud
from film.common.cloud import local_source
from film.common.paths import game_dir
from film.export.clips import clip_path, clips_for_game
from film.ingest.run import ingest
from film.interpret.events import Event, save_events


def _own_film(synthetic_video, tmp_path):
    """A copy per test: the cloud state is keyed by the film's path."""
    return shutil.copy(synthetic_video, tmp_path / "film.mp4")


def _events(gd, n=2):
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i * 0.5, rim_id=1, quarter=1, clock_s=100.0, team=None,
                    made=False, confidence=0.5, clip={"start": 0.5 + i * 0.5, "end": 1.5 + i * 0.5}, signals={}, flags=[])
              for i in range(n)]
    save_events(gd, events)
    return events


def test_ingest_pulls_a_cloud_only_film_and_puts_it_back(synthetic_video, tmp_path, cloud_env):
    film = cloud_env.offload(_own_film(synthetic_video, tmp_path))
    m = ingest(film, date="2026-01-01", profile="test")
    assert m.duration_s > 2
    assert len(cloud_env.downloads) == 1 and len(cloud_env.evictions) == 1 and cloud_env.is_cloud_only(film)


def test_a_film_that_was_on_the_mac_stays_on_the_mac(synthetic_video, tmp_path, cloud_env):
    film = _own_film(synthetic_video, tmp_path)
    m = ingest(film, date="2026-01-01", profile="test")
    _events(game_dir(m.game_id))
    assert clips_for_game(m.game_id, log=lambda *a: None) == 2
    assert cloud_env.downloads == [] and cloud_env.evictions == []


def test_run_pulls_once_and_puts_back_once_across_all_stages(synthetic_video, tmp_path, cloud_env, monkeypatch):
    import film.export.clips as clips
    import film.interpret.run as interpret_mod
    import film.perceive.run as perceive_mod

    film = cloud_env.offload(_own_film(synthetic_video, tmp_path))
    seen = []

    def stage(name):
        def run(game_id, **kw):
            with local_source(cloud.game_film(game_id), log=lambda *a: None):       # as each real stage does
                seen.append((name, cloud_env.is_cloud_only(film), len(cloud_env.evictions)))
        return run

    monkeypatch.setattr(perceive_mod, "perceive", stage("perceive"))
    monkeypatch.setattr(interpret_mod, "interpret", stage("interpret"))
    monkeypatch.setattr(clips, "clips_for_game", stage("clips"))
    assert main(["run", str(film), "--date", "2026-01-01", "--profile", "handheld"]) == 0
    assert seen == [("perceive", False, 0), ("interpret", False, 0), ("clips", False, 0)]
    assert len(cloud_env.downloads) == 1 and len(cloud_env.evictions) == 1 and cloud_env.is_cloud_only(film)


def test_clips_with_nothing_to_cut_pulls_nothing(synthetic_video, tmp_path, cloud_env):
    film = _own_film(synthetic_video, tmp_path)
    m = ingest(film, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _events(gd)
    assert clips_for_game(m.game_id, log=lambda *a: None) == 2
    cloud_env.offload(film)
    assert clips_for_game(m.game_id, log=lambda *a: None) == 0
    assert cloud_env.downloads == [] and cloud_env.is_cloud_only(film)


def test_clips_pulls_when_a_clip_is_missing(synthetic_video, tmp_path, cloud_env):
    film = _own_film(synthetic_video, tmp_path)
    m = ingest(film, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = _events(gd)
    cloud_env.offload(film)
    lines = []
    assert clips_for_game(m.game_id, log=lines.append) == 2
    assert clip_path(gd, events[1].id).exists()
    assert len(cloud_env.downloads) == 1 and len(cloud_env.evictions) == 1 and cloud_env.is_cloud_only(film)
    assert any("pulling film.mp4" in line for line in lines)


def test_a_stage_that_always_reads_the_film_holds_it_for_the_whole_call(synthetic_video, tmp_path, cloud_env):
    film = _own_film(synthetic_video, tmp_path)
    m = ingest(film, date="2026-01-01", profile="test")
    cloud_env.offload(film)
    lines = []

    @cloud.needs_film
    def stage(game_id, log=print):
        return cloud_env.is_cloud_only(film)

    assert stage(m.game_id, log=lines.append) is False
    assert cloud_env.is_cloud_only(film) and "pulling film.mp4" in lines[0]


def test_the_real_stages_are_wrapped():
    from film.perceive.run import perceive
    from film.train.prelabel import autolabel_rims, prelabel_balls, prelabel_rims
    from film.train.rimclf_dataset import build_rimclf_candidates, ensure_views

    for fn in (perceive, prelabel_rims, autolabel_rims, prelabel_balls, build_rimclf_candidates, ensure_views, clips_for_game):
        assert hasattr(fn, "__wrapped__"), fn.__name__


def test_a_cloud_error_exits_2_with_the_message(synthetic_video, tmp_path, cloud_env, monkeypatch, capsys):
    film = cloud_env.offload(_own_film(synthetic_video, tmp_path))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(cloud.shutil, "disk_usage", lambda p: usage(10**12, 10**12, 10**9))
    assert main(["ingest", str(film), "--date", "2026-01-01", "--profile", "handheld"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: not enough room to pull film.mp4 from iCloud")
    assert not (tmp_path / "games" / "2026-01-01-film").exists()
