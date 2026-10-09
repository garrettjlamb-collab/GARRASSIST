import os
import subprocess
import sys
from collections import namedtuple

import pytest

from film.common import cloud
from film.common.cloud import CloudError, film_state, hold, lazy_source, local_source


def _film(tmp_path, name="game.mp4"):
    path = tmp_path / name
    path.write_bytes(b"x" * 1000)
    return path


def _quiet(*a):
    pass


def test_a_local_film_is_never_pulled_or_offloaded(tmp_path, cloud_env):
    film = _film(tmp_path)
    with local_source(film, log=_quiet) as path:
        assert path == film
    assert cloud_env.downloads == [] and cloud_env.evictions == []
    assert film_state(film) == "local"


def test_a_cloud_only_film_is_pulled_then_put_back(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    lines = []
    with local_source(film, log=lines.append):
        assert not cloud_env.is_cloud_only(film) and film_state(film) == "pulled"
    assert len(cloud_env.downloads) == 1 and len(cloud_env.evictions) == 1
    assert cloud_env.is_cloud_only(film) and film_state(film) == "cloud"
    assert "pulling game.mp4" in lines[0] and "from iCloud" in lines[0]
    assert lines[-1] == "game.mp4 is back in iCloud only"


def test_nested_holds_offload_once_at_the_outermost_exit(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    with local_source(film, log=_quiet):
        with local_source(film, log=_quiet):
            pass
        assert cloud_env.evictions == [] and not cloud_env.is_cloud_only(film)
        with local_source(film, log=_quiet):
            pass
        assert cloud_env.evictions == []
    assert len(cloud_env.downloads) == 1 and len(cloud_env.evictions) == 1


def test_another_live_process_holding_the_film_blocks_the_offload(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    with local_source(film, log=_quiet):
        other = cloud._state_dir(film) / "holds" / f"{os.getppid()}-abcdef12"
        other.write_text("")
    assert cloud_env.evictions == [] and film_state(film) == "pulled"
    other.unlink()
    with local_source(film, log=_quiet):                      # the next job on this film puts it back
        pass
    assert len(cloud_env.evictions) == 1 and cloud_env.is_cloud_only(film)


def test_a_dead_process_hold_is_ignored(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    with local_source(film, log=_quiet):
        stale = cloud._state_dir(film) / "holds" / f"{gone.pid}-abcdef12"
        stale.write_text("")
    assert len(cloud_env.evictions) == 1 and not stale.exists()


def test_too_little_room_raises_before_any_download(tmp_path, cloud_env, monkeypatch):
    film = cloud_env.offload(_film(tmp_path))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(cloud.shutil, "disk_usage", lambda p: usage(10**12, 10**12, 2 * 10**9))
    with pytest.raises(CloudError, match="not enough room"):
        with local_source(film, log=_quiet):
            raise AssertionError("the job must not start")
    assert cloud_env.downloads == [] and list((cloud._state_dir(film) / "holds").iterdir()) == []


def test_the_cushion_is_configurable(tmp_path, cloud_env, monkeypatch):
    film = cloud_env.offload(_film(tmp_path))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(cloud.shutil, "disk_usage", lambda p: usage(10**12, 10**12, 2 * 10**9))
    monkeypatch.setenv("FILM_CLOUD_CUSHION_GB", "1")
    with local_source(film, log=_quiet):
        pass
    assert len(cloud_env.downloads) == 1


def test_a_failed_download_raises_and_leaves_no_hold(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    cloud_env.fail_download = "could not pull game.mp4 from iCloud: offline"
    with pytest.raises(CloudError, match="offline"):
        with local_source(film, log=_quiet):
            raise AssertionError("the job must not start")
    assert list((cloud._state_dir(film) / "holds").iterdir()) == [] and film_state(film) == "cloud"


def test_a_pull_that_never_finishes_times_out(tmp_path, cloud_env, monkeypatch):
    film = cloud_env.offload(_film(tmp_path))
    cloud_env.stall_download = True
    monkeypatch.setenv("FILM_CLOUD_TIMEOUT_S", "4")
    now = [0.0]
    monkeypatch.setattr(cloud, "_clock", lambda: now[0])
    monkeypatch.setattr(cloud, "_sleep", lambda s: now.__setitem__(0, now[0] + s))
    with pytest.raises(CloudError, match="did not arrive"):
        with local_source(film, log=_quiet):
            raise AssertionError("the job must not start")
    assert list((cloud._state_dir(film) / "holds").iterdir()) == []


def test_a_long_pull_reports_its_progress(tmp_path, cloud_env, monkeypatch):
    film = cloud_env.offload(_film(tmp_path))
    cloud_env.stall_download = True
    now = [0.0]

    def sleep(s):
        now[0] += s
        if now[0] >= 70:
            cloud_env.cloud.discard(film.resolve())

    monkeypatch.setattr(cloud, "_clock", lambda: now[0])
    monkeypatch.setattr(cloud, "_sleep", sleep)
    lines = []
    with local_source(film, log=lines.append):
        pass
    assert [line for line in lines if "still pulling" in line] == ["  still pulling game.mp4: 0.5 min", "  still pulling game.mp4: 1.0 min"]


def test_an_offload_that_does_not_take_is_logged_and_retried_by_the_next_job(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    cloud_env.refuse_evict = True
    lines = []
    with local_source(film, log=lines.append):
        pass
    assert lines[-1] == "game.mp4 is still on the Mac: macOS would not offload it"
    assert len(cloud_env.evictions) == 2 and film_state(film) == "pulled"       # one retry
    cloud_env.refuse_evict = False
    with local_source(film, log=_quiet):
        pass
    assert cloud_env.is_cloud_only(film) and film_state(film) == "cloud" and len(cloud_env.downloads) == 1


def test_an_evict_that_reports_success_but_does_not_take_counts_as_refused(tmp_path, cloud_env, monkeypatch):
    film = cloud_env.offload(_film(tmp_path))
    with local_source(film, log=_quiet):
        monkeypatch.setattr(cloud_env, "evict", lambda path: cloud_env.evictions.append(path) or True)
        now = [0.0]
        monkeypatch.setattr(cloud, "_clock", lambda: now[0])
        monkeypatch.setattr(cloud, "_sleep", lambda s: now.__setitem__(0, now[0] + s))
    assert film_state(film) == "pulled" and len(cloud_env.evictions) == 2


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_the_film_goes_back_when_the_job_raises(tmp_path, cloud_env, error):
    film = cloud_env.offload(_film(tmp_path))
    with pytest.raises(error):
        with local_source(film, log=_quiet):
            raise error()
    assert cloud_env.is_cloud_only(film) and len(cloud_env.evictions) == 1


def test_a_film_the_coach_offloaded_mid_job_is_left_alone(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    with local_source(film, log=_quiet):
        cloud_env.offload(film)
    assert cloud_env.evictions == [] and film_state(film) == "cloud"


def test_release_is_idempotent(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    h = hold(film, log=_quiet)
    h.release()
    h.release()
    assert len(cloud_env.evictions) == 1


def test_lazy_source_pulls_only_when_the_film_is_asked_for(tmp_path, cloud_env):
    film = cloud_env.offload(_film(tmp_path))
    with lazy_source(film, log=_quiet) as get:
        pass
    assert cloud_env.downloads == []
    with lazy_source(film, log=_quiet) as get:
        assert get() == film and get() == film
        assert len(cloud_env.downloads) == 1
    assert len(cloud_env.evictions) == 1 and cloud_env.is_cloud_only(film)


def test_film_state(tmp_path, cloud_env):
    film = _film(tmp_path)
    assert film_state(tmp_path / "nope.mp4") == "missing"
    assert film_state(film) == "local"
    cloud_env.offload(film)
    assert film_state(film) == "cloud"


def test_two_films_keep_separate_state(tmp_path, cloud_env):
    a, b = cloud_env.offload(_film(tmp_path, "a.mp4")), _film(tmp_path, "b.mp4")
    with local_source(a, log=_quiet), local_source(b, log=_quiet):
        pass
    assert cloud_env.evictions == [a]


def test_is_cloud_only_reads_the_dataless_flag(tmp_path, monkeypatch):
    film = _film(tmp_path)
    assert cloud.is_cloud_only(film) is False
    assert cloud.is_cloud_only(tmp_path / "nope.mp4") is False
    stat = namedtuple("stat", "st_flags")
    monkeypatch.setattr(cloud.os, "stat", lambda p: stat(cloud.SF_DATALESS | 0x40))
    assert cloud.is_cloud_only(film) is True


def test_the_brctl_backend_reports_a_failed_download(tmp_path, monkeypatch):
    done = namedtuple("done", "returncode stdout stderr")
    calls = []
    monkeypatch.setattr(cloud.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or done(1, "", "brctl: no network"))
    with pytest.raises(CloudError, match="no network"):
        cloud.Backend().download(tmp_path / "game.mp4")
    assert cloud.Backend().evict(tmp_path / "game.mp4") is False
    assert [c[:2] for c in calls] == [["brctl", "download"], ["brctl", "evict"]]
