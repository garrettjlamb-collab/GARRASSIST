import shutil

from fastapi.testclient import TestClient

from film.common import cloud
from film.common.cloud import film_state
from film.ingest.run import ingest
from film.review.app import create_app
from film.review.films import Films


def _quiet(*a):
    pass


def _game(synthetic_video, tmp_path):
    film = shutil.copy(synthetic_video, tmp_path / "film.mp4")
    return ingest(film, date="2026-01-01", profile="test").game_id, film


def test_a_film_on_the_mac_plays_straight_away(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    client = TestClient(create_app(Films(log=_quiet)))
    assert client.get(f"/api/games/{game_id}/film").json() == {"state": "local", "size_bytes": film.stat().st_size, "error": None}
    assert client.get(f"/video/{game_id}", headers={"Range": "bytes=0-99"}).status_code == 206
    assert cloud_env.downloads == []


def test_a_cloud_only_film_is_pulled_on_request_then_plays(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    cloud_env.offload(film)
    films = Films(log=_quiet)
    client = TestClient(create_app(films))
    assert client.get(f"/api/games/{game_id}/film").json()["state"] == "cloud"
    assert cloud_env.downloads == []                          # looking is not asking
    assert client.post(f"/api/games/{game_id}/film").json()["state"] in ("pulling", "local")
    films.wait()
    assert client.get(f"/api/games/{game_id}/film").json()["state"] == "local"
    assert client.get(f"/video/{game_id}", headers={"Range": "bytes=0-99"}).status_code == 206
    assert len(cloud_env.downloads) == 1 and cloud_env.evictions == []


def test_the_video_url_answers_503_while_the_film_is_coming_down(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    cloud_env.offload(film)
    cloud_env.stall_download = True                           # the download has started and not finished
    client = TestClient(create_app(Films(log=_quiet)))
    r = client.get(f"/video/{game_id}")
    assert r.status_code == 503 and "iCloud" in r.json()["detail"] and r.headers["retry-after"] == "30"
    assert client.get(f"/api/games/{game_id}/film").json()["state"] == "pulling"
    assert len(cloud_env.downloads) == 1
    client.get(f"/video/{game_id}")
    assert len(cloud_env.downloads) == 1                      # a second request does not start a second pull
    cloud_env.cloud.discard(film.resolve())                   # let the waiting thread finish


def test_a_failed_pull_is_shown_and_retried_on_the_next_request(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    cloud_env.offload(film)
    cloud_env.fail_download = "could not pull film.mp4 from iCloud: offline"
    films = Films(log=_quiet)
    client = TestClient(create_app(films))
    client.post(f"/api/games/{game_id}/film")
    films.wait()
    status = client.get(f"/api/games/{game_id}/film").json()
    assert status["state"] == "cloud" and "offline" in status["error"]
    cloud_env.fail_download = None
    client.post(f"/api/games/{game_id}/film")
    films.wait()
    assert client.get(f"/api/games/{game_id}/film").json() == {"state": "local", "size_bytes": film.stat().st_size, "error": None}


def test_an_idle_film_goes_back_and_a_watched_one_stays(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    cloud_env.offload(film)
    now = [0.0]
    films = Films(idle_s=1800, clock=lambda: now[0], log=_quiet)
    client = TestClient(create_app(films))
    client.post(f"/api/games/{game_id}/film")
    films.wait()
    now[0] = 1700.0
    assert films.sweep() == [] and not cloud_env.is_cloud_only(film)
    assert client.get(f"/video/{game_id}", headers={"Range": "bytes=0-99"}).status_code == 206     # watching resets the clock
    now[0] = 3400.0
    assert films.sweep() == []
    now[0] = 3501.0
    assert films.sweep() == [game_id]
    assert cloud_env.is_cloud_only(film) and len(cloud_env.evictions) == 1
    assert client.get(f"/video/{game_id}").status_code == 503                                      # and it can be pulled again
    films.wait()
    assert len(cloud_env.downloads) == 2


def test_a_film_that_started_on_the_mac_is_never_offloaded_by_the_server(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    now = [0.0]
    films = Films(idle_s=10, clock=lambda: now[0], log=_quiet)
    client = TestClient(create_app(films))
    assert client.get(f"/video/{game_id}", headers={"Range": "bytes=0-99"}).status_code == 206
    now[0] = 100.0
    assert films.sweep() == [game_id]
    films.release_all()
    assert cloud_env.evictions == [] and film_state(film) == "local"


def test_stopping_the_server_puts_every_pulled_film_back(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    cloud_env.offload(film)
    films = Films(log=_quiet)
    with TestClient(create_app(films)) as client:
        client.post(f"/api/games/{game_id}/film")
        films.wait()
        assert not cloud_env.is_cloud_only(film)
    assert cloud_env.is_cloud_only(film) and len(cloud_env.evictions) == 1


def test_a_job_and_the_server_share_a_film(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    cloud_env.offload(film)
    films = Films(log=_quiet)
    client = TestClient(create_app(films))
    with cloud.local_source(film, log=_quiet):                # a job pulled it
        assert client.post(f"/api/games/{game_id}/film").json()["state"] == "local"
    assert cloud_env.evictions == []                          # the job ended; the coach is still watching
    films.release_all()
    assert cloud_env.is_cloud_only(film) and len(cloud_env.downloads) == 1


def test_a_missing_film_is_a_404_not_a_crash(synthetic_video, tmp_path, cloud_env):
    game_id, film = _game(synthetic_video, tmp_path)
    film.unlink()
    client = TestClient(create_app(Films(log=_quiet)))
    assert client.get(f"/api/games/{game_id}/film").json()["state"] == "missing"
    assert client.get(f"/video/{game_id}").status_code == 404


def test_the_wait_page_is_served(cloud_env):
    client = TestClient(create_app(Films(log=_quiet)))
    page = client.get("/film.html?game=g&t=12.0")
    assert page.status_code == 200 and "Pulling the full film from iCloud" in page.text
    assert "/film.html?game=" in client.get("/").text
