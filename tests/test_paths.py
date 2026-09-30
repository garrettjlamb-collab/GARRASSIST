from pathlib import Path

from film.common import paths


def test_games_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "g"))
    assert paths.games_dir() == (tmp_path / "g").resolve()
    assert paths.game_dir("abc") == (tmp_path / "g" / "abc").resolve()


def test_slugify():
    assert paths.slugify("JV vs Marcos - Tactical") == "jv-vs-marcos-tactical"
    assert paths.slugify("   ") == "game"


def test_game_id_for():
    src = Path("/x/JV vs Marcos - Tactical.mp4")
    assert paths.game_id_for(src, "2025-02-05") == "2025-02-05-jv-vs-marcos-tactical"
