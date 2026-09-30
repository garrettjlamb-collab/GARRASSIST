import imageio_ffmpeg

from film.common import ffmpeg as ff


def test_prefers_system_ffmpeg(monkeypatch):
    monkeypatch.setattr(ff.shutil, "which", lambda name: "/usr/local/bin/ffmpeg")
    assert ff.ffmpeg_exe() == "/usr/local/bin/ffmpeg"


def test_falls_back_to_bundled_binary(monkeypatch):
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/bundled/ffmpeg")
    assert ff.ffmpeg_exe() == "/bundled/ffmpeg"
