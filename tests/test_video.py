import numpy as np

from film.common.video import iter_frames, probe


def test_probe(synthetic_video):
    info = probe(synthetic_video)
    assert info.width == 320 and info.height == 240
    assert abs(info.fps - 30) < 0.5
    assert 2.8 <= info.duration_s <= 3.2
    assert info.codec


def test_iter_frames_spacing(synthetic_video):
    frames = list(iter_frames(synthetic_video, fps=10))
    assert 28 <= len(frames) <= 31
    times = [t for t, _ in frames]
    assert times == sorted(times)
    gaps = np.diff(times)
    assert np.all(gaps > 0.05) and np.all(gaps < 0.15)
    t0, img = frames[0]
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8


def test_iter_frames_window(synthetic_video):
    frames = list(iter_frames(synthetic_video, fps=10, start=1.0, end=2.0))
    assert 9 <= len(frames) <= 12
    assert frames[0][0] >= 0.95
    assert frames[-1][0] <= 2.05
