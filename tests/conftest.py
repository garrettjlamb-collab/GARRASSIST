from __future__ import annotations

from pathlib import Path
from typing import Callable

import av
import numpy as np
import pytest


def make_test_video(
    path: Path,
    seconds: int = 3,
    fps: int = 30,
    size: tuple[int, int] = (320, 240),
    painter: Callable[[int, np.ndarray], None] | None = None,
) -> Path:
    """Encode a small mpeg4 video. `painter(i, img)` draws frame i in place; default draws a sweeping bar."""
    width, height = size
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    for i in range(seconds * fps):
        img = np.zeros((height, width, 3), np.uint8)
        if painter is None:
            x = (i * 5) % width
            img[:, x : x + 10] = 255
        else:
            painter(i, img)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    return make_test_video(tmp_path_factory.mktemp("video") / "synthetic.mp4")


@pytest.fixture
def video_factory(tmp_path):
    def _make(name="v.mp4", **kwargs) -> Path:
        return make_test_video(tmp_path / name, **kwargs)

    return _make
