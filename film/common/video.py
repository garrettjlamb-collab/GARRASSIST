from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    codec: str


def probe(path: Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(container.duration / av.time_base)
        return VideoInfo(
            duration_s=duration,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(stream.average_rate),
            codec=stream.codec_context.name,
        )


def iter_frames(
    path: Path, fps: float, start: float = 0.0, end: float | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_seconds, frame_bgr) about `fps` times per second of video, decoding on the fly."""
    step = 1.0 / fps
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start > 0:
            container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
        next_t = start
        for frame in container.decode(stream):
            t = frame.time
            if t is None or t < start - 1e-6:
                continue
            if end is not None and t > end + 1e-6:
                break
            if t + 1e-6 >= next_t:
                yield t, frame.to_ndarray(format="bgr24")
                while next_t <= t + 1e-6:
                    next_t += step
