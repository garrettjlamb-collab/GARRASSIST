from __future__ import annotations

from datetime import datetime
from pathlib import Path

from film.common.manifest import Manifest
from film.common.paths import game_dir, game_id_for
from film.common.video import iter_frames, probe
from film.ingest.profile import detect_profile

SAMPLE_TIMES = (60.0, 120.0, 180.0, 240.0, 300.0)


def sample_frames(source: Path, duration_s: float, times=SAMPLE_TIMES) -> list:
    frames = []
    for t in times:
        if t >= duration_s - 1:
            break
        for _, frame in iter_frames(source, fps=2.0, start=t, end=t + 0.6):
            frames.append(frame)
            break
    return frames


def ingest(source: Path, date: str | None = None, profile: str | None = None, fps: float = 10.0) -> Manifest:
    source = Path(source).resolve()
    if date is None:
        date = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d")
    info = probe(source)
    samples = []
    if profile is None:
        samples = sample_frames(source, info.duration_s)
        profile = detect_profile(samples)
    manifest = Manifest(
        game_id=game_id_for(source, date), source=str(source), date=date,
        duration_s=info.duration_s, width=info.width, height=info.height,
        src_fps=info.fps, fps=fps, profile=profile,
    )
    manifest.mark("ingest", "done", samples=len(samples))
    manifest.save(game_dir(manifest.game_id))
    return manifest
