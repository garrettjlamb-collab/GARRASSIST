from __future__ import annotations

from pathlib import Path

import numpy as np

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.video import iter_frames
from film.interpret.events import load_events
from film.perceive.ball import crop, rim_crop_box
from film.perceive.detectors import load_detector
from film.train.datasets import CLASS_NAMES, write_pending

ARRIVAL_OFFSETS_S = (-0.6, -0.4, -0.2, 0.0, 0.2)


def sample_times(duration_s: float, n: int, margin_s: float = 30.0) -> list[float]:
    if n <= 0 or duration_s <= 2 * margin_s:
        return []
    return [float(x) for x in np.linspace(margin_s, duration_s - margin_s, n)]


def first_frame_at(source: Path, t: float, fps: float) -> np.ndarray | None:
    for _, frame in iter_frames(source, fps=fps, start=t, end=t + 1.0):
        return frame
    return None


def prelabel_rims(game_id: str, n: int = 150, detector=None, device: str | None = None,
                  margin_s: float = 30.0, log=print) -> int:
    """Propose rim and backboard boxes on evenly spaced frames for the coach to confirm."""
    m = Manifest.load(game_dir(game_id))
    det = detector or load_detector("rim", pick_device(device), profile=m.profile)
    count = 0
    for t in sample_times(m.duration_s, n, margin_s):
        frame = first_frame_at(Path(m.source), t, m.fps)
        if frame is None:
            continue
        boxes = [b for b in det.detect(frame) if b.cls in CLASS_NAMES["rim"]]
        write_pending("rim", f"{game_id}_{int(round(t * 10)):06d}", frame, boxes, extra={"t": t})
        count += 1
    log(f"prelabel rim: {count} frames queued for {game_id}")
    return count


def prelabel_balls(game_id: str, n: int = 300, rim_detector=None, ball_detector=None, device: str | None = None,
                   margin_s: float = 30.0, log=print) -> int:
    """Propose ball boxes inside rim crops, sampled around known arrivals when events exist."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    device = pick_device(device)
    rim_det = rim_detector or load_detector("rim", device, profile=m.profile)
    ball_det = ball_detector or load_detector("ball", device)
    times = [e.t + d for e in load_events(gd) for d in ARRIVAL_OFFSETS_S]
    if not times:
        times = sample_times(m.duration_s, n, margin_s)
    times = [t for t in times if 0.0 <= t < m.duration_s][:n]
    count = 0
    for t in times:
        frame = first_frame_at(Path(m.source), t, m.fps)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        rims = sorted((b for b in rim_det.detect(frame) if b.cls == "rim"), key=lambda b: -b.conf)[:2]
        for i, rim in enumerate(rims):
            tile = crop(frame, rim_crop_box(rim, w, h))
            boxes = [b for b in ball_det.detect(tile) if b.cls == "ball"]
            write_pending("ball", f"{game_id}_{int(round(t * 10)):06d}_r{i}", tile, boxes, extra={"t": t})
            count += 1
    log(f"prelabel ball: {count} crops queued for {game_id}")
    return count
