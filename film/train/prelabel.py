from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.video import iter_frames
from film.interpret.ballpath import path_crop_box
from film.interpret.events import load_events
from film.perceive.ball import detector_tile, rim_crop_box
from film.perceive.detectors import load_detector
from film.perceive.rims import rim_boxes
from film.perceive.tracks import read_tracks
from film.review.decisions import load_decisions
from film.train.datasets import CLASS_NAMES, is_known, write_pending
from film.train.montage import rim_follower

ARRIVAL_OFFSETS_S = (-0.6, -0.4, -0.2, 0.0, 0.2)
# Around a shot the coach confirmed the ball is in the rim crop most of the time: sample it densely.
CONFIRMED_OFFSETS_S = tuple(round(-1.4 + 0.2 * i, 1) for i in range(13))         # -1.4 ... +1.0, on the 0.1 s grid of the image ids
SCOREBOARD_OFFSETS_S = tuple(round(-3.0 + 0.5 * i, 2) for i in range(13))        # the time is only approximate
# The made-or-missed classifier reads the ball's path in the second after it reaches the rim, and that is
# where the detector loses it most often (in the net, blurred, against the backboard).
AFTER_SHOT_WINDOW_S = 1.0
AFTER_SHOT_SURE_CONF = 0.4
AFTER_SHOT_PER_MADE, AFTER_SHOT_PER_MISSED = 4, 2


def spread(items: list, n: int) -> list:
    """At most n items, taken evenly across the whole list rather than from its start."""
    if len(items) <= n:
        return list(items)
    picks = sorted({int(round(i)) for i in np.linspace(0, len(items) - 1, n)})
    return [items[i] for i in picks]


def ball_sample_times(game_dir_path: Path) -> list[float]:
    """Moments worth a ball label: around confirmed shots once the game has been reviewed, else around every live event."""
    events = load_events(game_dir_path)
    decisions = load_decisions(game_dir_path)
    confirmed = [e for e in events if decisions.get(e.id, {}).get("status") == "confirmed"]
    if confirmed:
        return sorted({round(e.t + d, 2) for e in confirmed
                       for d in (SCOREBOARD_OFFSETS_S if e.signals.get("source") == "scoreboard" else CONFIRMED_OFFSETS_S)})
    return sorted({round(e.t + d, 2) for e in events if "warmup" not in e.flags for d in ARRIVAL_OFFSETS_S})


def after_shot_samples(game_dir_path: Path, fps: float) -> list[tuple[float, int]]:
    """(time, rim id) for moments in the second after a confirmed shot where the detector had no confident
    ball, from the paths interpret stored. More from makes: a make is decided by the ball in the net."""
    file = Path(game_dir_path) / "ballpaths.json"
    paths = json.loads(file.read_text()).get("paths", {}) if file.exists() else {}
    decisions = load_decisions(game_dir_path)
    out = []
    for e in load_events(game_dir_path):
        decision = decisions.get(e.id, {})
        path = paths.get(f"{int(round(e.t * fps))}-r{e.rim_id}")
        if decision.get("status") != "confirmed" or path is None:
            continue
        lost = [f["dt"] for f in path["frames"] if 0 < f["dt"] <= AFTER_SHOT_WINDOW_S + 1e-6
                and max((b[2] for b in f["balls"]), default=0.0) < AFTER_SHOT_SURE_CONF]
        made = decision.get("made", e.made)
        out += [(round(e.t + dt, 2), e.rim_id) for dt in spread(lost, AFTER_SHOT_PER_MADE if made else AFTER_SHOT_PER_MISSED)]
    return sorted(set(out))


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
    # No open-vocabulary fallback here: on film it was not trained for it boxes the ball or nothing, and a
    # wrong proposal costs the coach more than an empty frame.
    det = detector or load_detector("rim", pick_device(device), profile=m.profile, fallback=False)
    count = 0
    for t in sample_times(m.duration_s, n, margin_s):
        frame = first_frame_at(Path(m.source), t, m.fps)
        if frame is None:
            continue
        boxes = [b for b in det.detect(frame) if b.cls in CLASS_NAMES["rim"]] if det is not None else []
        write_pending("rim", f"{game_id}_{int(round(t * 10)):06d}", frame, boxes, extra={"t": t})
        count += 1
    log(f"prelabel rim: {count} frames queued for {game_id}")
    return count


def prelabel_balls(game_id: str, n: int = 300, rim_detector=None, ball_detector=None, device: str | None = None,
                   margin_s: float = 30.0, after_shots: bool = False, propose: bool = True, log=print) -> int:
    """Propose ball boxes inside rim crops, sampled around known arrivals when events exist.

    With `after_shots`, the crops are instead the ones the made-or-missed classifier's path reader sees, at
    the moments after confirmed shots where the detector lost the ball. With `propose=False` the crops are
    queued bare, for a gym where the detector's guesses are mostly wrong."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    device = pick_device(device)
    if after_shots:
        rims = read_tracks(gd, "rim")
        followers = {}
        count = 0
        for t, rim_id in spread(after_shot_samples(gd, m.fps), n):
            image_id = f"{game_id}_{int(round(t * 100)):07d}_p{rim_id}"
            if is_known("ball", image_id):
                continue
            rim = followers.setdefault(rim_id, rim_follower(rims, rim_id, m.fps))(t)
            frame = first_frame_at(Path(m.source), t, m.fps) if rim is not None else None
            if frame is None:
                continue
            tile, _ = detector_tile(frame, path_crop_box(rim, m.width, m.height))
            # No proposed boxes: the detector had no confident ball here, and its weak guesses (a water cooler,
            # a wall pad) are mostly wrong. Saved with no box, the crop teaches it exactly that.
            write_pending("ball", image_id, tile, [], extra={"t": t})
            count += 1
        log(f"prelabel ball: {count} after-shot crops queued for {game_id}")
        return count
    rim_det = rim_detector or load_detector("rim", device, profile=m.profile)
    ball_det = (ball_detector or load_detector("ball", device)) if propose else None
    times = ball_sample_times(gd) or sample_times(m.duration_s, n, margin_s)
    times = spread([t for t in times if 0.0 <= t < m.duration_s], n)
    count = 0
    for t in times:
        frame = first_frame_at(Path(m.source), t, m.fps)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        rims = sorted(rim_boxes(rim_det.detect(frame)), key=lambda b: -b.conf)[:2]
        for i, rim in enumerate(rims):
            tile, _ = detector_tile(frame, rim_crop_box(rim, w, h))
            boxes = [b for b in ball_det.detect(tile) if b.cls == "ball"] if propose else []
            image_id = f"{game_id}_{int(round(t * 10)):06d}_r{i}"
            if is_known("ball", image_id):                    # already queued, labeled or skipped on an earlier run
                continue
            write_pending("ball", image_id, tile, boxes, extra={"t": t})
            count += 1
    log(f"prelabel ball: {count} crops queued for {game_id}")
    return count
