from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from film.common.cloud import needs_film
from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.video import iter_frames
from film.perceive.ball import detect_ball_near_rim
from film.perceive.detectors import Box, load_detector
from film.perceive.motion import MotionEnergy
from film.perceive.rims import TRACKED_FIRST_ID, RimTracker, detect_rims, smooth_rims
from film.perceive.scoreboard import clean_timeline, has_scoreboard, load_profile, read_scoreboard, save_timeline
from film.perceive.tracks import replace_kind, rows_to_frame

MOTION_COLUMNS = ["frame", "t", "rim_id", "energy", "energy_global"]
LOW_RIM_FRACTION = 0.15


@needs_film
def perceive(game_id: str, device: str | None = None, rim_detector=None, ball_detector=None,
             limit_s: float | None = None, log=print) -> dict:
    """One decode pass: rims, the ball near each rim, motion energy, and the scoreboard once a second."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    device = pick_device(device)
    rim_detector = rim_detector or load_detector("rim", device, profile=m.profile)
    ball_detector = ball_detector or load_detector("ball", device)
    profile = load_profile(m.profile) if has_scoreboard(m.profile) else None
    tracker, motion = RimTracker(first_id=TRACKED_FIRST_ID), MotionEnergy()
    rim_rows, ball_rows, motion_rows, reads = [], [], [], []
    last_second, n_frames, started = -1, 0, time.time()
    report_every = max(1, int(60 * m.fps))
    log(f"perceive {game_id}: rims={rim_detector.name} ball={ball_detector.name} device={device} scoreboard={'yes' if profile else 'no'}")
    for t, frame in iter_frames(Path(m.source), fps=m.fps, end=limit_s):
        fi = int(round(t * m.fps))
        n_frames += 1
        rims = detect_rims(frame, rim_detector, tracker, fi, t, rim_detector.name)
        rim_rows.extend(rims)
        pairs = [(r["track_id"], Box(r["x1"], r["y1"], r["x2"], r["y2"], r["conf"], "rim")) for r in rims]
        ball_rows.extend(detect_ball_near_rim(frame, pairs, ball_detector, fi, t, ball_detector.name))
        motion_rows.extend(motion.update(frame, pairs, fi, t))
        if profile is not None and int(t) != last_second:
            last_second = int(t)
            reads.append(read_scoreboard(frame, profile, t=t))
        if n_frames % report_every == 0:
            elapsed = time.time() - started
            log(f"  t={t:7.1f}s  {elapsed:6.0f}s elapsed  {n_frames / elapsed:5.1f} frames/s")
    replace_kind(gd, smooth_rims(rows_to_frame(rim_rows)), "rim")
    replace_kind(gd, rows_to_frame(ball_rows), "ball")
    pd.DataFrame(motion_rows, columns=MOTION_COLUMNS).to_parquet(gd / "motion.parquet", index=False)
    n_scoreboard = 0
    if profile is not None:
        kept = clean_timeline(reads)
        save_timeline(kept, gd / "scoreboard.jsonl")
        n_scoreboard = len(kept)
    m.model_versions.update({"rim": rim_detector.name, "ball": ball_detector.name})
    m.mark("perceive", "done", frames=n_frames, rims=len(rim_rows), balls=len(ball_rows),
           scoreboard=n_scoreboard, seconds=round(time.time() - started, 1), device=device, limit_s=limit_s)
    m.save(gd)
    log(f"done: {n_frames} frames, {len(rim_rows)} rim rows, {len(ball_rows)} ball rows, {n_scoreboard} scoreboard reads in {m.stages['perceive']['seconds']}s")
    rim_frames = len({r["frame"] for r in rim_rows})
    fraction = rim_frames / n_frames if n_frames else 1.0
    if fraction < LOW_RIM_FRACTION:
        log(f"WARNING: rims found in only {fraction:.0%} of frames ({rim_frames} of {n_frames} frames) with "
            f"{rim_detector.name}. The rim templates may not match this gym or camera; few rims means few shots. "
            f"Check the camera profile ({m.profile}) before trusting the events.")
    return m.stages["perceive"]
