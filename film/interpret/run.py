from __future__ import annotations

from pathlib import Path

import pandas as pd

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.arrivals import find_ball_arrivals, find_motion_arrivals, merge_arrivals
from film.interpret.events import Event, build_shot_events, save_events, scoreboard_only_events
from film.perceive.run import MOTION_COLUMNS
from film.perceive.scoreboard import load_timeline
from film.perceive.tracks import read_tracks
from film.train.montage import build_montage
from film.train.rimclf import RimClassifier


def interpret(game_id: str, classifier=None, use_classifier: bool = True, device: str | None = None, log=print) -> list[Event]:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    ball, rims = read_tracks(gd, "ball"), read_tracks(gd, "rim")
    motion_path = gd / "motion.parquet"
    motion = pd.read_parquet(motion_path) if motion_path.exists() else pd.DataFrame(columns=MOTION_COLUMNS)
    arrivals = merge_arrivals(find_ball_arrivals(ball, rims, m.fps), find_motion_arrivals(motion, rims, m.fps))
    timeline = load_timeline(gd / "scoreboard.jsonl")
    clf = classifier
    if clf is None and use_classifier:
        clf = RimClassifier.load(pick_device(device))
    preds: list[tuple[str, float] | None] = []
    for arrival in arrivals:
        if clf is None:
            preds.append(None)
            continue
        montage = build_montage(Path(m.source), arrival.t, arrival.rim_box, m.width, m.height)
        label, prob, _ = clf.predict(montage)
        preds.append((label, prob))
    events = build_shot_events(arrivals, timeline, preds, game_id)
    sb_only = scoreboard_only_events(timeline, events)
    events = sorted(events + sb_only, key=lambda e: (e.t, e.rim_id))
    save_events(gd, events)
    version = clf.version if clf is not None else "none"
    m.model_versions["rimclf"] = version
    m.mark("interpret", "done", arrivals=len(arrivals), events=len(events),
           shots=sum(e.type == "shot" for e in events), scoreboard_only=len(sb_only), classifier=version, scoreboard_reads=len(timeline))
    m.save(gd)
    log(f"interpret {game_id}: {len(arrivals)} arrivals -> {len(events)} events ({m.stages['interpret']['shots']} shots, "
        f"{len(sb_only)} scoreboard-only), classifier={version}")
    return events
