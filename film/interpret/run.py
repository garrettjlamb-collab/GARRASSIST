from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.arrivals import Arrival, find_ball_arrivals, find_motion_arrivals, merge_arrivals
from film.interpret.events import Event, build_shot_events, load_events, mark_dead_time, save_events, scoreboard_only_events
from film.interpret.phases import game_phases
from film.interpret.quarters import load_quarters, mark_quarters
from film.perceive.run import MOTION_COLUMNS
from film.perceive.scoreboard import load_timeline
from film.perceive.tracks import read_tracks
from film.review.decisions import carry_over_decisions
from film.interpret.ballpath import hoop_offset, path_features, read_ball_path
from film.train.montage import rim_follower
from film.train.rimclf import RimClassifier


PATHS_FILE = "ballpaths.json"


def ball_paths(gd: Path, m: Manifest, arrivals: list[Arrival], rims: pd.DataFrame, detector) -> list[dict]:
    """The ball's path after each arrival. Kept in the game folder so only new arrivals are read from the
    film again; a different ball detector starts the file over."""
    file = gd / PATHS_FILE
    stored = json.loads(file.read_text()) if file.exists() else {}
    known = stored.get("paths", {}) if stored.get("detector") == detector.name else {}
    paths = {}
    for a in arrivals:
        key = f"{a.frame}-r{a.rim_id}"
        paths[key] = known.get(key) or read_ball_path(Path(m.source), a.t, a.rim_box, m.width, m.height, detector,
                                                      rim_at=rim_follower(rims, a.rim_id, m.fps))
    file.write_text(json.dumps({"detector": detector.name, "paths": paths}))
    return [paths[f"{a.frame}-r{a.rim_id}"] for a in arrivals]


def classify_arrivals(gd: Path, m: Manifest, arrivals: list[Arrival], rims: pd.DataFrame, clf, detector) -> list[tuple[str, float]]:
    paths = ball_paths(gd, m, arrivals, rims, detector)
    rim_ids = sorted({a.rim_id for a in arrivals})
    offsets = {r: hoop_offset([p for a, p in zip(arrivals, paths) if a.rim_id == r]) for r in rim_ids}
    return [clf.predict(path_features(p, offsets[a.rim_id]))[:2] for a, p in zip(arrivals, paths)]


def interpret(game_id: str, classifier=None, use_classifier: bool = True, device: str | None = None,
              ball_detector=None, log=print) -> list[Event]:
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
    preds: list[tuple[str, float] | None] = [None] * len(arrivals)
    if clf is not None and arrivals:
        if ball_detector is None:
            from film.perceive.detectors import load_detector

            ball_detector = load_detector("ball", pick_device(device))
        preds = list(classify_arrivals(gd, m, arrivals, rims, clf, ball_detector))
    events = build_shot_events(arrivals, timeline, preds, game_id)
    sb_only = scoreboard_only_events(timeline, events)
    events = sorted(events + sb_only, key=lambda e: (e.t, e.rim_id))
    warmups = mark_dead_time(events, game_phases(timeline))
    quarters = load_quarters(gd)
    if quarters:                                          # the coach's marks win over the scoreboard
        warmups = mark_quarters(events, quarters)
    previous = load_events(gd)
    save_events(gd, events)
    carried = carry_over_decisions(gd, previous, events)       # the coach's review follows each shot to its new card
    version = clf.version if clf is not None else "none"
    m.model_versions["rimclf"] = version
    m.mark("interpret", "done", arrivals=len(arrivals), events=len(events),
           shots=sum(e.type == "shot" for e in events), scoreboard_only=len(sb_only), warmups=warmups, decisions=carried, classifier=version, scoreboard_reads=len(timeline))
    m.save(gd)
    log(f"interpret {game_id}: {len(arrivals)} arrivals -> {len(events)} events ({m.stages['interpret']['shots']} shots, "
        f"{len(sb_only)} scoreboard-only, {warmups} in warmup time), classifier={version}")
    if any(carried.values()):
        log(f"review decisions: {carried['kept']} kept, {carried['moved']} moved to re-detected shots, "
            f"{carried['orphaned']} no longer match a card (kept in review.json)")
    return events
