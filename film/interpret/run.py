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
from film.perceive.rims import basket_sides
from film.perceive.run import MOTION_COLUMNS
from film.perceive.scoreboard import load_timeline
from film.perceive.tracks import read_tracks
from film.review.decisions import carry_over_decisions
from film.interpret.ballpath import FEATURES, hoop_offset, path_features, read_ball_path
from film.interpret.fuse import fuse
from film.interpret.rimview import read_rim_view
from film.train.montage import rim_follower
from film.train.rimclf import CLASSES, RimClassifier


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


VIEWS_FILE = "picviews.json"


def picture_scores(gd: Path, m: Manifest, arrivals: list[Arrival], rims: pd.DataFrame, hoop_dx: list[float], pictures) -> list[dict]:
    """The picture classifier's class probabilities for each arrival. Kept in the game folder, so only new
    arrivals are cut from the film again; a different picture classifier starts the file over."""
    file = gd / VIEWS_FILE
    stored = json.loads(file.read_text()) if file.exists() else {}
    known = stored.get("scores", {}) if stored.get("model") == pictures.version else {}
    keys = [f"{a.frame}-r{a.rim_id}" for a in arrivals]
    todo = [i for i, k in enumerate(keys) if k not in known]
    for start in range(0, len(todo), 32):                          # a view is 600 kB: a few at a time
        chunk = todo[start:start + 32]
        views = [read_rim_view(Path(m.source), arrivals[i].t, arrivals[i].rim_box, hoop_dx=hoop_dx[i],
                               rim_at=rim_follower(rims, arrivals[i].rim_id, m.fps)) for i in chunk]
        for i, row in zip(chunk, pictures.scores(views)):
            known[keys[i]] = [round(float(v), 4) for v in row]
    scores = {k: known[k] for k in keys}
    file.write_text(json.dumps({"model": pictures.version, "scores": scores}))
    return [dict(zip(CLASSES, scores[k])) for k in keys]


def classify_arrivals(gd: Path, m: Manifest, arrivals: list[Arrival], rims: pd.DataFrame, clf, detector,
                      pictures=None) -> list[dict]:
    """One opinion per arrival (see fuse): the ball-path classifier's call, joined by the picture classifier's
    when there is one."""
    paths = ball_paths(gd, m, arrivals, rims, detector)
    # The hoop's offset from the rim box is a property of the basket, not of one stretch of tracking.
    sides = basket_sides(rims, m.width)
    side_of = [sides.get(a.rim_id, a.rim_id) for a in arrivals]
    offsets = {s: hoop_offset([p for k, p in zip(side_of, paths) if k == s]) for s in set(side_of)}
    path_scores = [clf.predict(path_features(p, offsets[s]))[2] for s, p in zip(side_of, paths)]
    seen = picture_scores(gd, m, arrivals, rims, [offsets[s] for s in side_of], pictures) if pictures is not None else [None] * len(arrivals)
    return [fuse(a, b) for a, b in zip(path_scores, seen)]


def interpret(game_id: str, classifier=None, use_classifier: bool = True, device: str | None = None,
              ball_detector=None, picture_classifier=None, log=print) -> list[Event]:
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
    pictures = picture_classifier
    if pictures is None and clf is not None and classifier is None:
        from film.train.picclf import PictureClassifier

        pictures = PictureClassifier.load(pick_device(device))
    preds: list[tuple[str, float] | None] = [None] * len(arrivals)
    opinions: list[dict | None] = [None] * len(arrivals)
    if clf is not None and arrivals:
        if ball_detector is None:
            from film.perceive.detectors import load_detector

            ball_detector = load_detector("ball", pick_device(device))
        opinions = classify_arrivals(gd, m, arrivals, rims, clf, ball_detector, pictures)
        preds = [(o["label"], o["prob"]) for o in opinions]
    events = build_shot_events(arrivals, timeline, preds, game_id, opinions=opinions)
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
    m.model_versions["picclf"] = pictures.version if clf is not None and pictures is not None else "none"
    m.mark("interpret", "done", arrivals=len(arrivals), events=len(events),
           shots=sum(e.type == "shot" for e in events), scoreboard_only=len(sb_only), warmups=warmups, decisions=carried, classifier=version, scoreboard_reads=len(timeline))
    m.save(gd)
    log(f"interpret {game_id}: {len(arrivals)} arrivals -> {len(events)} events ({m.stages['interpret']['shots']} shots, "
        f"{len(sb_only)} scoreboard-only, {warmups} in warmup time), classifier={version}, pictures={m.model_versions['picclf']}")
    if any(carried.values()):
        log(f"review decisions: {carried['kept']} kept, {carried['moved']} moved to re-detected shots, "
            f"{carried['orphaned']} no longer match a card (kept in review.json)")
    return events
