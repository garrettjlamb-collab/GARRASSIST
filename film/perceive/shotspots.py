"""What the person and ball detectors see before each confirmed shot reaches the rim, kept in games/<id>/shot_looks.json.gz
so the location rules (film/interpret/location.py) can be run again without running the models. Takes about 3 s a shot."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.registry import base_weights_path, latest
from film.common.video import iter_frames
from film.interpret.events import load_events, save_events
from film.interpret import location
from film.interpret.location import Look, shot_spot
from film.interpret.zones import is_three, shot_type_of, zone_of
from film.perceive.court_run import load_mappings, mapping_at
from film.perceive.detectors import YoloDetector
from film.perceive.tracks import read_tracks
from film.review.decisions import load_decisions

LOOKS = "shot_looks.json.gz"
BEFORE_S, AFTER_S, FPS = 3.0, 0.3, 10.0
PERSON_IMGSZ, PERSON_CONF, BALL_CONF = 960, 0.2, 0.1
NO_RIM = 0                                  # a scoreboard-only card has no rim of its own: any rim will do


def confirmed_shots(gd: Path) -> list:
    decisions = load_decisions(gd)
    return [e for e in load_events(gd) if (decisions.get(e.id) or {}).get("status") == "confirmed" and e.signals.get("phase") != "warmup"]


def collect(game_id: str, people=None, ball=None, device: str = "mps", log=print) -> Path:
    """Run the detectors over the seconds before each confirmed shot. A shot already in the file is not run again."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    if people is None:
        people = YoloDetector.from_weights(base_weights_path("yolo11n.pt"), device, PERSON_IMGSZ, conf=PERSON_CONF, name="person")
    if ball is None:
        version, folder = latest("ball")
        ball = YoloDetector.from_weights(folder / "best.pt", device, m.width, conf=BALL_CONF, name=f"ball@{version}")
    path = gd / LOOKS
    done = {r["id"]: r for r in read_looks(gd)}
    rims = read_tracks(gd, "rim")
    shots = confirmed_shots(gd)
    start = time.time()
    for i, e in enumerate(shots):
        if e.id in done:
            continue
        frames = []
        for t, frame in iter_frames(Path(m.source), FPS, start=max(0.0, e.t - BEFORE_S), end=e.t + AFTER_S):
            frames.append({"t": t, "people": [[b.x1, b.y1, b.x2, b.y2, b.conf] for b in people.detect(frame) if b.cls == "person"],
                           "ball": [[b.x1, b.y1, b.x2, b.y2, b.conf] for b in ball.detect(frame) if b.cls == "ball"]})
        near = rims[(rims.t >= e.t - BEFORE_S) & (rims.t <= e.t + AFTER_S)] if len(rims) else rims
        done[e.id] = {"id": e.id, "t": e.t, "rim_id": e.rim_id, "frames": frames,
                      "rims": [[float(r.t), int(r.track_id), float(r.x1), float(r.y1), float(r.x2), float(r.y2)] for r in near.itertuples()]}
        if i % 25 == 24:
            _write(path, done)
            log(f"  {game_id}: {i + 1} of {len(shots)} shots, {time.time() - start:.0f} s")
    _write(path, done)
    return path


def _write(path: Path, rows: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(list(rows.values()), f)
    tmp.replace(path)


def read_looks(gd: Path) -> list[dict]:
    path = Path(gd) / LOOKS
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _rims_in_anchor(row: dict, looks: list[Look]) -> list[tuple[float, float]]:
    """The rims seen in the window, as the centres of their boxes carried into the last mapped frame's pixels. A card that
    came from a rim keeps to that rim; a scoreboard-only card takes any."""
    anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
    if anchor is None:
        return []
    inv, out = np.linalg.inv(anchor.H), []
    for t, rim_id, x1, y1, x2, y2 in row["rims"]:
        if row["rim_id"] >= NO_RIM and rim_id != row["rim_id"]:
            continue
        k = min(looks, key=lambda q: abs(q.t - t))
        if k.mapping is None:
            continue
        v = inv @ (k.mapping.H @ np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.0]))
        out.append((float(v[0] / v[2]), float(v[1] / v[2])))
    return out


def place(game_id: str, log=print) -> dict:
    """Run the location rules over the kept looks and write the spots onto the game's events."""
    gd = game_dir(game_id)
    scale = Manifest.load(gd).width / location.REFERENCE_WIDTH
    maps = load_mappings(gd)
    rows = {r["id"]: r for r in read_looks(gd)}
    events = load_events(gd)
    placed, methods, truth = 0, {}, {"sides": 0, "right": 0}
    for e in events:
        row = rows.get(e.id)
        if row is None:
            continue
        looks = [Look(f["t"], mapping_at(maps, f["t"]), [tuple(b) for b in f["people"]], [tuple(b) for b in f["ball"]]) for f in row["frames"]]
        points = e.signals.get("points") if e.made else None
        spot = shot_spot(looks, _rims_in_anchor(row, looks), e.t if row["rim_id"] >= NO_RIM else None, e.t, points, scale=scale)
        methods[spot["method"]] = methods.get(spot["method"], 0) + 1
        e.signals["location"] = {k: spot[k] for k in ("method", "release_t", "flight_points", "reach_px", "box")}
        e.flags = [f for f in e.flags if f not in ("no-flight", "no-shooter", "shooter-far-from-ball")] + spot["flags"]
        if spot["x"] is None:
            e.x = e.y = e.zone = e.shot_type = e.location_conf = None
            e.needs_location = True
            continue
        placed += 1
        e.x, e.y, e.location_conf = spot["x"], spot["y"], spot["conf"]
        e.zone = zone_of(e.x, e.y)
        e.shot_type = spot["shot_type"] or shot_type_of(e.x, e.y)
        e.needs_location = False
        if spot["method"] == "flight" and points in (2, 3):
            truth["sides"] += 1
            truth["right"] += is_three(e.x, e.y) == (points == 3)
    save_events(gd, events)
    result = {"shots": len(rows), "placed": placed, "methods": methods, "made_twos_threes": truth["sides"], "arc_side_right": truth["right"]}
    log(f"film locate {game_id}: {placed} of {len(rows)} shots placed {methods}; "
        + (f"arc side right on {truth['right']} of {truth['sides']} made twos and threes" if truth["sides"] else "no scoreboard to check the arc against"))
    return result
