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
from film.interpret.camera import from_homography
from film.interpret.court import HALF_DEPTH, WIDTH, fold
from film.interpret.flight3d import fit_path, rims_of
from film.interpret.location import Look, ball_sightings, shot_spot
from film.interpret.zones import is_three, shot_type_of, zone_of
from film.perceive.court_run import load_mappings, mapping_at, mapping_between
from film.perceive.detectors import YoloDetector
from film.perceive.tracks import read_tracks
from film.review.decisions import load_decisions

LOOKS = "shot_looks.json.gz"
BEFORE_S, AFTER_S, FPS = 3.0, 0.3, 10.0
PERSON_IMGSZ, PERSON_CONF, BALL_CONF = 960, 0.2, 0.1
MAX_SET_ASIDE = 3
OFFICIAL_THRESHOLD = 0.5
SET_ASIDE_CONF = 0.8                        # the first choice looked wrong, so the second is a little less sure
PATH_RELEASE_Z = 7.0                       # held out and tuned games both did better at 7 ft than 8 or 9
PATH_RADIUS_FT = 3.0
PATH_AGREE_FT = 5.0
PATH_DIFFERS_CONF, PATH_DISAGREES_CONF = 0.8, 0.6
PATH_BEFORE_S, PATH_AFTER_S = 0.6, 0.4
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


def _person_key(t: float, box) -> str:
    return f"{t:.2f}:" + ",".join(f"{v:.0f}" for v in box[:4])


def official_score(row: dict, t: float, box, source: Path | None, classifier) -> float | None:
    """How much this person looks like a non-player, kept on the shot so the rules can run again without the film. None when
    it cannot be read: no model, or no film at hand."""
    kept = row.setdefault("officials", {})
    key = _person_key(t, box)
    if key in kept:
        return kept[key]
    if classifier is None or source is None:
        return None
    from film.train.prelabel import first_frame_at

    frame = first_frame_at(source, t, FPS)
    if frame is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return None
    kept[key] = round(float(classifier.scores([crop])[0]), 3)
    return kept[key]


def _from_path(spot: dict, looks: list[Look], row: dict, arrival_t, t: float, scale: float, size: tuple[int, int], source, classifier) -> dict:
    """The ball's path in 3D (film/interpret/flight3d.py), used to check the rules' choice. A player within PATH_RADIUS_FT of where
    the path starts takes the dot; with no one there the rules' choice stands, marked when it is far from the path."""
    anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
    cam = from_homography(anchor.H, *size) if anchor is not None else None
    if cam is None:
        return spot
    lo, hi = (t - PATH_BEFORE_S, t + PATH_AFTER_S) if arrival_t is not None else (looks[0].t + PATH_AFTER_S, looks[-1].t)
    path = fit_path(cam, ball_sightings(looks, scale), rims_of(anchor.length), lo, hi, anchor.length, scale, release_z=PATH_RELEASE_Z)
    if path is None:
        return spot
    at = min(looks, key=lambda k: abs(k.t - path.release_t))
    near = []
    if at.mapping is not None:
        for b in at.people:
            feet = at.mapping.to_court((b[0] + b[2]) / 2, b[3])
            if feet is not None and at.mapping.contains((b[0] + b[2]) / 2, b[3], margin=location.COURT_MARGIN_FT):
                near.append((float(np.hypot(feet[0] - path.spot[0], feet[1] - path.spot[1])), b, feet))
    x, y = fold(*path.spot, anchor.length)
    apart = None if spot["x"] is None else float(np.hypot(spot["x"] - x, spot["y"] - y))
    for gap, box, feet in sorted(near, key=lambda n: n[0]):
        if gap > PATH_RADIUS_FT:
            break
        score = official_score(row, at.t, box, source, classifier)
        if score is not None and score >= OFFICIAL_THRESHOLD:
            continue
        fx, fy = fold(*feet, at.mapping.length)
        differs = apart is not None and apart > PATH_AGREE_FT
        conf = min(1.0, path.points / location.FULL_CONF_POINTS) * at.mapping.conf * (PATH_DIFFERS_CONF if differs else 1.0)
        return {**spot, "x": round(float(min(max(fx, 0.0), WIDTH)), 1), "y": round(float(min(max(fy, 0.0), HALF_DEPTH)), 1), "conf": round(float(conf), 2), "method": "path",
                "flags": ["rule-differs"] if differs else [], "release_t": round(float(at.t), 2), "flight_points": path.points, "reach_px": None,
                "box": tuple(round(float(v), 1) for v in box[:4]), "path_gap_ft": round(gap, 1)}
    if apart is not None and apart > PATH_AGREE_FT:
        return {**spot, "conf": round(spot["conf"] * PATH_DISAGREES_CONF, 2), "flags": [*spot["flags"], "path-disagrees"]}
    return spot


def spot_of_a_player(looks: list[Look], row: dict, arrival_t, t: float, points, scale: float, source: Path | None, classifier,
                     size: tuple[int, int] | None = None) -> dict:
    """The rules' spot, setting aside anyone chosen as shooter who looks like a non-player and choosing again, then checked
    against the ball's path when the picture's size is given."""
    rims = _rims_in_anchor(row, looks)
    original = looks
    aside = 0
    for _ in range(MAX_SET_ASIDE + 1):
        spot = shot_spot(looks, rims, arrival_t, t, points, scale=scale)
        if spot["method"] != "flight" or not spot["box"]:
            break
        at = min(looks, key=lambda k: abs(k.t - spot["release_t"]))
        person = max(at.people, key=lambda b: location._iou(b, spot["box"]), default=spot["box"])
        score = official_score(row, at.t, person, source, classifier)
        if score is None or score < OFFICIAL_THRESHOLD:
            break
        looks = location.set_aside(looks, at.t, person)
        aside += 1
    else:
        spot = {**spot, "x": None, "y": None, "conf": None, "method": "none", "flags": ["no-player-found"], "shot_type": None, "release_t": None, "box": None}
    if aside and spot["x"] is not None:
        spot["flags"] = [*spot["flags"], "official-set-aside"]
        spot["conf"] = round(spot["conf"] * SET_ASIDE_CONF, 2)
    spot["set_aside"] = aside
    if size is not None and spot["method"] != "free-throw":
        spot = _from_path(spot, original, row, arrival_t, t, scale, size, source, classifier)
    return spot


def place(game_id: str, source: Path | None = None, classifier=None, log=print) -> dict:
    """Run the location rules over the kept looks and write the spots onto the game's events. With the film and a model,
    people chosen as shooter are checked and non-players set aside; what was read is kept in the looks file."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    scale = m.width / location.REFERENCE_WIDTH
    maps = load_mappings(gd)
    rows = {r["id"]: r for r in read_looks(gd)}
    events = load_events(gd)
    placed, set_aside, methods, truth = 0, 0, {}, {"sides": 0, "right": 0}
    for e in events:
        row = rows.get(e.id)
        if row is None:
            continue
        looks = [Look(f["t"], mapping_between(maps, f["t"]), [tuple(b) for b in f["people"]], [tuple(b) for b in f["ball"]]) for f in row["frames"]]
        points = e.signals.get("points") if e.made else None
        spot = spot_of_a_player(looks, row, e.t if row["rim_id"] >= NO_RIM else None, e.t, points, scale, source, classifier, size=(m.width, m.height))
        set_aside += spot["set_aside"]
        methods[spot["method"]] = methods.get(spot["method"], 0) + 1
        e.signals["location"] = {k: spot[k] for k in ("method", "release_t", "flight_points", "reach_px", "box")}
        e.flags = [f for f in e.flags if f not in ("no-flight", "no-shooter", "shooter-far-from-ball", "official-set-aside", "no-player-found", "rule-differs", "path-disagrees")] + spot["flags"]
        if spot["x"] is None:
            e.x = e.y = e.zone = e.shot_type = e.location_conf = None
            e.needs_location = True
            continue
        placed += 1
        e.x, e.y, e.location_conf = spot["x"], spot["y"], spot["conf"]
        e.zone = zone_of(e.x, e.y)
        e.shot_type = spot["shot_type"] or shot_type_of(e.x, e.y)
        e.needs_location = False
        if spot["method"] in ("flight", "path") and points in (2, 3):
            truth["sides"] += 1
            truth["right"] += is_three(e.x, e.y) == (points == 3)
    save_events(gd, events)
    _write(gd / LOOKS, rows)
    result = {"shots": len(rows), "placed": placed, "methods": methods, "set_aside": set_aside, "made_twos_threes": truth["sides"], "arc_side_right": truth["right"]}
    log(f"film locate {game_id}: {placed} of {len(rows)} shots placed {methods}; {set_aside} non-players set aside; "
        + (f"arc side right on {truth['right']} of {truth['sides']} made twos and threes" if truth["sides"] else "no scoreboard to check the arc against"))
    return result
