"""Experiment: choose the shooter by raised arms with the ball at the hands, around the start of the ball's flight. Scored on the
scoreboard (side of the arc) and free throws (feet from the line, rule off). The coach's dot-check answers are not read here."""
from __future__ import annotations

import glob
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from film.common.manifest import Manifest
from film.interpret import location as L
from film.interpret import shooter_pose as SP
from film.interpret.camera import from_homography
from film.interpret.court import FT_LINE_Y, WIDTH, fold
from film.interpret.flight3d import fit_path, rims_of
from film.interpret.location import GRAVITY_PX, Look, _carry, ball_sightings, find_flight
from film.interpret.zones import is_three
from film.perceive.court_run import load_mappings, mapping_between
from film.perceive.shotspots import NO_RIM, PATH_AFTER_S, PATH_BEFORE_S, PATH_RELEASE_Z, _rims_in_anchor, confirmed_shots, read_looks

HELD = ("2025-02-05-arcadia", "2025-02-05-jv-vs-marcos")
NOSE, LS, RS, LW, RW, LH, RH, LA, RA = 0, 5, 6, 9, 10, 11, 12, 15, 16
KP_CONF = 0.3
BEFORE, AFTER = 1.0, 0.3
MIN_SCORE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3


CASES = []
tot = defaultdict(list)
files = sorted(glob.glob("games/*/shot_poses.json.gz"))
for f in files:
    gd = Path(f).parent
    with gzip.open(f, "rt") as fh:
        poses = json.load(fh)
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    rows = {r["id"]: r for r in read_looks(gd)}
    scale = m.width / L.REFERENCE_WIDTH
    res = defaultdict(list)
    for e in confirmed_shots(gd):
        r = rows.get(e.id)
        if r is None or e.id not in poses:
            continue
        looks = [Look(x["t"], mapping_between(maps, x["t"]), [tuple(b) for b in x["people"]], [tuple(b) for b in x["ball"]]) for x in r["frames"]]
        anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
        if anchor is None:
            continue
        rims = _rims_in_anchor(r, looks)
        arrival = e.t if r["rim_id"] >= NO_RIM else None
        s = ball_sightings(looks, scale)
        flight = find_flight(s, rims, end_t=arrival, scale=scale)
        spots = {}
        rule = L.locate(looks, rims, arrival_t=arrival, scale=scale)
        if rule.x is not None:
            spots["rule"] = fold(rule.x, rule.y, anchor.length)
        pathspot = None
        cam = from_homography(anchor.H, m.width, m.height)
        if cam is not None:
            lo, hi = (e.t - PATH_BEFORE_S, e.t + PATH_AFTER_S) if arrival is not None else (looks[0].t + PATH_AFTER_S, looks[-1].t)
            path = fit_path(cam, s, rims_of(anchor.length), lo, hi, anchor.length, scale, release_z=PATH_RELEASE_Z)
            if path is not None:
                at = min(looks, key=lambda k: abs(k.t - path.release_t))
                if at.mapping is not None:
                    near = []
                    for b in at.people:
                        a = at.mapping.to_court((b[0] + b[2]) / 2, b[3])
                        if a is not None and L._on_court(b, at.mapping):
                            near.append((float(np.hypot(a[0] - path.spot[0], a[1] - path.spot[1])), a))
                    if near and min(near, key=lambda n: n[0])[0] <= 3.0:
                        pathspot = fold(*min(near, key=lambda n: n[0])[1], anchor.length)
        spots["path+rule"] = pathspot or spots.get("rule")
        pose = None
        if len(flight) >= L.FLIGHT_MIN_POINTS:
            mapping_fn = lambda t: mapping_between(maps, t)
            cam_xy = tuple(cam.centre[:2]) if cam is not None else None
            got = SP.picks(poses[e.id], flight, anchor, mapping_fn, MIN_SCORE, scale=scale, camera_xy=None)
            got2 = SP.picks(poses[e.id], flight, anchor, mapping_fn, MIN_SCORE, scale=scale, camera_xy=cam_xy)
            if got:
                pose = fold(*got[0].feet, anchor.length)
                res["pose used"].append(1)
            if got2:
                spots["pose-floor+path+rule"] = fold(*got2[0].feet, anchor.length)
            res["pose tried"].append(1)
        spots["pose+path+rule"] = pose or pathspot or spots.get("rule")
        spots.setdefault("pose-floor+path+rule", pathspot or spots.get("rule"))
        spots["path+pose+rule"] = pathspot or pose or spots.get("rule")
        pts = e.signals.get("points") if e.made else None
        if pose is not None and spots.get("path+rule") is not None:
            gap = float(np.hypot(pose[0] - spots["path+rule"][0], pose[1] - spots["path+rule"][1]))
            res["moved"].append(gap)
            if gap > 5:
                CASES.append({"game": gd.name, "id": e.id, "t": got[0].t, "pose_box": got[0].box, "score": got[0].score, "gap": round(gap, 1), "pose": pose, "other": spots["path+rule"],
                              "rule_box": rule.box, "rule_t": rule.release_t, "rel": round(got[0].t - e.t, 1), "pts": pts, "fps": m.fps, "source": m.source, "width": m.width})
        for name, sp in spots.items():
            if sp is None:
                continue
            res["dots " + name].append(1)
            if pts in (2, 3):
                res[f"{name} {pts}s"].append(is_three(*sp) == (pts == 3))
            if pts == 1:
                res[f"{name} ft"].append(float(np.hypot(sp[0] - WIDTH / 2, sp[1] - FT_LINE_Y)))
    key = gd.name[:42].replace("2026-10-09-new-marcos-de-niza-high-school-vs-", "")
    group = "held" if gd.name.startswith(HELD) else "tuned"
    for g in (group,):
        for k, v in res.items():
            tot[f"{g} {k}"] += v
keys = sorted({k.split(" ")[0] for k in tot})
for g in keys:
    print(g, "pose used", len(tot[f"{g} pose used"]), "of", len(tot[f"{g} pose tried"]), "min", MIN_SCORE, "moved>5ft", sum(v > 5 for v in tot[f"{g} moved"]))
    for name in ("rule", "path+rule", "pose+path+rule", "pose-floor+path+rule", "path+pose+rule"):
        a, b, ft = tot[f"{g} {name} 2s"], tot[f"{g} {name} 3s"], tot[f"{g} {name} ft"]
        print(f"  {name:22s} dots {len(tot[f'{g} dots {name}']):3d} | 2s {sum(a)}/{len(a)} 3s {sum(b)}/{len(b)} | FT n{len(ft)} median {np.median(ft) if ft else 0:.1f} ft, within 3 ft {np.mean(np.array(ft) <= 3) if ft else 0:.0%}")
if len(sys.argv) > 2:
    import cv2

    from film.common.cloud import game_film, local_source
    from film.train.prelabel import first_frame_at

    tiles = []
    for c in sorted(CASES, key=lambda c: -c["score"]):
        with local_source(game_film(c["game"]), log=lambda x: None):
            frame = first_frame_at(Path(c["source"]), c["t"], c["fps"])
        if frame is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in c["pose_box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if c["rule_box"]:
            a1, b1, a2, b2 = (int(v) for v in c["rule_box"])
            cv2.rectangle(frame, (a1, b1), (a2, b2), (0, 0, 255), 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        h = frame.shape[0]
        half = 300
        crop = frame[max(cy - half, 0):min(cy + half, h), max(cx - 533, 0):cx + 533]
        crop = cv2.resize(crop, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(crop, (0, 0), (640, 22), (0, 0, 0), -1)
        cv2.putText(crop, f"#{len(tiles)} {c['id'][5:]} s{c['score']} gap {c['gap']:.0f}ft rel{c['rel']} {c['pts'] if c['pts'] is not None else 'miss'} green({c['pose'][0]:.0f},{c['pose'][1]:.0f}) old({c['other'][0]:.0f},{c['other'][1]:.0f})", (3, 15), 0, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    for j in range(0, len(tiles), 15):
        part = tiles[j:j + 15]
        cv2.imwrite(f"{sys.argv[2]}{j // 15}.jpg", np.vstack([np.hstack(part[i:i + cols]) for i in range(0, len(part), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(len(CASES), "cases")
