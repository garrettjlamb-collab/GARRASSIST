"""Dead end, kept as a record (2026-11-01): made twos 98/103 fell to 62-87/103, it picks the passer. Experiment: choose the shooter as the player who had the ball, not whoever the first sighting of the flight touches. People are
followed through the window, and each sighting before the flight counts for whoever it is in the hands of. Scored on the scoreboard
(side of the arc) and on free throws (feet from the line, with the rule that puts them on the line off). The coach's dot-check answers
are not read here: they are the final test."""
from __future__ import annotations

import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from film.common.manifest import Manifest
from film.interpret import location as L
from film.interpret.camera import from_homography
from film.interpret.court import FT_LINE_Y, WIDTH, fold
from film.interpret.flight3d import fit_path, rims_of
from film.interpret.location import Look, _carry, _iou, _on_court, _reach, ball_sightings, find_flight
from film.interpret.zones import is_three
from film.perceive.court_run import load_mappings, mapping_between
from film.perceive.shotspots import NO_RIM, PATH_AFTER_S, PATH_BEFORE_S, PATH_RELEASE_Z, _rims_in_anchor, confirmed_shots, read_looks

HELD = ("2025-02-05-arcadia", "2025-02-05-jv-vs-marcos")
WINDOW_S, MIN_IOU = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5, 0.25


def tracks_of(looks):
    """Follow people through the window by overlap: {track: {look index: box}}."""
    tracks, open_ = {}, []
    for i, k in enumerate(looks):
        taken = set()
        for b in k.people:
            best, score = None, MIN_IOU
            for tid in open_:
                last_i = max(tracks[tid])
                if tid in taken or i - last_i > 3:
                    continue
                s = _iou(b, tracks[tid][last_i])
                if s > score:
                    best, score = tid, s
            if best is None:
                best = len(tracks)
                tracks[best] = {}
                open_.append(best)
            tracks[best][i] = b
            taken.add(best)
    return tracks


def holder(looks, sightings, flight, anchor, scale):
    """The track with the most sightings in its hands before the flight, the later the better, and the last frame it had the ball."""
    start = flight[0][0]
    inv = {}
    scores, last = defaultdict(float), {}
    tracks = tracks_of(looks)
    times = np.array([k.t for k in looks])
    onflight = {(p[0], p[1]) for p in flight[1:]}
    for t, x, y, conf in sightings:
        if not start - WINDOW_S <= t <= start + 0.05 or (t, x) in onflight:
            continue
        i = int(np.argmin(np.abs(times - t)))
        k = looks[i]
        if k.mapping is None:
            continue
        p = _carry(anchor.H, np.linalg.inv(k.mapping.H), x, y)
        if p is None:
            continue
        here = [(tid, tr[i]) for tid, tr in tracks.items() if i in tr and _on_court(tr[i], k.mapping) and _reach(tr[i], *p) == 0.0]
        if not here:
            continue
        # A ball in several people's reach counts for the one whose box it is most inside.
        tid, box = min(here, key=lambda h: abs((h[1][0] + h[1][2]) / 2 - p[0]) / max(h[1][2] - h[1][0], 1))
        recency = 1.0 - 0.5 * (start - t) / WINDOW_S
        scores[tid] += recency * max(conf, 0.3) ** 0.5
        if tid not in last or t > last[tid][0]:
            last[tid] = (t, i, box)
    if not scores:
        return None
    tid = max(scores, key=lambda a: (scores[a], last[a][0]))
    t, i, box = last[tid]
    return {"t": t, "box": box, "look": looks[i], "score": scores[tid], "second": max([v for a, v in scores.items() if a != tid], default=0.0), "n": len(scores)}


def feet(box, look):
    return look.mapping.to_court((box[0] + box[2]) / 2, box[3])


tot = defaultdict(list)
for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
    gd = Path(f).parent
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    rows = read_looks(gd)
    by = {r["id"]: r for r in rows}
    scale = m.width / L.REFERENCE_WIDTH
    res = defaultdict(list)
    for e in confirmed_shots(gd):
        r = by.get(e.id)
        if r is None:
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
        # The 3D path, as in shotspots._from_path, without the classifier.
        pathspot, psp = None, None
        cam = from_homography(anchor.H, m.width, m.height)
        if cam is not None:
            lo, hi = (e.t - PATH_BEFORE_S, e.t + PATH_AFTER_S) if arrival is not None else (looks[0].t + PATH_AFTER_S, looks[-1].t)
            path = fit_path(cam, s, rims_of(anchor.length), lo, hi, anchor.length, scale, release_z=PATH_RELEASE_Z)
            if path is not None:
                psp = path.spot
                at = min(looks, key=lambda k: abs(k.t - path.release_t))
                if at.mapping is not None:
                    near = []
                    for b in at.people:
                        a = at.mapping.to_court((b[0] + b[2]) / 2, b[3])
                        if a is not None and _on_court(b, at.mapping):
                            near.append((float(np.hypot(a[0] - psp[0], a[1] - psp[1])), a))
                    if near and min(near, key=lambda n: n[0])[0] <= 3.0:
                        pathspot = fold(*min(near, key=lambda n: n[0])[1], anchor.length)
        spots["path+rule"] = pathspot or spots.get("rule")
        h = holder(looks, s, flight, anchor, scale) if len(flight) >= L.FLIGHT_MIN_POINTS else None
        hsp = None
        if h is not None:
            a = feet(h["box"], h["look"])
            if a is not None:
                hsp = fold(*a, anchor.length)
                res["holders"].append(h["n"])
        spots["hold+rule"] = hsp or spots.get("rule")
        spots["path+hold+rule"] = pathspot or hsp or spots.get("rule")
        spots["hold+path+rule"] = hsp or pathspot or spots.get("rule")
        pts = e.signals.get("points") if e.made else None
        for name, sp in spots.items():
            if sp is None:
                continue
            res["dots " + name].append(1)
            if pts in (2, 3):
                res[f"{name} {pts}s"].append(is_three(*sp) == (pts == 3))
            if pts == 1:
                res[f"{name} ft"].append(float(np.hypot(sp[0] - WIDTH / 2, sp[1] - FT_LINE_Y)))
    key = "held" if gd.name.startswith(HELD) else "tuned"
    for k, v in res.items():
        tot[f"{key} {k}"] += v
    tot[key + " shots"] += [1] * len(confirmed_shots(gd))
for key in ("tuned", "held"):
    print(key.upper(), "shots", len(tot[key + " shots"]), "window", WINDOW_S)
    for name in ("rule", "path+rule", "hold+rule", "path+hold+rule", "hold+path+rule"):
        a, b, ft = tot[f"{key} {name} 2s"], tot[f"{key} {name} 3s"], tot[f"{key} {name} ft"]
        print(f"  {name:16s} dots {len(tot[f'{key} dots {name}']):3d} | 2s {sum(a)}/{len(a)} 3s {sum(b)}/{len(b)} | FT n{len(ft)} median {np.median(ft) if ft else 0:.1f} ft, within 3 ft {np.mean(np.array(ft) <= 3) if ft else 0:.0%}")
