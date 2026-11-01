"""Experiment: learn who the shooter is from the coach's dots. Every on-court person around each reviewed shot is a candidate, described by a
few features; a linear choice model picks one. Scored leave-one-game-out against his dots, beside the machine's dot on the same shots, with hidden cards reported
on their own because only those were placed without seeing the machine's dot."""
from __future__ import annotations

import glob
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret import location as L
from film.interpret import shooter_pose as SP
from film.interpret.camera import from_homography
from film.interpret.court import BASKET, WIDTH, fold
from film.interpret.flight3d import fit_path, rims_of
from film.interpret.location import Look, _carry, _iou, ball_sightings, find_flight
from film.interpret.zones import is_three, zone_of
from film.perceive.court_run import load_mappings, mapping_between
from film.perceive.shotspots import NO_RIM, PATH_AFTER_S, PATH_BEFORE_S, PATH_RELEASE_Z, _rims_in_anchor, read_looks, read_poses
from film.review import dotcheck, dotreview

CACHE = Path(sys.argv[1])
FEATURES = ["arms_ball", "arms", "seen", "rule", "path_near", "path_far", "no_path", "jump", "rim_near", "rim_mid", "t_early", "t_late", "no_flight", "size", "at_line"]


def tracks_of(frames):
    tracks, open_ = [], []
    for i, f in enumerate(frames):
        taken = set()
        for p in f["people"]:
            best, score = None, 0.25
            for k, tr in enumerate(tracks):
                li = max(tr)
                if k in taken or i - li > 3:
                    continue
                s = _iou(p["box"], tr[li]["box"])
                if s > score:
                    best, score = k, s
            if best is None:
                tracks.append({})
                best = len(tracks) - 1
            tracks[best][i] = p
            taken.add(best)
    return tracks


def arms_only(person):
    return SP.arms_and_ball(person, tuple(np.array(person["kp"])[9:11, :2].mean(0)))


def features_of(gd, m, maps, row, frames, e_t, arrival, length_default):
    looks = [Look(x["t"], mapping_between(maps, x["t"]), [tuple(b) for b in x["people"]], [tuple(b) for b in x["ball"]]) for x in row["frames"]]
    anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
    if anchor is None:
        return None
    scale = m.width / L.REFERENCE_WIDTH
    s = ball_sightings(looks, scale)
    rims = _rims_in_anchor(row, looks)
    flight = find_flight(s, rims, end_t=arrival, scale=scale)
    has_flight = len(flight) >= L.FLIGHT_MIN_POINTS
    rule = L.locate(looks, rims, arrival_t=arrival, scale=scale)
    cam = from_homography(anchor.H, m.width, m.height)
    path = None
    if cam is not None:
        lo, hi = (e_t - PATH_BEFORE_S, e_t + PATH_AFTER_S) if arrival is not None else (looks[0].t + PATH_AFTER_S, looks[-1].t)
        path = fit_path(cam, s, rims_of(anchor.length), lo, hi, anchor.length, scale, release_z=PATH_RELEASE_Z)
    camera_xy = tuple(cam.centre[:2]) if cam is not None else None
    start = flight[0][0] if has_flight else (path.release_t if path is not None else e_t - 1.2)
    ball_at = {}
    if has_flight:
        first, second = flight[0], flight[1]
        dt = second[0] - first[0]
        vx, vy = (second[1] - first[1]) / dt, (second[2] - first[2]) / dt
        seen = {round(p[0], 1): p for p in flight}
    out = []
    for tr in tracks_of(frames):
        if len(tr) < 3:
            continue
        best = None
        for i, p in tr.items():
            t = frames[i]["t"]
            if not start - SP.BEFORE_S <= t <= start + SP.AFTER_S:
                continue
            mp = mapping_between(maps, t)
            if mp is None:
                continue
            b = p["box"]
            if not mp.contains((b[0] + b[2]) / 2, b[3], margin=L.COURT_MARGIN_FT):
                continue
            ab = 0.0
            if has_flight:
                q = seen.get(round(t, 1))
                here = (q[1], q[2]) if q is not None else (first[1] - vx * (first[0] - t), first[2] - vy * (first[0] - t) + 0.5 * L.GRAVITY_PX * scale * (first[0] - t) ** 2)
                ball = _carry(anchor.H, np.linalg.inv(mp.H), *here)
                if ball is not None:
                    ab = SP.arms_and_ball(p, ball)
            a = arms_only(p)
            rank = ab + 0.1 * a
            if best is None or rank > best[0]:
                best = (rank, ab, a, i, p, mp)
        if best is None:
            continue
        _, ab, a, i, p, mp = best
        floor = SP.feet_on_floor(frames, i, p, lambda t: mapping_between(maps, t), camera_xy)
        at = mp.to_court(*SP.foot_pixel(p))
        if floor is None or at is None:
            continue
        hx, hy = fold(*floor, mp.length)
        hx2, hy2 = fold(*at, mp.length)
        dist = math.hypot(hx - BASKET[0], hy - BASKET[1])
        ankles = [frames[j]["people"] for j in ()]
        ys = [SP.foot_pixel(q)[1] for j, q in tr.items() if frames[i]["t"] - 0.6 <= frames[j]["t"] <= frames[i]["t"]]
        h = max(p["box"][3] - p["box"][1], 1.0)
        jump = max(0.0, (max(ys) - SP.foot_pixel(p)[1]) / h) if ys else 0.0
        rule_near = 0.0
        if rule.x is not None:
            rx, ry = rule.x, rule.y
            rule_near = float(math.hypot(floor[0] - rx, floor[1] - ry) <= 3.0 or math.hypot(at[0] - rx, at[1] - ry) <= 3.0)
        pgap = None if path is None else math.hypot(floor[0] - path.spot[0], floor[1] - path.spot[1])
        rel = frames[i]["t"] - start
        heights = sorted(q["box"][3] - q["box"][1] for f in frames[max(0, i - 1):i + 1] for q in f["people"])
        feats = {"arms_ball": ab, "arms": a, "seen": len(tr) / len(frames), "rule": rule_near, "path_near": 0.0 if pgap is None else math.exp(-pgap / 3.0), "path_far": 0.0 if pgap is None else min(pgap, 30.0) / 30.0,
                 "no_path": float(pgap is None), "jump": min(jump, 1.0), "rim_near": math.exp(-dist / 6.0), "rim_mid": min(dist, 40.0) / 40.0, "t_early": max(-rel, 0.0), "t_late": max(rel, 0.0),
                 "no_flight": float(not has_flight), "size": h / (median(heights) if heights else h), "at_line": math.exp(-math.hypot(hx - WIDTH / 2, hy - 19.0) / 2.0)}
        out.append({"f": [feats[k] for k in FEATURES], "floor": (hx, hy), "ankle": (hx2, hy2), "t": frames[i]["t"], "box": p["box"]})
    return out


def collect():
    answers = [(a, "review") for a in dotreview.read_answers() if not a.get("skipped") and a.get("source") == "coach" or (not a.get("skipped") and a.get("source") == "approved")]
    answers += [(a, "check") for a in dotcheck.read_answers() if not a.get("skipped") and not a.get("doubtful")]
    by = defaultdict(list)
    for a, kind in answers:
        by[a["game_id"]].append((a, kind))
    data = []
    for game, group in by.items():
        gd = game_dir(game)
        m = Manifest.load(gd)
        maps = load_mappings(gd)
        looks, poses = {r["id"]: r for r in read_looks(gd)}, read_poses(gd)
        from film.interpret.events import load_events

        events = {e.id: e for e in load_events(gd)}
        for a, kind in group:
            r, frames, e = looks.get(a["event_id"]), poses.get(a["event_id"]), events.get(a["event_id"])
            if r is None or frames is None or e is None:
                continue
            cands = features_of(gd, m, maps, r, frames, e.t, e.t if r["rim_id"] >= NO_RIM else None, 94.0)
            if not cands:
                continue
            independent = kind == "check" or bool(a.get("blind"))
            data.append({"game": game, "id": a["event_id"], "x": a["x"], "y": a["y"], "mx": a["machine_x"], "my": a["machine_y"], "independent": independent, "source": a.get("source", "check"),
                         "profile": m.profile, "made": e.made, "points": e.signals.get("points"), "c": cands, "conf": a.get("conf")})
        print(game[:40], len(data), flush=True)
    CACHE.write_bytes(pickle.dumps(data))
    return data


def fit(shots, l2=1e-2, steps=300):
    keep = [s for s in shots if s["label"] is not None]
    w = torch.zeros(len(FEATURES), dtype=torch.float64, requires_grad=True)
    X = [torch.tensor(np.array([c["f"] for c in s["c"]]), dtype=torch.float64) for s in keep]
    Y = [s["label"] for s in keep]
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = sum(-torch.log_softmax(x @ w, 0)[y] for x, y in zip(X, Y)) / len(X) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def label(s, radius):
    d = [math.hypot(c["floor"][0] - s["x"], c["floor"][1] - s["y"]) for c in s["c"]]
    i = int(np.argmin(d))
    return i if d[i] <= radius else None


def report(name, rows, key):
    rows = [r for r in rows if r.get(key) is not None]
    if not rows:
        return
    e = [math.hypot(r["x"] - r[key][0], r["y"] - r[key][1]) for r in rows]
    print(f"  {name:22s} n{len(rows):3d} median {median(e):5.1f} within3 {sum(v <= 3 for v in e) / len(e):4.0%} within6 {sum(v <= 6 for v in e) / len(e):4.0%} over10 {sum(v > 10 for v in e) / len(e):4.0%} | arc {sum(is_three(r['x'], r['y']) == is_three(*r[key]) for r in rows) / len(rows):4.0%} zone {sum(zone_of(r['x'], r['y']) == zone_of(*r[key]) for r in rows) / len(rows):4.0%}")


if __name__ == "__main__":
    data = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else collect()
    print("shots", len(data), "independent", sum(d["independent"] for d in data), "candidates per shot", round(np.mean([len(d["c"]) for d in data]), 1))
    for radius in (3.0, 5.0):
        for d in data:
            d["label"] = label(d, radius)
        print("radius", radius, "shots with a candidate on his dot", sum(d["label"] is not None for d in data))
    for d in data:
        d["label"] = label(d, 4.0)
    games = sorted({d["game"] for d in data})
    for g in games:
        w = fit([d for d in data if d["game"] != g])
        for d in data:
            if d["game"] == g:
                p = np.array([c["f"] for c in d["c"]]) @ w
                probs = np.exp(p - p.max())
                probs = probs / probs.sum()
                i = int(np.argmax(p))
                d["model"] = d["c"][i]["floor"]
                d["model_ankle"] = d["c"][i]["ankle"]
                d["prob"] = float(probs[i])
                d["machine"] = None if d["mx"] is None else (d["mx"], d["my"])
    w = fit(data)
    print("weights", {k: round(float(v), 2) for k, v in zip(FEATURES, w)})
    for name, rows in (("HIDDEN (independent)", [d for d in data if d["independent"]]), ("all his dots", data), ("all, machine had a dot", [d for d in data if d["mx"] is not None])):
        print(name)
        report("machine", rows, "machine")
        report("model, feet on floor", rows, "model")
        report("model, ankles", rows, "model_ankle")
    ind = [d for d in data if d["independent"]]
    for lo, hi in ((0, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        print(f"model prob {lo}-{hi}")
        report("hidden", [d for d in ind if lo <= d["prob"] < hi], "model")
        report("all", [d for d in data if lo <= d["prob"] < hi], "model")
    for p in sorted({d["profile"] for d in data}):
        print(p)
        report("machine", [d for d in data if d["profile"] == p], "machine")
        report("model", [d for d in data if d["profile"] == p], "model")
