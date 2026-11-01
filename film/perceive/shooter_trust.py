"""How far to trust a machine dot, learned from the coach's dots. Every person around a shot is a candidate, described by a few features, and a linear choice model
picks one. On games it had not seen it picked no better than the rules (54% good enough against 56%), so it does not move a dot. What it adds is its certainty: where it is
sure and lands on the machine's dot, big misses nearly vanish. `film train shooter` learns it, `film dotreview trust` writes it onto the events."""
from __future__ import annotations

import math
from statistics import median

import numpy as np
import torch

from film.common import registry
from film.common.manifest import Manifest
from film.common.paths import game_dir, games_dir
from film.interpret import location as L
from film.interpret import shooter_pose as SP
from film.interpret.camera import from_homography
from film.interpret.court import BASKET, FT_LINE_Y, WIDTH, fold
from film.interpret.events import load_events, save_events
from film.interpret.flight3d import fit_path, rims_of
from film.interpret.location import Look, _carry, _iou, ball_sightings, find_flight
from film.interpret.zones import is_three, zone_of
from film.perceive.court_run import load_mappings, mapping_between
from film.perceive.shotspots import NO_RIM, PATH_AFTER_S, PATH_BEFORE_S, PATH_RELEASE_Z, LOOKS, _rims_in_anchor, confirmed_shots, read_looks, read_poses

FEATURES = ["arms_ball", "arms", "seen", "rule", "path_near", "path_far", "no_path", "jump", "rim_near", "rim_mid", "t_early", "t_late", "no_flight", "size", "at_line",
            "near_now", "top_arms_ball", "ab_margin"]
PAINT_ZONES = ("paint", "restricted")
PAINT_TOL_FT, OTHER_TOL_FT = 6.0, 4.0      # he nudged layups and says the paint has room; elsewhere the arc is close by
AGREE_FT = 3.0
TRUST_PROB = 0.6
L2, STEPS = 1e-2, 300
MIN_TRACK_FRAMES = 3
TRACK_IOU = 0.25


def good_enough(his: tuple[float, float], dot: tuple[float, float] | None) -> bool:
    """A dot of his in the paint counts the machine's within 6 ft; elsewhere within 4 ft and on the same side of the arc."""
    if dot is None:
        return False
    gap = math.hypot(his[0] - dot[0], his[1] - dot[1])
    if zone_of(*his) in PAINT_ZONES:
        return gap <= PAINT_TOL_FT
    return gap <= OTHER_TOL_FT and is_three(*his) == is_three(*dot)


def _tracks(frames: list[dict]) -> list[dict]:
    tracks: list[dict] = []
    for i, f in enumerate(frames):
        taken = set()
        for p in f["people"]:
            best, score = None, TRACK_IOU
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


def _arms(person: dict) -> float:
    return SP.arms_and_ball(person, tuple(np.array(person["kp"])[9:11, :2].mean(0)))


def candidates(row: dict, frames: list[dict], maps, width: int, height: int, t: float, now: tuple[float, float] | None) -> list[dict]:
    """Each person around the shot: their features, and where their feet were on the half court."""
    looks = [Look(x["t"], mapping_between(maps, x["t"]), [tuple(b) for b in x["people"]], [tuple(b) for b in x["ball"]]) for x in row["frames"]]
    anchor = next((k.mapping for k in reversed(looks) if k.mapping is not None), None)
    if anchor is None:
        return []
    arrival = t if row["rim_id"] >= NO_RIM else None
    scale = width / L.REFERENCE_WIDTH
    seen_balls = ball_sightings(looks, scale)
    rims = _rims_in_anchor(row, looks)
    flight = find_flight(seen_balls, rims, end_t=arrival, scale=scale)
    has_flight = len(flight) >= L.FLIGHT_MIN_POINTS
    rule = L.locate(looks, rims, arrival_t=arrival, scale=scale)
    cam = from_homography(anchor.H, width, height)
    path = None
    if cam is not None:
        lo, hi = (t - PATH_BEFORE_S, t + PATH_AFTER_S) if arrival is not None else (looks[0].t + PATH_AFTER_S, looks[-1].t)
        path = fit_path(cam, seen_balls, rims_of(anchor.length), lo, hi, anchor.length, scale, release_z=PATH_RELEASE_Z)
    camera_xy = tuple(cam.centre[:2]) if cam is not None else None
    start = flight[0][0] if has_flight else (path.release_t if path is not None else t - 1.2)
    if has_flight:
        first, second = flight[0], flight[1]
        dt = second[0] - first[0]
        vx, vy = (second[1] - first[1]) / dt, (second[2] - first[2]) / dt
        on_flight = {round(p[0], 1): p for p in flight}
    out = []
    for tr in _tracks(frames):
        if len(tr) < MIN_TRACK_FRAMES:
            continue
        best = None
        for i, p in tr.items():
            ft = frames[i]["t"]
            if not start - SP.BEFORE_S <= ft <= start + SP.AFTER_S:
                continue
            mp = mapping_between(maps, ft)
            b = p["box"]
            if mp is None or not mp.contains((b[0] + b[2]) / 2, b[3], margin=L.COURT_MARGIN_FT):
                continue
            ab = 0.0
            if has_flight:
                q = on_flight.get(round(ft, 1))
                back = first[0] - ft
                here = (q[1], q[2]) if q is not None else (first[1] - vx * back, first[2] - vy * back + 0.5 * L.GRAVITY_PX * scale * back * back)
                ball = _carry(anchor.H, np.linalg.inv(mp.H), *here)
                if ball is not None:
                    ab = SP.arms_and_ball(p, ball)
            a = _arms(p)
            if best is None or ab + 0.1 * a > best[0]:
                best = (ab + 0.1 * a, ab, a, i, p, mp)
        if best is None:
            continue
        _, ab, a, i, p, mp = best
        floor = SP.feet_on_floor(frames, i, p, lambda s: mapping_between(maps, s), camera_xy)
        if floor is None:
            continue
        hx, hy = fold(*floor, mp.length)
        dist = math.hypot(hx - BASKET[0], hy - BASKET[1])
        ys = [SP.foot_pixel(q)[1] for j, q in tr.items() if frames[i]["t"] - 0.6 <= frames[j]["t"] <= frames[i]["t"]]
        h = max(p["box"][3] - p["box"][1], 1.0)
        rule_near = float(rule.x is not None and math.hypot(floor[0] - rule.x, floor[1] - rule.y) <= 3.0)
        gap = None if path is None else math.hypot(floor[0] - path.spot[0], floor[1] - path.spot[1])
        rel = frames[i]["t"] - start
        heights = [q["box"][3] - q["box"][1] for q in frames[i]["people"]]
        out.append({"ab": ab, "floor": (hx, hy), "f": [ab, a, len(tr) / len(frames), rule_near, 0.0 if gap is None else math.exp(-gap / 3.0), 0.0 if gap is None else min(gap, 30.0) / 30.0,
                                                      float(gap is None), min(max(0.0, (max(ys) - SP.foot_pixel(p)[1]) / h), 1.0) if ys else 0.0, math.exp(-dist / 6.0), min(dist, 40.0) / 40.0,
                                                      max(-rel, 0.0), max(rel, 0.0), float(not has_flight), h / (median(heights) if heights else h), math.exp(-math.hypot(hx - WIDTH / 2, hy - FT_LINE_Y) / 2.0),
                                                      0.0 if now is None else math.exp(-math.hypot(hx - now[0], hy - now[1]) / 2.0)]})
    if out:
        abs_ = np.array([c["ab"] for c in out])
        top, second = float(abs_.max()), float(np.sort(abs_)[-2]) if len(abs_) > 1 else 0.0
        for c in out:
            is_top = c["ab"] == top and top > 0
            c["f"] += [float(is_top), top - second if is_top else 0.0]
    return out


def label_of(cands: list[dict], his: tuple[float, float]) -> int | None:
    """The candidate on his dot, within the tolerance for where it is."""
    if not cands:
        return None
    gaps = [math.hypot(c["floor"][0] - his[0], c["floor"][1] - his[1]) for c in cands]
    i = int(np.argmin(gaps))
    return i if gaps[i] <= (PAINT_TOL_FT if zone_of(*his) in PAINT_ZONES else OTHER_TOL_FT) else None


def fit_model(shots: list[dict]) -> np.ndarray:
    """A linear choice among each shot's candidates. `shots`: {"c": candidates, "label": index or None}."""
    keep = [s for s in shots if s["label"] is not None]
    w = torch.zeros(len(FEATURES), dtype=torch.float64, requires_grad=True)
    X = [torch.tensor(np.array([c["f"] for c in s["c"]]), dtype=torch.float64) for s in keep]
    Y = [s["label"] for s in keep]
    opt = torch.optim.LBFGS([w], max_iter=STEPS, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = sum(-torch.log_softmax(x @ w, 0)[y] for x, y in zip(X, Y)) / len(X) + L2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def judge(weights: np.ndarray, cands: list[dict], now: tuple[float, float] | None) -> dict | None:
    """The model's pick and certainty, and whether that makes the machine's dot one to trust."""
    if not cands:
        return None
    scores = np.array([c["f"] for c in cands]) @ weights
    probs = np.exp(scores - scores.max())
    probs = probs / probs.sum()
    i = int(np.argmax(scores))
    pick = cands[i]["floor"]
    agree = now is not None and math.hypot(pick[0] - now[0], pick[1] - now[1]) <= AGREE_FT
    return {"prob": round(float(probs[i]), 3), "agree": bool(agree), "trusted": bool(agree and probs[i] >= TRUST_PROB), "pick": (round(pick[0], 1), round(pick[1], 1))}


def _game_inputs(gd):
    m = Manifest.load(gd)
    return m, load_mappings(gd), {r["id"]: r for r in read_looks(gd)}, read_poses(gd), {e.id: e for e in load_events(gd)}


def training_shots(answers: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for a in answers:
        by.setdefault(a["game_id"], []).append(a)
    shots = []
    for game, group in sorted(by.items()):
        m, maps, looks, poses, events = _game_inputs(game_dir(game))
        for a in group:
            r, frames, e = looks.get(a["event_id"]), poses.get(a["event_id"]), events.get(a["event_id"])
            if r is None or frames is None or e is None:
                continue
            now = None if e.x is None else (e.x, e.y)
            cands = candidates(r, frames, maps, m.width, m.height, e.t, now)
            if cands:
                his = (a["x"], a["y"])
                shots.append({"game": game, "id": a["event_id"], "his": his, "now": now, "c": cands, "label": label_of(cands, his), "independent": bool(a.get("blind")) or "blind" not in a})
    return shots


def held_out_report(shots: list[dict]) -> dict:
    """Each game judged by a model that never saw it. Only shots with a machine dot."""
    for g in sorted({s["game"] for s in shots}):
        w = fit_model([s for s in shots if s["game"] != g])
        for s in shots:
            if s["game"] == g:
                s["judged"] = judge(w, s["c"], s["now"])

    def share(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        err = [math.hypot(s["his"][0] - s["now"][0], s["his"][1] - s["now"][1]) for s in rows]
        return {"n": len(rows), "good_enough": round(sum(good_enough(s["his"], s["now"]) for s in rows) / len(rows), 2), "median_ft": round(median(err), 1), "over_10ft": round(sum(e > 10 for e in err) / len(err), 2)}

    rows = [s for s in shots if s["now"] is not None]
    trusted, rest = [s for s in rows if s["judged"]["trusted"]], [s for s in rows if not s["judged"]["trusted"]]
    return {"trusted": share(trusted), "not_trusted": share(rest), "trusted_hidden": share([s for s in trusted if s["independent"]]), "not_trusted_hidden": share([s for s in rest if s["independent"]])}


def train(answers: list[dict], log=print) -> dict:
    shots = training_shots(answers)
    report = held_out_report(shots)
    version, folder = registry.new_version_dir("shooter")
    weights = fit_model(shots)
    np.save(folder / "weights.npy", weights)
    (folder / "best.pt").write_bytes(b"")                 # registry.latest looks for this name
    registry.write_manifest(folder, shots=len(shots), features=FEATURES, held_out=report)
    log(f"shooter@{version}: {len(shots)} shots; on games it did not see {report}")
    return report


def load_weights() -> np.ndarray | None:
    found = registry.latest("shooter")
    return None if found is None else np.load(found[1] / "weights.npy")


def write_trust(log=print) -> dict:
    """Judge every confirmed shot that has poses and write it onto its event. Dots do not move."""
    weights = load_weights()
    if weights is None:
        raise RuntimeError("no shooter model: run film train shooter")
    total = {"judged": 0, "trusted": 0}
    for gd in sorted(p for p in games_dir().iterdir() if (p / LOOKS).exists()):
        m, maps, looks, poses, _ = _game_inputs(gd)
        events = load_events(gd)
        shots = {e.id for e in confirmed_shots(gd)}
        for e in events:
            r, frames = looks.get(e.id), poses.get(e.id)
            if e.id not in shots or r is None or frames is None:
                continue
            now = None if e.x is None else (e.x, e.y)
            got = judge(weights, candidates(r, frames, maps, m.width, m.height, e.t, now), now)
            if got is None:
                continue
            e.signals["trust"] = got
            total["judged"] += 1
            total["trusted"] += got["trusted"]
        save_events(gd, events)
    log(f"dot trust: {total['trusted']} of {total['judged']} shots trusted")
    return total
