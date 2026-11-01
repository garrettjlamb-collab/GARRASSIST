"""The court in every frame from a few frames the coach marked: the references.

A reference is a court frame he labelled in the Label court tab (datasets/court, id `<game_id>_<t*10:06d>`). Each
sampled frame is lined up with every reference it shares enough of the gym with (film/perceive/register.py); the
references' marked points are carried into the frame and fit_mapping turns them into the court mapping, so points
from several references pin one fit. A frame no reference lines up with is carried from the last placed frame, for
at most MAX_CHAIN_S after a reference last placed one. The answer is court.parquet, the file the court finder writes.

A Hudl Focus camera never moves, so one game's references place every game filmed from it (`refs_from`)."""
from __future__ import annotations

import glob
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

import cv2
import numpy as np

import film.train.datasets as ds
from film.common.manifest import Manifest
from film.common.paths import datasets_dir, game_dir
from film.common.video import iter_frames
from film.interpret.court import END_OF_POINT, FULL_DEPTH, Mapping, fit_mapping, template
from film.perceive.court_run import default_step, mapping_row, write_mappings
from film.perceive.register import Features, features, masks_for, match, project

DATASET = "court"
REF_ID = re.compile(r"^(?P<game>.+)_(?P<t10>\d{6})$")
MAX_CHAIN_S = 30.0
CHAIN_CONF = 0.8                  # a carried court is trusted less than one a reference placed
CENTRE_MARGIN_FT = 15.0           # the middle of the picture must be on (or near) the court
FIT_TOL_PX_AT_1080P = 8.0
TRAIN_MIN_CONF = 0.8
TRAIN_MIN_POINTS = 4
LENGTHS_FT = np.arange(70.0, 94.01, 0.5)    # court lengths tried when measuring one
STANDARD_LENGTHS_FT = (74.0, 84.0, 94.0)    # junior high, high school, college
SNAP_FT = 3.0
RESIDUAL_CAP_PX = 20.0                      # one slipped click cannot outvote the rest


@dataclass(frozen=True)
class Reference:
    image_id: str
    t: float
    points: dict[str, tuple[float, float]]
    features: Features


@dataclass(frozen=True)
class _Carried:
    features: Features
    points: dict[str, tuple[float, float]]
    anchor_t: float               # when a reference last placed the court


def reference_ids(game_id: str) -> list[str]:
    """The game's marked court frames. Machine-made labels (ids ending in "a") and other games are not references."""
    root = datasets_dir() / DATASET / "images"
    if not root.exists():
        return []
    ids = sorted({p.stem for p in root.glob(f"*/{glob.escape(game_id)}_*.jpg")})
    return [i for i in ids if (m := REF_ID.match(i)) is not None and m.group("game") == game_id]


def load_references(game_id: str, masks: list | tuple) -> list[Reference]:
    refs = []
    for image_id in reference_ids(game_id):
        got = ds.read_points(DATASET, image_id)
        if got is None or not got[1]:
            continue
        image, points, _, _ = got
        refs.append(Reference(image_id=image_id, t=int(REF_ID.match(image_id).group("t10")) / 10.0, points=points,
                              features=features(cv2.imread(str(image)), masks)))
    return refs


def place_points(f: Features, refs: list[Reference]) -> tuple[dict[str, tuple[float, float]], int]:
    """The references' points carried into this frame, from every reference it lines up with; where two give the
    same point the one with more matches wins. Also the best match's inlier count (0 when none lines up)."""
    found = []
    for r in refs:
        m = match(r.features, f)
        if m is not None:
            found.append((m.inliers, project(r.points, m.H)))
    points: dict[str, tuple[float, float]] = {}
    for _, p in sorted(found, key=lambda x: x[0]):
        points.update(p)
    return points, max((i for i, _ in found), default=0)


def _fit(points: dict, w: int, h: int, length: float = FULL_DEPTH) -> Mapping | None:
    mapping = fit_mapping(points, reproj_tol_px=max(3.0, FIT_TOL_PX_AT_1080P * w / 1920), length=length)
    if mapping is None or not mapping.contains(w / 2, h / 2, margin=CENTRE_MARGIN_FT):
        return None
    return mapping


def points_in_view(mapping: Mapping, w: int, h: int) -> dict[str, tuple[float, float]]:
    """The template's points through the fitted mapping, those inside the picture only."""
    back = np.linalg.inv(mapping.H)
    out = {}
    for name, feet in template(mapping.length).items():
        v = back @ np.array([*feet, 1.0])
        if v[2] <= 0:
            continue
        x, y = float(v[0] / v[2]), float(v[1] / v[2])
        if 0 <= x < w and 0 <= y < h:
            out[name] = (x, y)
    return out


def measure_length(refs: list[Reference]) -> float | None:
    """The court's length from the references: each one's own points plus the others' carried into it; wherever
    points of both ends meet in one picture, the length that fits them best (snapped to 74, 84 or 94 ft when within
    3 ft). None when no picture holds points of an end and of midcourt or the other end."""
    sets = []
    for r in refs:
        carried, _ = place_points(r.features, [o for o in refs if o.image_id != r.image_id])
        pts = {**carried, **r.points}
        ends = Counter(END_OF_POINT[n] for n in pts)
        two_marks = sum(ends[e] >= 2 for e in ("near", "far", "mid")) >= 2 and (ends["near"] >= 2 or ends["far"] >= 2)
        if two_marks and len(pts) >= 6:
            sets.append(pts)
    if not sets:
        return None

    def cost(length: float) -> float:
        T, total = template(length), 0.0
        for pts in sets:
            names = list(pts)
            src = np.array([T[n] for n in names], np.float64)
            dst = np.array([pts[n] for n in names], np.float64)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
            if H is None:
                return float("inf")
            proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
            total += float(np.minimum(np.hypot(*(proj - dst).T), RESIDUAL_CAP_PX).mean())
        return total

    best = float(min(LENGTHS_FT, key=cost))
    standard = min(STANDARD_LENGTHS_FT, key=lambda s: abs(s - best))
    return standard if abs(standard - best) <= SNAP_FT else best


def _save_training_frame(game_id: str, t: float, frame: np.ndarray, mapping: Mapping) -> bool:
    h, w = frame.shape[:2]
    points = points_in_view(mapping, w, h)
    image_id = f"{game_id}_{int(round(t * 10)):06d}a"
    if len(points) < TRAIN_MIN_POINTS or ds.is_known(DATASET, image_id):
        return False
    ds.write_pending(DATASET, image_id, frame, [], extra={"t": t, "points": {}, "auto": True})
    ds.accept_points(DATASET, image_id, points)
    return True


def refs_pass(game_id: str, refs_from: str | None = None, step_s: float | None = None, frames=None,
              train_frames: int = 0, log=print) -> dict:
    """Fit the court mapping every step_s seconds from the references and write court.parquet. With train_frames,
    also save that many evenly spaced reference-placed frames as court-finder training labels."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    masks = masks_for(m)
    source_game = refs_from or game_id
    refs = load_references(source_game, masks)
    if not refs:
        raise RuntimeError(f"{source_game} has no reference pictures: `film prelabel court {source_game} --refs`, "
                           "then mark them in the Label court tab")
    measured = measure_length(refs)
    length = measured or FULL_DEPTH
    step = step_s or default_step(m.profile)
    if frames is None:
        frames = iter_frames(Path(m.source), fps=1.0 / step)
    train_every = m.duration_s / train_frames if train_frames > 0 else None
    next_train = 0.0
    rows, counts, saved, n, started = [], Counter(), 0, 0, time.time()
    carried: _Carried | None = None
    for t, frame in frames:
        n += 1
        h, w = frame.shape[:2]
        f = features(frame, masks)
        points, _ = place_points(f, refs)
        source = "ref"
        if not points and carried is not None and t - carried.anchor_t <= MAX_CHAIN_S:
            back = match(carried.features, f)
            if back is not None:
                points, source = project(carried.points, back.H), "chain"
        mapping = _fit(points, w, h, length) if points else None
        if mapping is None:
            continue
        if source == "chain":
            mapping = replace(mapping, conf=round(mapping.conf * CHAIN_CONF, 3))
        rows.append(mapping_row(t, m.fps, mapping, source))
        counts[source] += 1
        carried = _Carried(f, points, t if source == "ref" else carried.anchor_t)
        if train_every is not None and source == "ref" and mapping.conf >= TRAIN_MIN_CONF and t + 1e-6 >= next_train:
            if _save_training_frame(game_id, t, frame, mapping):
                saved += 1
                next_train = t + train_every
    write_mappings(gd, rows)
    m.model_versions["court"] = f"refs@{source_game}:{len(refs)}"
    m.mark("court", "done", frames=n, fitted=len(rows), step_s=step, refs=len(refs), refs_from=source_game,
           length_ft=length, length_measured=measured is not None,
           from_ref=counts["ref"], chained=counts["chain"], training_frames=saved, seconds=round(time.time() - started, 1))
    m.save(gd)
    log(f"court {game_id}: {len(rows)} of {n} frames placed ({counts['ref']} by a reference, {counts['chain']} carried) "
        f"from {len(refs)} reference pictures of {source_game}; court {length:g} ft long"
        + (" (measured)" if measured else " (assumed: no picture shows both ends)")
        + (f"; {saved} training frames saved" if train_every else ""))
    return m.stages["court"]


def check_references(game_id: str, masks: list | tuple | None = None) -> list[dict]:
    """Each reference against the others: their points carried into it, compared with the coach's own clicks there.
    A big error on one point is a slipped click; a reference that lines up with none of the others is fine (it sees
    a part of the gym they do not), it just cannot be checked."""
    if masks is None:
        masks = masks_for(Manifest.load(game_dir(game_id)))
    refs = load_references(game_id, masks)
    out = []
    for r in refs:
        others = [o for o in refs if o.image_id != r.image_id]
        carried, _ = place_points(r.features, others)
        errors = {n: float(np.hypot(carried[n][0] - x, carried[n][1] - y)) for n, (x, y) in r.points.items() if n in carried}
        worst = max(errors, key=errors.get) if errors else None
        out.append({"image_id": r.image_id, "t": r.t, "points": len(r.points), "lined_up": bool(carried),
                    "shared": len(errors), "median_px": round(median(errors.values()), 2) if errors else None,
                    "max_px": round(errors[worst], 2) if worst else None, "worst": worst})
    return out


def _ref_id(game_id: str, t: float) -> str:
    return f"{game_id}_{int(round(t * 10)):06d}"


K_MIN = 3


def _spread_picks(feats, links: list[set[int]], anchor: Features, candidates: set[int]) -> list[int]:
    """The candidates looking furthest left and furthest right of the anchor picture: where each frame's centre lands
    in the anchor's pixels, through a chain of neighbours for a frame that does not line up with the anchor itself.
    A wide picture that lines up with everything still cuts off the baselines; these see them."""
    to_anchor: dict[int, np.ndarray] = {}
    frontier = []
    for i in range(len(feats)):
        m = match(feats[i][1], anchor)
        if m is not None:
            to_anchor[i] = m.H
            frontier.append(i)
    while frontier:
        reached = []
        for i in frontier:
            for j in sorted(links[i]):
                if j in to_anchor:
                    continue
                m = match(feats[j][1], feats[i][1])
                if m is not None:
                    to_anchor[j] = to_anchor[i] @ m.H
                    reached.append(j)
        frontier = reached
    xs = {}
    for i in candidates & set(to_anchor):
        f = feats[i][1]
        v = to_anchor[i] @ np.array([f.width / 2, f.height / 2, 1.0])
        if v[2] > 1e-9:                                   # composed far round, a frame can land behind the anchor
            xs[i] = float(v[0] / v[2])
    if not xs:
        return []
    centre = anchor.width / 2                             # the end further from the anchor's own view first
    ends = sorted(dict.fromkeys([min(xs, key=xs.get), max(xs, key=xs.get)]), key=lambda i: -abs(xs[i] - centre))
    return ends


def choose_references(game_id: str, n: int = 90, k_max: int = 5, coverage: float = 0.95, samples=None,
                      margin_s: float = 30.0, log=print) -> tuple[list[float], float]:
    """Which frames to ask the coach to mark: n frames spread through the game, each lined up with every other,
    then a greedy cover: the frame that lines up with the most not-yet-covered frames, until `coverage` of them line
    up with a reference or there are k_max references. References he already marked count first; a frame he skipped
    is never asked for again. At least K_MIN references in all: when fewer cover the game, the views turned furthest
    left and right of the first are added. Returns the chosen times and the share of the sampled frames they cover."""
    from film.train.prelabel import first_frame_at, sample_times

    m = Manifest.load(game_dir(game_id))
    masks = masks_for(m)
    if samples is None:
        samples = ((t, first_frame_at(Path(m.source), t, m.fps)) for t in sample_times(m.duration_s, n, margin_s))
    feats = [(t, features(frame, masks)) for t, frame in samples if frame is not None]
    if not feats:
        return [], 0.0
    links: list[set[int]] = [set() for _ in feats]
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            if match(feats[i][1], feats[j][1]) is not None:
                links[i].add(j)
                links[j].add(i)
    existing = load_references(game_id, masks)
    covered = {j for j, (_, f) in enumerate(feats) if any(match(r.features, f) is not None for r in existing)}
    candidates = {i for i, (t, _) in enumerate(feats) if not ds.is_known(DATASET, _ref_id(game_id, t))}
    chosen: list[int] = []
    while len(existing) + len(chosen) < k_max and len(covered) < coverage * len(feats) and candidates:
        gain, _, best = max((len(({i} | links[i]) - covered), len(links[i]), i) for i in candidates)
        if gain == 0:
            break
        chosen.append(best)
        candidates.discard(best)
        covered |= {best} | links[best]
    if len(existing) + len(chosen) < K_MIN and (existing or chosen):
        anchor = existing[0].features if existing else feats[chosen[0]][1]
        for i in _spread_picks(feats, links, anchor, candidates):
            if len(existing) + len(chosen) >= min(K_MIN, k_max):
                break
            chosen.append(i)
            candidates.discard(i)
    share = len(covered) / len(feats)
    log(f"references for {game_id}: {len(existing)} marked, {len(chosen)} to mark; together they line up with "
        f"{len(covered)} of {len(feats)} sampled frames ({share:.0%})")
    return sorted(feats[i][0] for i in chosen), share
