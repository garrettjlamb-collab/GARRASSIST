from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import pandas as pd

from film.common.device import pick_device
from film.common.cloud import may_need_film
from film.common.manifest import Manifest
from film.common.paths import datasets_dir, game_dir
from film.interpret.ballpath import hoop_offset, read_ball_path
from film.interpret.rimview import read_rim_view
from film.interpret.events import load_events
from film.perceive.rims import basket_sides
from film.perceive.tracks import read_tracks
from film.review.decisions import load_decisions
from film.train.datasets import (
    find_class_images, list_pending, pending_images_dir, remove_pending, store_class_image, write_pending,
)
from film.train.montage import build_montage, rim_follower

EVENT_EXCLUSION_S = 3.0


def item_id(game_id: str, local_id: str) -> str:
    """Event ids repeat from game to game; a dataset item carries its game."""
    return f"{game_id}--{local_id}"


def path_file(image_id: str) -> Path:
    return datasets_dir() / "rimclf" / "paths" / f"{image_id}.json"


def save_path(image_id: str, path: dict, game_id: str, rim_id: int, t: float, detector: str | None = None,
              side: str | None = None) -> None:
    """The ball path next to the montage: the montage is what the coach looks at, the path is what trains."""
    target = path_file(image_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"game": game_id, "rim": int(rim_id), "t": round(float(t), 3), "detector": detector,
                                  "side": side, **path}))


def load_path(image_id: str) -> dict | None:
    target = path_file(image_id)
    return json.loads(target.read_text()) if target.exists() else None


def view_file(image_id: str) -> Path:
    return datasets_dir() / "rimclf" / "views" / f"{image_id}.jpg"


@may_need_film
def ensure_views(game_id: str, log=print, *, film) -> int:
    """Cut the rim view (see film.interpret.rimview) of every item of this game that has a ball path and no
    view yet; returns how many were cut. The hoop is placed per basket side from the game's stored paths."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    rims = read_tracks(gd, "rim")
    root = datasets_dir() / "rimclf" / "paths"
    paths = {p.stem: json.loads(p.read_text()) for p in sorted(root.glob(f"{game_id}--*.json"))} if root.exists() else {}
    by_side: dict = {}
    for path in paths.values():
        by_side.setdefault(path.get("side") or path["rim"], []).append(path)
    offsets = {side: hoop_offset(group) for side, group in by_side.items()}
    cut = 0
    for image_id, path in paths.items():
        target = view_file(image_id)
        if target.exists():
            continue
        box = rim_box_at(rims, int(round(path["t"] * m.fps)), path["rim"])
        if box is None:
            continue
        sheet = read_rim_view(film(), path["t"], box, hoop_dx=offsets[path.get("side") or path["rim"]],
                              rim_at=rim_follower(rims, path["rim"], m.fps))
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cut += 1
    if cut:
        log(f"rimclf views for {game_id}: {cut} cut")
    return cut


def _nearest_rim_id(rims: pd.DataFrame, frame: int) -> int:
    return int(rims.loc[(rims["frame"].astype(int) - frame).abs().idxmin(), "track_id"])


def rim_box_at(rims: pd.DataFrame, frame: int, rim_id: int | None = None, max_gap: int = 10) -> tuple | None:
    """The rim box at the nearest tracked frame within max_gap frames, for a given rim id or any rim."""
    if rims.empty:
        return None
    df = rims if rim_id is None else rims[rims["track_id"] == rim_id]
    df = df.dropna(subset=["x1", "y1", "x2", "y2"])
    if df.empty:
        return None
    gaps = (df["frame"].astype(int) - frame).abs()
    i = gaps.idxmin()
    if int(gaps.loc[i]) > max_gap:
        return None
    row = df.loc[i]
    return float(row.x1), float(row.y1), float(row.x2), float(row.y2)


def _auto_made(e) -> bool:
    """Only a clean scoreboard verdict of made, worth 2 or 3, is trusted without a human look."""
    return e.type == "shot" and e.signals.get("scoreboard") == "made" and e.signals.get("points") in (2, 3)


def _route(e, decision: dict | None) -> tuple[str | None, str | None]:
    """(class to store straight away, or None) and, when None, the class to suggest in the Label tab.

    The coach's review comes first: confirmed with a made/missed call goes straight into that class,
    rejected into none, unsure (or confirmed with no call) to pending. Without a decision, only a clean
    scoreboard make is trusted."""
    if decision:
        status = decision.get("status")
        if status == "rejected":
            # A rejected warmup card may only mean "not part of the game", not "nothing at the rim".
            if "warmup" in e.flags:
                return None, "none"
            return "none", None
        if status == "confirmed":
            made = decision["made"] if "made" in decision else e.made
            if made is True:
                return "made", None
            if made is False:
                return "missed", None
            return None, "made" if e.signals.get("scoreboard") == "made" else "missed"
        if status == "unsure":
            return None, "missed" if e.type == "shot" else "none"
    if _auto_made(e):
        return "made", None
    return None, "missed" if e.type == "shot" else "none"


@may_need_film
def build_rimclf_candidates(game_id: str, n_none: int = 40, seed: int = 0, margin_s: float = 30.0,
                            exclusion_s: float = EVENT_EXCLUSION_S, detector=None, device: str | None = None,
                            log=print, *, film) -> dict:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    events = load_events(gd)
    rims = read_tracks(gd, "rim")
    decisions = load_decisions(gd)
    sides = basket_sides(rims, m.width)
    if detector is None:
        from film.perceive.detectors import load_detector

        detector = load_detector("ball", pick_device(device))

    def store_path(image_id: str, t: float, box, rim_id: int) -> None:
        path = read_ball_path(film(), t, box, m.width, m.height, detector, rim_at=rim_follower(rims, rim_id, m.fps))
        save_path(image_id, path, game_id, rim_id, t, detector=detector.name, side=sides.get(int(rim_id)))

    def refresh_path(image_id: str, t: float, rim_id: int) -> None:
        """A path is only as good as the ball detector that read it: after a retrain, read it again."""
        stored = load_path(image_id)
        if stored is not None and stored.get("detector") == detector.name:
            if stored.get("side") is None and sides.get(int(rim_id)) is not None:     # stored before sides existed
                stored["side"] = sides[int(rim_id)]
                path_file(image_id).write_text(json.dumps(stored))
            return
        box = rim_box_at(rims, int(round(t * m.fps)), rim_id)
        if box is not None:
            store_path(image_id, t, box, rim_id)
            counts["refreshed"] += 1

    counts = {"made_auto": 0, "pending": 0, "none_auto": 0, "already": 0, "no_rim": 0, "from_review": 0, "refreshed": 0}
    queued = set(list_pending("rimclf"))
    for e in events:
        image_id = item_id(game_id, e.id)
        if image_id in queued and e.id in decisions:
            # Queued by an earlier run; if the coach has since given a class, promote the queued montage.
            cls, _ = _route(e, decisions[e.id])
            image = cv2.imread(str(pending_images_dir("rimclf") / f"{image_id}.jpg")) if cls is not None else None
            if image is not None:
                store_class_image("rimclf", image_id, image, cls)
                remove_pending("rimclf", image_id)
                refresh_path(image_id, e.t, e.rim_id)
                counts["from_review"] += 1
                continue
        if image_id in queued or find_class_images("rimclf", image_id):
            refresh_path(image_id, e.t, e.rim_id)
            counts["already"] += 1
            continue
        box = None if e.rim_id < 0 else rim_box_at(rims, int(round(e.t * m.fps)), e.rim_id)
        if box is None:
            counts["no_rim"] += 1
            log(f"rimclf: no rim box near {e.id}, skipped")
            continue
        montage = build_montage(film(), e.t, box, m.width, m.height, rim_at=rim_follower(rims, e.rim_id, m.fps))
        store_path(image_id, e.t, box, e.rim_id)
        cls, suggested = _route(e, decisions.get(e.id))
        if cls is not None:
            store_class_image("rimclf", image_id, montage, cls)
            counts["from_review" if e.id in decisions else "made_auto"] += 1
        else:
            write_pending("rimclf", image_id, montage, [], extra={"t": e.t, "suggested": suggested, "event_id": e.id})
            counts["pending"] += 1
    rng = random.Random(seed)
    event_times = [e.t for e in events]
    tries = 0
    none_seen = 0
    while counts["none_auto"] + none_seen < n_none and tries < n_none * 20 and m.duration_s > 2 * margin_s:
        tries += 1
        t = rng.uniform(margin_s, m.duration_s - margin_s)
        if any(abs(t - et) < exclusion_s for et in event_times):
            continue
        box = rim_box_at(rims, int(round(t * m.fps)))
        if box is None:
            continue
        none_id = item_id(game_id, f"none-{int(round(t * 10)):06d}")
        if none_id in queued or find_class_images("rimclf", none_id):
            refresh_path(none_id, t, _nearest_rim_id(rims, int(round(t * m.fps))))
            none_seen += 1
            counts["already"] += 1
            continue
        rim_id = _nearest_rim_id(rims, int(round(t * m.fps)))
        store_path(none_id, t, box, rim_id)
        store_class_image("rimclf", none_id,
                          build_montage(film(), t, box, m.width, m.height, rim_at=rim_follower(rims, rim_id, m.fps)), "none")
        counts["none_auto"] += 1
    log(f"rimclf candidates for {game_id}: {counts}")
    ensure_views(game_id, log=log)
    return counts
