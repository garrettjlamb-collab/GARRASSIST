from __future__ import annotations

import random
from pathlib import Path

import cv2
import pandas as pd

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.events import load_events
from film.perceive.tracks import read_tracks
from film.review.decisions import load_decisions
from film.train.datasets import (
    find_class_images, list_pending, pending_images_dir, remove_pending, store_class_image, write_pending,
)
from film.train.montage import build_montage

EVENT_EXCLUSION_S = 3.0


def rim_box_at(rims: pd.DataFrame, frame: int, rim_id: int | None = None, max_gap: int = 10) -> tuple | None:
    """The rim box at the nearest tracked frame within max_gap frames, for a given rim id or any rim."""
    if rims.empty:
        return None
    df = rims if rim_id is None else rims[rims["track_id"] == rim_id]
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


def build_rimclf_candidates(game_id: str, n_none: int = 40, seed: int = 0, margin_s: float = 30.0,
                            exclusion_s: float = EVENT_EXCLUSION_S, log=print) -> dict:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    source = Path(m.source)
    events = load_events(gd)
    rims = read_tracks(gd, "rim")
    decisions = load_decisions(gd)
    counts = {"made_auto": 0, "pending": 0, "none_auto": 0, "already": 0, "no_rim": 0, "from_review": 0}
    queued = set(list_pending("rimclf"))
    for e in events:
        if e.id in queued and e.id in decisions:
            # Queued by an earlier run; if the coach has since given a class, promote the queued montage.
            cls, _ = _route(e, decisions[e.id])
            image = cv2.imread(str(pending_images_dir("rimclf") / f"{e.id}.jpg")) if cls is not None else None
            if image is not None:
                store_class_image("rimclf", e.id, image, cls)
                remove_pending("rimclf", e.id)
                counts["from_review"] += 1
                continue
        if e.id in queued or find_class_images("rimclf", e.id):
            counts["already"] += 1
            continue
        box = None if e.rim_id < 0 else rim_box_at(rims, int(round(e.t * m.fps)), e.rim_id)
        if box is None:
            counts["no_rim"] += 1
            log(f"rimclf: no rim box near {e.id}, skipped")
            continue
        montage = build_montage(source, e.t, box, m.width, m.height)
        cls, suggested = _route(e, decisions.get(e.id))
        if cls is not None:
            store_class_image("rimclf", e.id, montage, cls)
            counts["from_review" if e.id in decisions else "made_auto"] += 1
        else:
            write_pending("rimclf", e.id, montage, [], extra={"t": e.t, "suggested": suggested, "event_id": e.id})
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
        none_id = f"none-{game_id}-{int(round(t * 10)):06d}"
        if none_id in queued or find_class_images("rimclf", none_id):
            none_seen += 1
            counts["already"] += 1
            continue
        store_class_image("rimclf", none_id, build_montage(source, t, box, m.width, m.height), "none")
        counts["none_auto"] += 1
    log(f"rimclf candidates for {game_id}: {counts}")
    return counts
