from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.events import load_events
from film.perceive.tracks import read_tracks
from film.train.datasets import store_class_image, write_pending
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


def build_rimclf_candidates(game_id: str, n_none: int = 40, seed: int = 0, margin_s: float = 30.0,
                            exclusion_s: float = EVENT_EXCLUSION_S, log=print) -> dict:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    source = Path(m.source)
    events = load_events(gd)
    rims = read_tracks(gd, "rim")
    counts = {"made_auto": 0, "pending": 0, "none_auto": 0}
    for e in events:
        box = rim_box_at(rims, int(round(e.t * m.fps)), e.rim_id)
        if box is None:
            continue
        montage = build_montage(source, e.t, box, m.width, m.height)
        if _auto_made(e):
            store_class_image("rimclf", e.id, montage, "made")
            counts["made_auto"] += 1
        else:
            suggested = "missed" if e.type == "shot" else "none"
            write_pending("rimclf", e.id, montage, [], extra={"t": e.t, "suggested": suggested, "event_id": e.id})
            counts["pending"] += 1
    rng = random.Random(seed)
    event_times = [e.t for e in events]
    tries = 0
    while counts["none_auto"] < n_none and tries < n_none * 20 and m.duration_s > 2 * margin_s:
        tries += 1
        t = rng.uniform(margin_s, m.duration_s - margin_s)
        if any(abs(t - et) < exclusion_s for et in event_times):
            continue
        box = rim_box_at(rims, int(round(t * m.fps)))
        if box is None:
            continue
        store_class_image("rimclf", f"none-{game_id}-{int(round(t * 10)):06d}", build_montage(source, t, box, m.width, m.height), "none")
        counts["none_auto"] += 1
    log(f"rimclf candidates for {game_id}: {counts}")
    return counts
