# film/perceive/court_run.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.video import iter_frames
from film.interpret.court import Mapping
from film.perceive.courtfinder import CourtFinder

COLUMNS = ["frame", "t", "end", "points_used", "error_px", "conf"] + [f"h{i}{j}" for i in range(3) for j in range(3)]
FILE = "court.parquet"
MAX_GAP_S = 2.0


def default_step(profile: str) -> float:
    return 1.0 if profile.startswith("hudl") else 0.5


def court_pass(game_id: str, finder=None, device: str | None = None, step_s: float | None = None, log=print) -> dict:
    """Fit the court mapping every step_s seconds and write court.parquet."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    finder = finder or CourtFinder.load(pick_device(device))
    if finder is None:
        raise RuntimeError("no court model trained yet: label frames in the Label court tab, then `film train court`")
    step = step_s or default_step(m.profile)
    rows, n, started = [], 0, time.time()
    for t, frame in iter_frames(Path(m.source), fps=1.0 / step):
        n += 1
        mapping = finder.mapping(frame)
        if mapping is None:
            continue
        rows.append({"frame": int(round(t * m.fps)), "t": float(t), "end": mapping.end, "points_used": mapping.points_used,
                     "error_px": mapping.error_px, "conf": mapping.conf, **{f"h{i}{j}": float(mapping.H[i, j]) for i in range(3) for j in range(3)}})
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(gd / FILE, index=False)
    m.model_versions["court"] = finder.version
    m.mark("court", "done", frames=n, fitted=len(rows), step_s=step, seconds=round(time.time() - started, 1))
    m.save(gd)
    log(f"court {game_id}: {len(rows)} of {n} frames fitted ({step} s apart) with {finder.version}")
    return m.stages["court"]


def load_mappings(gd: Path) -> list[tuple[float, Mapping]]:
    path = Path(gd) / FILE
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    out = []
    for r in df.itertuples():
        H = np.array([[getattr(r, f"h{i}{j}") for j in range(3)] for i in range(3)])
        out.append((float(r.t), Mapping(H=H, points_used=int(r.points_used), error_px=float(r.error_px), conf=float(r.conf), end=str(r.end))))
    return sorted(out, key=lambda p: p[0])


def mapping_at(mappings: list[tuple[float, Mapping]], t: float, max_gap_s: float = MAX_GAP_S) -> Mapping | None:
    """The fitted mapping nearest in time, if within max_gap_s."""
    if not mappings:
        return None
    best = min(mappings, key=lambda p: abs(p[0] - t))
    return best[1] if abs(best[0] - t) <= max_gap_s else None
