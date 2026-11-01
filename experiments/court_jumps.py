"""Where a game's fitted court jumps: centre court's pixel between neighbouring fits. A camera turns smoothly, so a big jump
in a second is a frame lined up wrongly. Also how many confirmed shots sit within 3 s of one."""
import glob
from pathlib import Path

import numpy as np

from film.perceive.court_run import load_mappings
from film.perceive.shotspots import confirmed_shots

for f in sorted(glob.glob("games/*/court.parquet")):
    gd = Path(f).parent
    maps = load_mappings(gd)
    if len(maps) < 3:
        print(gd.name[:40], "no court")
        continue
    ts, pts = [], []
    for t, m in maps:
        v = np.linalg.inv(m.H) @ np.array([25.0, m.length / 2, 1.0])
        ts.append(t)
        pts.append((v[0] / v[2], v[1] / v[2]))
    ts, pts = np.array(ts), np.array(pts)
    speed = np.hypot(*np.diff(pts, axis=0).T) / np.maximum(np.diff(ts), 0.1)
    bad = np.flatnonzero(speed > 1500)
    shots = [e.t for e in confirmed_shots(gd)]
    near = sum(any(abs(s - ts[i]) < 3 for i in bad) for s in shots)
    print(f"{gd.name.replace('2026-10-09-new-marcos-de-niza-high-school-vs-', '')[:34]:34s} fits {len(maps):6d} median {np.median(speed):6.0f} px/s  jumps {len(bad):4d} ({100 * len(bad) / len(speed):.1f}%)  shots near a jump {near}/{len(shots)}")
