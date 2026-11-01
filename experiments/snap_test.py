"""Does snapping recover a court? Approved pictures (his, and Claude's checked ones) knocked out of place by a known amount, then snapped: how far from the approved court it lands."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from film.common.paths import datasets_dir
from film.interpret.court import fit_handles, template
from film.perceive.court_refs import points_in_view
from court_snap import snap
from film.train import datasets as ds

CORNERS = [(5.0, 5.0), (45.0, 5.0), (45.0, 40.0), (5.0, 40.0)]
rng = np.random.default_rng(5)
rows = []
for img in sorted((datasets_dir() / "court" / "images").glob("*/*.jpg")):
    got = ds.read_points("court", img.stem)
    if got is None or len(got[1]) < 6:
        continue
    path, pts, w, h = got
    image = cv2.imread(str(path))
    for length in (94.0, 84.0):
        T = template(length)
        names = list(pts)
        true = fit_handles([T[n] for n in names], [pts[n] for n in names], [False] * len(names), length=length)
        if true is not None and true.error_px < 6:
            break
    if true is None:
        continue
    in_view = list(points_in_view(true, w, h).items())
    if len(in_view) < 5:
        continue
    far = np.array([p for _, p in in_view])
    centre = far.mean(0)
    # Knock it: move each of 4 spread template points by up to 40 px at 1920.
    idx = np.argsort(np.arctan2(far[:, 1] - centre[1], far[:, 0] - centre[0]))[np.linspace(0, len(far) - 1, 4).astype(int)]
    corners = [T[in_view[i][0]] for i in idx]
    if len({c for c in corners}) < 4:
        continue
    k = 40.0 * w / 1920
    moved = far[idx] + rng.uniform(-k, k, (4, 2))
    rough = fit_handles(corners, [tuple(m) for m in moved], [False] * 4, length=length)
    if rough is None:
        continue
    res = snap(image, rough, corners)
    fixed = fit_handles(corners, [tuple(p) for p in res["pixels"]], [False] * 4, length=length)
    if fixed is None:
        continue

    def off(m):
        back, tb = np.linalg.inv(m.H), np.linalg.inv(true.H)
        d = []
        for _, p in in_view:
            a = np.array([*true.to_court(*p), 1.0])
            v = back @ a
            d.append(np.hypot(v[0] / v[2] - p[0], v[1] / v[2] - p[1]))
        return float(np.median(d)) * 1920 / w

    rows.append((img.stem[:30], img.stem[-6:], round(off(rough), 1), round(off(fixed), 1), res["before"], res["after"]))
    print(rows[-1], flush=True)
if rows:
    print("median px off: rough", np.median([r[2] for r in rows]), "snapped", np.median([r[3] for r in rows]), "better", sum(r[3] < r[2] for r in rows), "of", len(rows))
