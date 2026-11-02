"""Rough courts on the low-camera games' queued pictures, for the coach to drag: marks read by eye on a seed picture, carried into the game's other pictures where they line up."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import film.train.datasets as ds
from film.export.court_sheet import draw_court
from film.interpret.court import fit_handles, template
from film.perceive import court_draft as cd
from film.perceive.court_refs import points_in_view
from film.perceive.register import features, match, project

out = Path(sys.argv[1])
SEEDS = {
    "2026-10-01-combs-vs-mdn-angle-1": (84.0, "010590", {"mid-left": [916, 666], "mid-right": [850, 1020], "near-ft-top": [200, 730], "far-ft-top": [1590, 790]},
                                        {"020881": {"near-lane-left": [670, 690], "near-lane-right": [560, 756], "near-ft-left": [1060, 704], "near-ft-right": [1260, 820]},
                                         "025781": {"far-ft-left": [600, 865], "far-lane-left": [1227, 758], "far-lane-right": [862, 721], "far-ft-top": [241, 852]}}),
    "2026-10-01-rkda-mdn-1-24": (94.0, "016441", {"mid-left": [1338, 353], "mid-right": [1403, 679], "mid-circle-left": [1349, 440], "mid-circle-right": [1381, 541], "near-ft-right": [340, 430], "near-ft-top": [660, 378]},
                                 {"009684": {"mid-left": [596, 345], "mid-right": [665, 622], "far-ft-top": [1270, 400], "near-ft-top": [40, 400]}}),
    "2026-10-08-new-arcadia-vs-cortez-angle-1": (84.0, "026043", {"far-ft-left": [655, 570], "far-ft-right": [655, 490], "far-ft-top": [417, 540], "mid-circle-right": [-171, 581]}, {}),
    "2026-10-08-new-marcos-de-niza-vs-cactus-shadows-angle-1": (84.0, "024267", {"mid-left": [960, 620], "mid-right": [1030, 960], "near-ft-top": [310, 730], "far-ft-top": [1630, 730]}, {}),
    "2026-10-08-new-marcos-de-niza-vs-tempe-angle-1": (84.0, "030351", {"near-ft-left": [480, 764], "near-ft-right": [340, 842], "near-ft-top": [650, 800], "mid-left": [1290, 716], "mid-right": [1260, 1000]}, {}),
    "2026-10-09-new-cactus-shadows-angle-1": (84.0, "028195", {"mid-left": [770, 600], "mid-right": [810, 894], "near-ft-top": [212, 700], "far-ft-top": [1286, 690]}, {}),
}
for i in ds.list_pending("court"):
    if i.startswith("2025-02-05-jv-vs-marcos-tactical_"):
        ds.remove_pending("court", i)        # the game has its court from Arcadia's references
for game, (length, seed, marks, fallback) in SEEDS.items():
    masks = cd._masks(game)
    sid = f"{game}_{seed}"
    fit = fit_handles([template(length)[n] for n in marks], [tuple(v) for v in marks.values()], [False] * len(marks), length=length)
    print(game[11:40], "seed", seed, "fit", None if fit is None else (fit.points_used, round(fit.error_px, 1)))
    if fit is None:
        continue
    src = ds.read_pending("court", sid)
    src.update(points=marks, length=length, draft={"by": "claude"})
    (ds.pending_dir("court") / f"{sid}.json").write_text(json.dumps(src, indent=1))
    from_pts = {**points_in_view(fit, src["width"], src["height"]), **{k: tuple(v) for k, v in marks.items()}}
    fa = features(cv2.imread(src["image"]), masks)
    tiles = []
    for i in ds.list_pending("court"):
        if not i.startswith(game + "_"):
            continue
        d = ds.read_pending("court", i)
        if i != sid:
            m = match(fa, features(cv2.imread(d["image"]), masks))
            new = None
            if m is not None:
                got = {n: list(p) for n, p in project(from_pts, m.H).items()}
                if fit_handles([template(length)[n] for n in got], [tuple(v) for v in got.values()], [False] * len(got), length=length) is not None:
                    new = got
            how = "carried" if new else "none"
            if new is None and i[-6:] in fallback:
                new, how = fallback[i[-6:]], "own"
            print("  ", i[-6:], how, None if m is None else m.inliers)
            if new:
                d.update(points=new, length=length, draft={"by": "claude"})
                (ds.pending_dir("court") / f"{i}.json").write_text(json.dumps(d, indent=1))
        d = ds.read_pending("court", i)
        img = cv2.imread(d["image"])
        pts = d.get("points") or {}
        fm = fit_handles([template(length)[n] for n in pts], [tuple(v) for v in pts.values()], [False] * len(pts), length=length) if pts else None
        if fm is not None:
            img = draw_court(img, fm, thickness=2)
        t = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{i[-6:]} {len(pts)} marks", (6, 22), 0, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / f"rough_{game}.jpg"), np.vstack([np.hstack(tiles[r:r + 2]) for r in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
