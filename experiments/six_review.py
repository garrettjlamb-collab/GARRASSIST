"""What he did with the six games' pictures, in the order he saw them, and each game's references against each other."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import film.train.datasets as ds
from film.common.paths import datasets_dir, labels_dir
from film.export.court_sheet import draw_court
from film.interpret.court import fit_handles, template
from film.perceive.court_refs import check_references, load_references, measure_length, reference_ids
from film.perceive.court_draft import _masks

out = Path(sys.argv[1])
GAMES = ["2026-10-01-combs-vs-mdn-angle-1", "2026-10-01-rkda-mdn-1-24", "2026-10-08-new-arcadia-vs-cortez-angle-1", "2026-10-08-new-marcos-de-niza-vs-cactus-shadows-angle-1", "2026-10-08-new-marcos-de-niza-vs-tempe-angle-1", "2026-10-09-new-cactus-shadows-angle-1"]
log = [json.loads(l) for l in (labels_dir() / "court" / "review_log.jsonl").read_text().splitlines()]
rows = [r for r in log if any(r["image_id"].startswith(g + "_") for g in GAMES)]
print("logged", len(rows), "pending", ds.list_pending("court"))
skipped = (labels_dir() / "court" / "skipped.txt").read_text().split()
print("skipped now", [s[-40:] for s in skipped if any(s.startswith(g + "_") for g in GAMES)])
for n, r in enumerate(sorted(rows, key=lambda r: r["at"]), 1):
    print(f"{n:2d} {r['image_id'][11:]:62.62s} by {r['by']:8s} len {r['length']:.0f} pulled {r['pulled']:2d} moved {r['moved_px']} sec {round(r['seconds'] or 0)} pts {r['points']}")
print("total min", round(sum(r["seconds"] or 0 for r in rows) / 60, 1))
for g in GAMES:
    masks = _masks(g)
    refs = load_references(g, masks)
    print(g[11:45], "refs", len(refs), "measured length", measure_length(refs))
    for c in check_references(g, masks):
        print("   ", c["image_id"][-6:], "pts", c["points"], "shared", c["shared"], "median", c["median_px"], "worst", c["worst"], c["max_px"])
    tiles = []
    for r in refs:
        got = ds.read_points("court", r.image_id)
        img = cv2.imread(str(got[0]))
        lens = [e["length"] for e in rows if e["image_id"] == r.image_id]
        L = lens[-1] if lens else 94.0
        names = list(got[1])
        m = fit_handles([template(L)[n] for n in names], [got[1][n] for n in names], [False] * len(names), length=L)
        if m is not None:
            img = draw_court(img, m, thickness=2)
        t = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{r.image_id[-6:]} len {L:.0f}", (6, 24), 0, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"approved_{g}.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
