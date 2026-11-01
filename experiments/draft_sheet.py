"""All of a game's queued court pictures with their drafts drawn, tiled into one picture to check by eye."""
import sys
from pathlib import Path

import cv2
import numpy as np

import film.train.datasets as ds
from film.export.court_sheet import draw_court
from film.perceive import court_draft as cd

game, out = sys.argv[1], Path(sys.argv[2])
tiles = []
for i in ds.list_pending("court"):
    if not i.startswith(game + "_"):
        continue
    d = ds.read_pending("court", i)
    img = cv2.imread(d["image"])
    h = [{"name": n, "px": p, "pinned": False} for n, p in (d.get("points") or {}).items()]
    m = cd.fit_of(h, d.get("length") or 94.0) if h else None
    if m is not None:
        img = draw_court(img, m, thickness=2)
    t = cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])))
    cv2.putText(t, f"{i[-6:]} {(d.get('draft') or {}).get('by')} {len(h)} marks", (8, 24), 0, 0.7, (0, 255, 255), 2)
    tiles.append(t)
while len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(out), np.vstack([np.hstack(tiles[r:r + 2]) for r in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(out, len(tiles))
