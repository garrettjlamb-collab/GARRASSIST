"""Every chosen shooter at the release, numbered, tiled by game, to count by eye how many are not players. Crops and the
numbering are kept so a rule can be scored against the count."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from film.common.manifest import Manifest
from film.perceive.shotspots import confirmed_shots
from film.train.prelabel import first_frame_at
from referee_stripes import stripe_score

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rows, tiles = [], []
for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
    gd = Path(f).parent
    m = Manifest.load(gd)
    if os.stat(m.source).st_blocks == 0:
        print("in iCloud, skipped", gd.name[:50])
        continue
    for e in confirmed_shots(gd):
        loc = e.signals.get("location") or {}
        if e.x is None or not loc.get("box"):
            continue
        frame = first_frame_at(Path(m.source), loc["release_t"], m.fps)
        if frame is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in loc["box"])
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            continue
        n = len(rows)
        rows.append({"n": n, "game": gd.name, "id": e.id, "x": e.x, "y": e.y, "conf": e.location_conf, "box": loc["box"], "release_t": loc["release_t"],
                     "stripes": stripe_score(frame, loc["box"])})
        cv2.imwrite(str(out / f"{n:04d}.jpg"), crop)
        t = np.full((158, 72, 3), 30, np.uint8)
        t[:144] = cv2.resize(crop, (72, 144), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(n), (2, 155), 0, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    print(gd.name[:40], len(rows), flush=True)
(out / "rows.json").write_text(json.dumps(rows))
cols, lines = 25, 6
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
per = cols * lines
for s in range(0, len(tiles), per):
    part = tiles[s:s + per]
    while len(part) % cols:
        part.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / f"sheet{s // per}.jpg"), np.vstack([np.hstack(part[i:i + cols]) for i in range(0, len(part), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(rows), "crops,", len(tiles) // per + 1, "sheets")
