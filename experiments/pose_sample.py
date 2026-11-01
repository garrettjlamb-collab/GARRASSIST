"""A random sample of shots where the dot moved over 5 ft when poses came in: the old choice in red, the new in green, at the new release."""
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from film.common.cloud import game_film, local_source
from film.common.manifest import Manifest
from film.interpret.events import Event
from film.perceive.shotspots import confirmed_shots
from film.train.prelabel import first_frame_at

N, out = int(sys.argv[1]), sys.argv[2]
cases = []
for f in sorted(glob.glob("games/*/shot_poses.json.gz")):
    gd = Path(f).parent
    old = {d["id"]: Event.from_json(d) for d in json.loads((gd / "events.before-pose.json").read_text())}
    for e in confirmed_shots(gd):
        o = old.get(e.id)
        loc = e.signals.get("location") or {}
        if o is None or e.x is None or o.x is None or loc.get("method") != "pose":
            continue
        if np.hypot(e.x - o.x, e.y - o.y) > 5:
            cases.append((gd.name, e, o))
print("moved over 5 ft:", len(cases))
random.Random(11).shuffle(cases)
tiles = []
for name, e, o in sorted(cases[:N], key=lambda c: c[0]):
    m = Manifest.load(Path("games") / name)
    loc, ol = e.signals["location"], o.signals.get("location") or {}
    with local_source(game_film(name), log=lambda s: None):
        frame = first_frame_at(Path(m.source), loc["release_t"], m.fps)
    if frame is None:
        continue
    x1, y1, x2, y2 = (int(v) for v in loc["box"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    if ol.get("box"):
        a, b, c, d = (int(v) for v in ol["box"])
        cv2.rectangle(frame, (a, b), (c, d), (0, 0, 255), 3)
    s = m.width / 1920
    cx, cy, hw, hh = (x1 + x2) // 2, (y1 + y2) // 2, int(533 * s), int(300 * s)
    crop = frame[max(cy - hh, 0):cy + hh, max(cx - hw, 0):cx + hw]
    crop = cv2.resize(crop, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.rectangle(crop, (0, 0), (640, 22), (0, 0, 0), -1)
    cv2.putText(crop, f"#{len(tiles)} {name[11:24]} {e.id[5:]} s{loc.get('pose_score', '')} new({e.x:.0f},{e.y:.0f}) old({o.x:.0f},{o.y:.0f}) {e.signals.get('points') if e.made else 'miss'}", (3, 15), 0, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(crop)
cols = 3
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
for j in range(0, len(tiles), 15):
    part = tiles[j:j + 15]
    cv2.imwrite(f"{out}{j // 15}.jpg", np.vstack([np.hstack(part[i:i + cols]) for i in range(0, len(part), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
print(len(tiles), "drawn")
