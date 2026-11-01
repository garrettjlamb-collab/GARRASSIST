"""What setting non-players aside did: the scoreboard score before and after, and, on the games the classifier did not train
on, every shooter chosen now, tiled, with each changed shot's old and new choice side by side."""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from film.common.cloud import game_film, local_source
from film.common.manifest import Manifest
from film.interpret.events import Event, load_events
from film.interpret.zones import is_three
from film.perceive.shotspots import confirmed_shots
from film.train.prelabel import first_frame_at

TRAINED = {p.stem.split("--")[0] for p in Path("datasets/official/player").glob("*.jpg")}
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def side(events):
    made = [e for e in events if e.x is not None and e.made and e.signals.get("points") in (2, 3) and e.signals["location"]["method"] == "flight"]
    p3 = [e for e in made if e.signals["points"] == 3]
    return sum(is_three(e.x, e.y) == (e.signals["points"] == 3) for e in made), len(made), sum(is_three(e.x, e.y) for e in p3), len(p3)


tot = Counter()
changed_tiles, held_tiles = [], []


def tile(frame, box, label, colour=(255, 255, 255)):
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    t = np.full((158, 72, 3), 30, np.uint8)
    if crop.size:
        t[:144] = cv2.resize(crop, (72, 144), interpolation=cv2.INTER_AREA)
    cv2.putText(t, label, (2, 155), 0, 0.38, colour, 1, cv2.LINE_AA)
    return t


for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
    gd = Path(f).parent
    m = Manifest.load(gd)
    now = {e.id: e for e in confirmed_shots(gd)}
    before = {d["id"]: Event.from_json(d) for d in json.loads((gd / "events.before-official.json").read_text()) if d["id"] in now}
    a, b = side(list(before.values())), side(list(now.values()))
    changed = [i for i, e in now.items() if (e.signals.get("location") or {}).get("box") != (before[i].signals.get("location") or {}).get("box")]
    lost = [i for i in changed if now[i].x is None]
    print(f"{gd.name.replace('2026-10-09-new-marcos-de-niza-high-school-vs-', '')[:24]:24s} {'held out' if gd.name not in TRAINED else 'trained '} shots {len(now):3d} changed {len(changed):2d} lost {len(lost)} side {a[0]}/{a[1]} -> {b[0]}/{b[1]} threes {a[2]}/{a[3]} -> {b[2]}/{b[3]}")
    for k, v in zip(("a", "an", "a3", "a3n", "b", "bn", "b3", "b3n"), (*a, *b)):
        tot[k] += v
    tot["changed"] += len(changed)
    tot["lost"] += len(lost)
    tot["shots"] += len(now)
    held = gd.name not in TRAINED
    if not held and not changed:
        continue
    with local_source(game_film(gd.name), log=lambda s: None):
        for i, e in now.items():
            is_changed = i in changed
            if not (is_changed or held):
                continue
            old, new = before[i].signals.get("location") or {}, e.signals.get("location") or {}
            if is_changed:
                for loc, c in ((old, (0, 0, 255)), (new, (0, 255, 0))):
                    if loc.get("box"):
                        fr = first_frame_at(Path(m.source), loc["release_t"], m.fps)
                        if fr is not None:
                            changed_tiles.append(tile(fr, loc["box"], f"{len(changed_tiles)}", c))
                    else:
                        changed_tiles.append(np.full((158, 72, 3), 30, np.uint8))
            if held and new.get("box"):
                fr = first_frame_at(Path(m.source), new["release_t"], m.fps)
                if fr is not None:
                    held_tiles.append(tile(fr, new["box"], f"{len(held_tiles)}"))
print(dict(tot))


def sheets(tiles, name, cols=25, lines=6):
    per = cols * lines
    for s in range(0, len(tiles), per):
        part = tiles[s:s + per]
        while len(part) % cols:
            part.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"{name}{s // per}.jpg"), np.vstack([np.hstack(part[i:i + cols]) for i in range(0, len(part), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(name, len(tiles))


sheets(changed_tiles, "changed", cols=24)
sheets(held_tiles, "held")
