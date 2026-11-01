"""After the dot check: the score with flagged shots left out, whether confidence tells good dots from bad, a strip of the clips the
coach flagged, and each big miss at the machine's release with its shooter in red and his dot in green."""
import json
import math
import sys
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.export.clips import clip_path
from film.interpret.court import WIDTH
from film.interpret.events import load_events
from film.interpret.zones import is_three, zone_of
from film.perceive.court_run import load_mappings, mapping_between
from film.review.dotcheck import read_answers

FLAGGED = {12, 22}
out = Path(sys.argv[1])
rows = read_answers()
for i, r in enumerate(rows, 1):
    r["n"], r["err"] = i, math.hypot(r["x"] - r["machine_x"], r["y"] - r["machine_y"])


def line(name, g):
    e = [r["err"] for r in g]
    if not e:
        return
    print(f"{name:30s} n{len(g):3d} median {median(e):5.1f} ft | within 3 {sum(v <= 3 for v in e) / len(e):.0%} within 6 {sum(v <= 6 for v in e) / len(e):.0%} over 10 {sum(v > 10 for v in e) / len(e):.0%} | arc side {sum(is_three(r['x'], r['y']) == is_three(r['machine_x'], r['machine_y']) for r in g) / len(g):.0%} zone {sum(zone_of(r['x'], r['y']) == zone_of(r['machine_x'], r['machine_y']) for r in g) / len(g):.0%}")


clean = [r for r in rows if r["n"] not in FLAGGED]
line("all 30", rows)
line("without clips 12 and 22", clean)
for p in sorted({r["profile"] for r in clean}):
    line("camera " + p, [r for r in clean if r["profile"] == p])
for m in ("flight", "path"):
    line("method " + m, [r for r in clean if r["method"] == m])
line("conf 0.8 and up", [r for r in clean if (r["conf"] or 0) >= 0.8])
line("conf under 0.8", [r for r in clean if (r["conf"] or 0) < 0.8])
line("path-disagrees", [r for r in clean if "path-disagrees" in r["flags"]])
line("not path-disagrees", [r for r in clean if "path-disagrees" not in r["flags"]])
line("his three", [r for r in clean if is_three(r["x"], r["y"])])
line("his two", [r for r in clean if not is_three(r["x"], r["y"])])
print("three, his vs machine:", sum(is_three(r["x"], r["y"]) for r in clean), sum(is_three(r["machine_x"], r["machine_y"]) for r in clean))


def frame_at(gd, e, t):
    cap = cv2.VideoCapture(str(clip_path(gd, e.id)))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - e.clip["start"]) * 1000)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


# Strips of the two flagged clips.
tiles = []
for r in rows:
    if r["n"] not in FLAGGED:
        continue
    gd = game_dir(r["game_id"])
    e = next(x for x in load_events(gd) if x.id == r["event_id"])
    for s in np.arange(0, e.clip["end"] - e.clip["start"], 1.0):
        f = frame_at(gd, e, e.clip["start"] + s)
        if f is None:
            continue
        f = cv2.resize(f, (480, 270))
        cv2.putText(f, f"clip {r['n']} {e.clip['start'] + s - e.t:+.0f}s from card's shot", (6, 20), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(f)
cols = 4
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(out / "flagged.jpg"), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 80])

# Big misses.
tiles = []
for r in sorted([r for r in clean if r["err"] > 8], key=lambda r: -r["err"]):
    gd = game_dir(r["game_id"])
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    e = next(x for x in load_events(gd) if x.id == r["event_id"])
    loc = e.signals["location"]
    f = frame_at(gd, e, loc["release_t"])
    mp = mapping_between(maps, loc["release_t"])
    if f is None or mp is None:
        continue
    sx, sy = f.shape[1] / m.width, f.shape[0] / m.height
    x1, y1, x2, y2 = loc["box"]
    cv2.rectangle(f, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 0, 255), 3)
    feet = mp.to_court((x1 + x2) / 2, y2)
    far = feet is not None and feet[1] > mp.length / 2
    hx, hy = (WIDTH - r["x"], mp.length - r["y"]) if far else (r["x"], r["y"])
    v = np.linalg.inv(mp.H) @ np.array([hx, hy, 1.0])
    if v[2] > 0:
        cv2.drawMarker(f, (int(v[0] / v[2] * sx), int(v[1] / v[2] * sy)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 34, 4)
    f = cv2.resize(f, (640, 360))
    cv2.rectangle(f, (0, 0), (640, 24), (0, 0, 0), -1)
    cv2.putText(f, f"#{r['n']} err {r['err']:.0f}ft his({r['x']:.0f},{r['y']:.0f}) mach({r['machine_x']:.0f},{r['machine_y']:.0f}) {r['method']} {e.signals.get('points') if e.made else 'miss'} rel {loc['release_t'] - e.t:.1f}", (4, 17), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(f)
cols = 3
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(out / "misses.jpg"), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
