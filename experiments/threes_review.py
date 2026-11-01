"""Every made three: right or wrong side of the arc, how the flight looks, and for the wrong ones a picture at the chosen
release with the flight drawn, numbered, tiled. Also the same numbers for made twos, to compare."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from film.common.cloud import game_film, local_source
from film.common.manifest import Manifest
from film.export.court_sheet import draw_court
from film.interpret.court import BASKET, fold
from film.interpret.location import Look, ball_sightings, find_flight
from film.interpret.zones import is_three
from film.perceive.court_run import load_mappings, mapping_at
from film.perceive.shotspots import NO_RIM, _rims_in_anchor, confirmed_shots, read_looks
from film.train.prelabel import first_frame_at

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
tiles, table = [], []
for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
    gd = Path(f).parent
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    rows = {r["id"]: r for r in read_looks(gd)}
    shots = [e for e in confirmed_shots(gd) if e.made and e.signals.get("points") in (2, 3) and e.x is not None and e.signals["location"]["method"] == "flight"]
    wrong3 = [e for e in shots if e.signals["points"] == 3 and not is_three(e.x, e.y)]
    for e in shots:
        loc = e.signals["location"]
        r = rows[e.id]
        table.append({"game": gd.name[-60:-40], "id": e.id, "pts": e.signals["points"], "ok": is_three(e.x, e.y) == (e.signals["points"] == 3),
                      "dist": round(float(np.hypot(e.x - BASKET[0], e.y - BASKET[1])), 1), "before": round(e.t - loc["release_t"], 2), "n": loc["flight_points"], "reach": loc["reach_px"],
                      "sb": r["rim_id"] < NO_RIM, "conf": e.location_conf, "aside": "official-set-aside" in e.flags})
    if not wrong3:
        continue
    scale = m.width / 1920
    with local_source(game_film(gd.name), log=lambda s: None):
        for e in wrong3:
            r, loc = rows[e.id], e.signals["location"]
            looks = [Look(x["t"], mapping_at(maps, x["t"]), [tuple(b) for b in x["people"]], [tuple(b) for b in x["ball"]]) for x in r["frames"]]
            anchor = next(k.mapping for k in reversed(looks) if k.mapping is not None)
            s = ball_sightings(looks, scale)
            flight = find_flight(s, _rims_in_anchor(r, looks), e.t if r["rim_id"] >= NO_RIM else None, scale)
            on = {(a[0], a[1]) for a in flight}
            t = loc["release_t"]
            frame = first_frame_at(Path(m.source), t, m.fps)
            look = min(looks, key=lambda k: abs(k.t - t))
            frame = draw_court(frame, look.mapping, thickness=1)
            to_here = np.linalg.inv(look.mapping.H) @ anchor.H
            for tt, x, y, c in s:
                v = to_here @ np.array([x, y, 1.0])
                px, py = int(v[0] / v[2]), int(v[1] / v[2])
                u = (tt - looks[0].t) / max(looks[-1].t - looks[0].t, 1e-6)
                col = (int(255 * (1 - u)), 80, int(255 * u))
                cv2.circle(frame, (px, py), 12 if (tt, x) in on else 4, col, 3 if (tt, x) in on else 1)
                if (tt, x) in on:
                    cv2.putText(frame, f"{tt - e.t:.1f}", (px + 14, py + 5), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            x1, y1, x2, y2 = (int(v) for v in loc["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            n = len(tiles)
            frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (960, 30), (0, 0, 0), -1)
            cv2.putText(frame, f"#{n} {gd.name[-55:-38]} {e.id} dot({e.x:.0f},{e.y:.0f}) rel{t - e.t:.1f} {loc['flight_points']}pts reach{loc['reach_px']}", (6, 21), 0, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(frame)
(out / "table.json").write_text(json.dumps(table))
for s in range(0, len(tiles), 6):
    part = tiles[s:s + 6]
    while len(part) % 2:
        part.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / f"wrong{s // 6}.jpg"), np.vstack([np.hstack(part[i:i + 2]) for i in range(0, len(part), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(tiles), "wrong threes")
for pts in (2, 3):
    for ok in (True, False):
        g = [r for r in table if r["pts"] == pts and r["ok"] == ok]
        if g:
            print(f"made {pts}s {'right' if ok else 'WRONG'} n={len(g)} median distance {np.median([r['dist'] for r in g]):.1f} ft; release before rim {np.median([r['before'] for r in g]):.2f} s; flight points {np.median([r['n'] for r in g]):.0f}; share from scoreboard-only cards {np.mean([r['sb'] for r in g]):.0%}; reach>0 {np.mean([(r['reach'] or 0) > 0 for r in g]):.0%}")
print("wrong threes:", [(r["dist"], r["before"], r["n"], r["reach"], r["sb"]) for r in table if r["pts"] == 3 and not r["ok"]])
print("dist right threes:", sorted(r["dist"] for r in table if r["pts"] == 3 and r["ok"]))
print("wrong twos:", [(r["dist"], r["before"], r["n"], r["sb"]) for r in table if r["pts"] == 2 and not r["ok"]])
