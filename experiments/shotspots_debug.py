"""One picture per shot: the frame where the ball is about to reach the rim, every ball sighting carried into it with
time as colour (blue early, red late) and its seconds-before-arrival, and the on-court people at the arrival."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.video import iter_frames
from film.interpret.location import Look, ball_sightings, find_flight
from shotspots_score import _rims_in_anchor
from film.perceive.court_run import load_mappings, mapping_at


def main(game_id: str, cache: Path, out_dir: Path, ids: list[str]) -> None:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in json.loads(cache.read_text())}
    for i in ids:
        r = rows[i]
        looks = [Look(f["t"], mapping_at(maps, f["t"]), [tuple(b) for b in f["people"]], [tuple(b) for b in f["ball"]]) for f in r["frames"]]
        frame = None
        for t, f in iter_frames(Path(m.source), 10.0, start=r["t"], end=r["t"] + 0.5):
            frame = f
            at = t
            break
        anchor = next(k.mapping for k in reversed(looks) if k.mapping is not None)
        # find_arc works in the last mapped frame's pixels; this draws in them too (the frame is near enough)
        s = ball_sightings(looks)
        arc = {(a[0], a[1]) for a in find_flight(s, _rims_in_anchor(r, looks), end_t=r["t"] if r["rim_id"] >= 0 else None)}
        t0, t1 = looks[0].t, looks[-1].t
        for t, x, y, conf in s:
            u = (t - t0) / max(t1 - t0, 1e-6)
            colour = (int(255 * (1 - u)), 60, int(255 * u))
            cv2.circle(frame, (int(x), int(y)), 9 if (t, x) in arc else 5, colour, 2)
            cv2.putText(frame, f"{t - r['t']:.1f}", (int(x) + 8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        last = min(looks, key=lambda k: abs(k.t - at))
        for b in last.people:
            if last.mapping is not None and last.mapping.contains((b[0] + b[2]) / 2, b[3], margin=3.0):
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
        cv2.putText(frame, f"{i} pts {r['points']} made {r['made']} src {r['source']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{i}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(out_dir / f"{i}.jpg", len(s), len(arc))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
