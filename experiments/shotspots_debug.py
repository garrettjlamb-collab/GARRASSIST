"""One picture per shot, at the chosen release: the flight carried into that frame (blue early, red late, seconds before
the rim), the on-court people in green and the chosen shooter in red."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.events import load_events
from film.interpret.location import Look, ball_sightings, find_flight
from film.perceive.court_run import load_mappings, mapping_at
from film.perceive.shotspots import NO_RIM, _rims_in_anchor, read_looks
from film.train.prelabel import first_frame_at


def main(game_id: str, out_dir: Path, ids: list[str]) -> None:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in read_looks(gd)}
    events = {e.id: e for e in load_events(gd)}
    scale = m.width / 1920
    for i in ids:
        r, e = rows[i], events[i]
        looks = [Look(f["t"], mapping_at(maps, f["t"]), [tuple(b) for b in f["people"]], [tuple(b) for b in f["ball"]]) for f in r["frames"]]
        anchor = next(k.mapping for k in reversed(looks) if k.mapping is not None)
        loc = e.signals.get("location", {})
        rel = loc.get("release_t") or e.t - 1.0
        frame = first_frame_at(Path(m.source), rel, m.fps)
        look = min(looks, key=lambda k: abs(k.t - rel))
        s = ball_sightings(looks, scale)
        flight = {(a[0], a[1]) for a in find_flight(s, _rims_in_anchor(r, looks), e.t if r["rim_id"] >= NO_RIM else None, scale)}
        to_here = np.linalg.inv(look.mapping.H) @ anchor.H
        for t, x, y, conf in s:
            v = to_here @ np.array([x, y, 1.0])
            px, py = int(v[0] / v[2]), int(v[1] / v[2])
            u = (t - looks[0].t) / max(looks[-1].t - looks[0].t, 1e-6)
            on = (t, x) in flight
            cv2.circle(frame, (px, py), 10 if on else 4, (int(255 * (1 - u)), 60, int(255 * u)), 2 if on else 1)
            if on:
                cv2.putText(frame, f"{t - e.t:.1f}", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for b in look.people:
            if look.mapping.contains((b[0] + b[2]) / 2, b[3], margin=3.0):
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
        if loc.get("box"):
            x1, y1, x2, y2 = (int(v) for v in loc["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"{i} pts {e.signals.get('points')} made {e.made} dot ({e.x},{e.y}) rel {rel - e.t:.1f} {loc.get('flight_points')}pts", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{i}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        print(out_dir / f"{i}.jpg")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
