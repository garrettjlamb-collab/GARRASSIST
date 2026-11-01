"""Experiment: what the person and ball detectors see in the seconds before a shot reaches the rim, cached per shot
so the release and shooter rules can be tried without running the models again."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.registry import base_weights_path, latest
from film.common.video import iter_frames
from film.interpret.events import load_events
from film.perceive.detectors import YoloDetector
from film.perceive.tracks import read_tracks
from film.review.decisions import load_decisions

BEFORE_S, AFTER_S, FPS = 3.0, 0.3, 10.0
BALL_IMGSZ = 1920


def run(game_id: str, n: int, out: Path) -> None:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    decisions = load_decisions(gd)
    shots = [e for e in load_events(gd) if (decisions.get(e.id) or {}).get("status") == "confirmed" and e.signals.get("phase") != "warmup"]
    step = max(1, len(shots) // n)
    shots = shots[::step][:n]
    people = YoloDetector.from_weights(base_weights_path("yolo11n.pt"), "mps", 960, conf=0.2, name="person")
    version, folder = latest("ball")
    ball = YoloDetector.from_weights(folder / "best.pt", "mps", BALL_IMGSZ, conf=0.1, name=f"ball@{version}")
    rims = read_tracks(gd, "rim")
    rows, start = [], time.time()
    for e in shots:
        frames = []
        for t, frame in iter_frames(Path(m.source), FPS, start=max(0.0, e.t - BEFORE_S), end=e.t + AFTER_S):
            p = [b for b in people.detect(frame) if b.cls == "person"]
            b = ball.detect(frame)
            frames.append({"t": t, "people": [[x.x1, x.y1, x.x2, x.y2, x.conf] for x in p],
                           "ball": [[x.x1, x.y1, x.x2, x.y2, x.conf] for x in b if x.cls == "ball"]})
        near = rims[(rims.t >= e.t - BEFORE_S) & (rims.t <= e.t + AFTER_S)]
        rim_boxes = [[float(r.t), int(r.track_id), r.x1, r.y1, r.x2, r.y2] for r in near.itertuples()]
        rows.append({"rims": rim_boxes, "id": e.id, "t": e.t, "rim_id": e.rim_id, "points": e.signals.get("points"), "made": (decisions[e.id]).get("made", e.made),
                     "source": e.signals.get("source"), "frames": frames})
        print(e.id, e.signals.get("points"), "ball frames", sum(bool(f["ball"]) for f in frames), "of", len(frames),
              "people", round(sum(len(f["people"]) for f in frames) / max(1, len(frames)), 1), f"{time.time() - start:.0f}s", flush=True)
    out.write_text(json.dumps(rows))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]))
