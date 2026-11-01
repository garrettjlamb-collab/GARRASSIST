"""Experiment: body poses over the seconds before each confirmed shot, kept in games/<id>/shot_poses.json.gz, so a rule on raised
arms can be tried without running the model again."""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

from film.common.cloud import game_film, local_source
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.registry import base_weights_path
from film.common.video import iter_frames
from film.perceive.shotspots import AFTER_S, BEFORE_S, FPS, confirmed_shots


def run(game_id: str, model) -> None:
    gd = game_dir(game_id)
    path = gd / "shot_poses.json.gz"
    done = {}
    if path.exists():
        with gzip.open(path, "rt") as f:
            done = json.load(f)
    m = Manifest.load(gd)
    shots = [e for e in confirmed_shots(gd) if e.id not in done]
    start = time.time()
    with local_source(game_film(game_id)) as source:
        for i, e in enumerate(shots):
            frames = []
            for t, frame in iter_frames(Path(source), FPS, start=max(0.0, e.t - BEFORE_S), end=e.t + AFTER_S):
                r = model.predict(frame, imgsz=m.width, conf=0.2, device="mps", verbose=False)[0]
                people = []
                if r.keypoints is not None and len(r.boxes):
                    for box, conf, kp in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.keypoints.data.tolist()):
                        people.append({"box": [round(v, 1) for v in box], "conf": round(conf, 2), "kp": [[round(a, 1), round(b, 1), round(c, 2)] for a, b, c in kp]})
                frames.append({"t": round(t, 3), "people": people})
            done[e.id] = frames
            if i % 25 == 24:
                with gzip.open(path, "wt") as f:
                    json.dump(done, f)
                print(f"{game_id[:30]} {i + 1}/{len(shots)} {time.time() - start:.0f}s", flush=True)
    with gzip.open(path, "wt") as f:
        json.dump(done, f)
    print(f"{game_id[:30]} done {len(done)} shots, {time.time() - start:.0f}s, {path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO(str(base_weights_path("yolo11n-pose.pt")))
    for g in sys.argv[1:]:
        run(g, model)
