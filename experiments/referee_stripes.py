"""Experiment: can a referee be told from a player by the stripes on the torso? For a sample of each game's shots, the
person chosen as shooter at the release, scored for stripes, tiled best-scored first with the score on each crop."""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from film.common.manifest import Manifest
from film.perceive.shotspots import confirmed_shots
from film.train.prelabel import first_frame_at


def stripe_score(frame: np.ndarray, box) -> dict:
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    w, h = x2 - x1, y2 - y1
    t = frame[max(y1 + int(0.15 * h), 0): y1 + int(0.5 * h), max(x1 + int(0.2 * w), 0): x1 + int(0.8 * w)]
    if t.size == 0 or t.shape[1] < 6:
        return {"alt": 0, "amp": 0.0, "sat": 255.0, "dark": 0.0, "light": 0.0}
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 2].astype(float)
    # Stripes run down the shirt, so they alternate along each row; a row's own crossings of its mean, averaged over rows.
    alts = []
    for row in gray:
        a = np.sign(row - row.mean())
        alts.append(int(np.sum(a[1:] * a[:-1] < 0)))
    prof = gray.mean(axis=0)
    return {"alt": float(np.mean(alts)), "amp": float(np.std(prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 3).ravel())),
            "sat": float(np.median(hsv[:, :, 1])), "dark": float(np.mean(gray < 70)), "light": float(np.mean(gray > 150))}


def main(out: str, per_game: int) -> None:
    rows = []
    for f in sorted(glob.glob("games/*/shot_looks.json.gz")):
        gd = Path(f).parent
        m = Manifest.load(gd)
        if os.stat(m.source).st_blocks == 0:
            continue
        shots = [e for e in confirmed_shots(gd) if e.x is not None and e.signals["location"].get("box")]
        for e in shots[:: max(1, len(shots) // per_game)][:per_game]:
            loc = e.signals["location"]
            frame = first_frame_at(Path(m.source), loc["release_t"], m.fps)
            if frame is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in loc["box"])
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            s = stripe_score(frame, loc["box"])
            rows.append((s, cv2.resize(crop, (90, 180)), gd.name[:6] + " " + e.id[5:11], y2 - y1))
        print(gd.name[:40], len(rows), flush=True)
    rows.sort(key=lambda r: -r[0]["alt"] * (1 if r[0]["sat"] < 80 else 0.3))
    tiles = []
    for s, c, name, h in rows:
        t = np.full((222, 90, 3), 30, np.uint8)
        t[:180] = c
        cv2.putText(t, f"a{s['alt']:.1f} s{s['sat']:.0f}", (1, 193), 0, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(t, f"d{s['dark']:.1f} l{s['light']:.1f} h{h}", (1, 206), 0, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(t, name[-6:], (1, 219), 0, 0.33, (180, 180, 180), 1, cv2.LINE_AA)
        tiles.append(t)
    cols = 20
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out, len(rows))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
