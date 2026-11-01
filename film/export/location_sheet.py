"""The dots drawn back on the film: a contact sheet of shots at their release with the court, the shooter's box and a half
court showing the dot, the check by eye that a dot is where the shot was taken (games/<id>/location_sheet.jpg)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.export.court_sheet import draw_court
from film.interpret.events import load_events
from film.perceive.court_run import load_mappings, mapping_at

SHEET = "location_sheet.jpg"


def location_sheet(game_id: str, n: int = 12, cols: int = 3, tile_width: int = 640, log=print) -> Path | None:
    from film.train.prelabel import first_frame_at

    gd = game_dir(game_id)
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    placed = [e for e in load_events(gd) if e.x is not None and e.signals.get("location", {}).get("method") in ("flight", "path")]
    if not placed:
        log(f"location sheet {game_id}: no flight-placed shots to draw")
        return None
    tiles = []
    for i in sorted({int(i) for i in np.linspace(0, len(placed) - 1, min(n, len(placed)))}):
        e = placed[i]
        loc = e.signals["location"]
        frame = first_frame_at(Path(m.source), loc["release_t"], m.fps)
        mapping = mapping_at(maps, loc["release_t"])
        if frame is None or mapping is None:
            continue
        img = draw_court(frame, mapping, thickness=1)
        if loc["box"]:
            x1, y1, x2, y2 = (int(v) for v in loc["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        scale = tile_width / img.shape[1]
        img = cv2.resize(img, (tile_width, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        label = f"{int(e.t // 60)}:{int(e.t % 60):02d} {e.zone} ({e.x:.0f},{e.y:.0f}) conf {e.location_conf} {e.signals.get('points') if e.made else 'miss'}"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    path = gd / SHEET
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log(f"location sheet {game_id}: {len(tiles)} shots drawn -> {path}")
    return path
