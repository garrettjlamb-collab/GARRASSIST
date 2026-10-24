"""The court drawn back onto the film: a contact sheet of frames with the fitted court's lines over them, the check
by eye that a court mapping is right before anything is measured with it (games/<id>/court_sheet.jpg)."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.court import (BASKET, FT_CIRCLE_RADIUS, FT_LINE_Y, FULL_DEPTH, HALF_DEPTH, LANE_LEFT, LANE_RIGHT,
                                  THREE_RADIUS, WIDTH, Mapping)
from film.perceive.court_run import FILE, load_mappings

SHEET = "court_sheet.jpg"
LINE_COLOR = (0, 255, 255)            # yellow, BGR


def _segment(a, b, n: int = 40) -> np.ndarray:
    return np.linspace(a, b, n)


def _arc(cx: float, cy: float, r: float, start: float, stop: float, n: int = 60) -> np.ndarray:
    a = np.linspace(start, stop, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def _near_end() -> list[np.ndarray]:
    bx, by = BASKET
    meet = math.asin(-by / THREE_RADIUS)                      # where the arc meets the baseline (y = 0)
    return [
        _segment((LANE_LEFT, 0.0), (LANE_LEFT, FT_LINE_Y)), _segment((LANE_RIGHT, 0.0), (LANE_RIGHT, FT_LINE_Y)),
        _segment((LANE_LEFT, FT_LINE_Y), (LANE_RIGHT, FT_LINE_Y)),
        _arc(BASKET[0], FT_LINE_Y, FT_CIRCLE_RADIUS, 0.0, 2 * math.pi),
        _arc(bx, by, THREE_RADIUS, meet, math.pi - meet),
    ]


def court_lines() -> list[np.ndarray]:
    """Every painted line of the template court, as polylines in full-court feet."""
    outline = [_segment((0.0, 0.0), (WIDTH, 0.0)), _segment((WIDTH, 0.0), (WIDTH, FULL_DEPTH)),
               _segment((WIDTH, FULL_DEPTH), (0.0, FULL_DEPTH)), _segment((0.0, FULL_DEPTH), (0.0, 0.0)),
               _segment((0.0, HALF_DEPTH), (WIDTH, HALF_DEPTH)), _arc(WIDTH / 2, HALF_DEPTH, 6.0, 0.0, 2 * math.pi)]
    near = _near_end()
    far = [np.stack([WIDTH - p[:, 0], FULL_DEPTH - p[:, 1]], axis=1) for p in near]
    return outline + near + far


def draw_court(frame: np.ndarray, mapping: Mapping, color=LINE_COLOR, thickness: int = 2) -> np.ndarray:
    """The frame with the court's lines drawn where the mapping puts them. Pieces behind the camera are left out."""
    out = frame.copy()
    back = np.linalg.inv(mapping.H)
    for line in court_lines():
        v = back @ np.vstack([line.T, np.ones(len(line))])
        ok = v[2] > 1e-9
        px = (v[:2] / np.where(ok, v[2], 1.0)).T
        run: list[list[int]] = []
        for good, (x, y) in zip(ok, px):
            if good and abs(x) < 1e5 and abs(y) < 1e5:
                run.append([int(round(x)), int(round(y))])
            elif run:
                cv2.polylines(out, [np.array(run, np.int32)], False, color, thickness, cv2.LINE_AA)
                run = []
        if len(run) > 1:
            cv2.polylines(out, [np.array(run, np.int32)], False, color, thickness, cv2.LINE_AA)
    return out


def court_sheet(game_id: str, n: int = 12, cols: int = 3, tile_width: int = 640, log=print) -> Path | None:
    """n frames spread through the game's fitted court, lines drawn, tiled into games/<id>/court_sheet.jpg."""
    from film.train.prelabel import first_frame_at

    gd = game_dir(game_id)
    m = Manifest.load(gd)
    maps = load_mappings(gd)
    if not maps:
        log(f"court sheet {game_id}: no court.parquet rows to draw")
        return None
    df = pd.read_parquet(gd / FILE)
    source = dict(zip(df["t"].astype(float), df["source"])) if "source" in df else {}
    picks = sorted({int(i) for i in np.linspace(0, len(maps) - 1, min(n, len(maps)))})
    tiles = []
    for i in picks:
        t, mapping = maps[i]
        frame = first_frame_at(Path(m.source), t, m.fps)
        if frame is None:
            continue
        img = draw_court(frame, mapping)
        scale = tile_width / img.shape[1]
        img = cv2.resize(img, (tile_width, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        label = f"{int(t // 60)}:{int(t % 60):02d}  {source.get(t, 'model')}  conf {mapping.conf:.2f}  err {mapping.error_px:.1f}px"
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
    log(f"court sheet {game_id}: {len(picks)} frames drawn -> {path}")
    return path
