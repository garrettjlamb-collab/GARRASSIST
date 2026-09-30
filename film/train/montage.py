from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from film.common.video import iter_frames
from film.perceive.ball import crop

TILE = 96
GRID = 3
N_TILES = GRID * GRID
STEP_S = 0.2
LEAD_S = 0.4


def montage_box(rim_box, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """A square three rim-widths wide, centred half a rim-width below the rim: made balls drop, misses bounce."""
    x1, y1, x2, y2 = rim_box
    width = max(x2 - x1, 8.0)
    side = min(int(round(3 * width)), frame_w, frame_h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + 0.5 * width
    bx1 = max(0, min(int(round(cx - side / 2)), frame_w - side))
    by1 = max(0, min(int(round(cy - side / 2)), frame_h - side))
    return bx1, by1, bx1 + side, by1 + side


def montage_from_tiles(tiles: list[np.ndarray], tile: int = TILE, grid: int = GRID) -> np.ndarray:
    tiles = list(tiles)[: grid * grid]
    blank = np.zeros((tile, tile, 3), np.uint8)
    while len(tiles) < grid * grid:
        tiles.append(tiles[-1].copy() if tiles else blank.copy())
    rows = [np.hstack(tiles[r * grid:(r + 1) * grid]) for r in range(grid)]
    return np.vstack(rows)


def build_montage(source: Path, t: float, rim_box, frame_w: int, frame_h: int,
                  tile: int = TILE, n: int = N_TILES, step_s: float = STEP_S, lead_s: float = LEAD_S) -> np.ndarray:
    """A grid of rim crops from t - lead_s onward, every step_s: the ball's fate after it reaches the rim."""
    box = montage_box(rim_box, frame_w, frame_h)
    start = max(0.0, t - lead_s)
    end = start + (n - 1) * step_s + step_s / 2
    tiles = []
    for _, frame in iter_frames(source, fps=1 / step_s, start=start, end=end):
        tiles.append(cv2.resize(crop(frame, box), (tile, tile), interpolation=cv2.INTER_AREA))
        if len(tiles) == n:
            break
    return montage_from_tiles(tiles, tile)
