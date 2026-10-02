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


def rim_follower(rims_df, rim_id: int | None, fps: float, max_gap_frames: int = 10):
    """A function from a time to the rim's box at that moment (None when the rim is not tracked near it).

    The Hudl camera pans, so the rim moves across the frame within a second; a montage must follow it."""
    df = rims_df if rim_id is None else rims_df[rims_df["track_id"] == rim_id]
    df = df.dropna(subset=["x1", "y1", "x2", "y2"])
    boxes = {int(r.frame): (float(r.x1), float(r.y1), float(r.x2), float(r.y2)) for r in df.itertuples()}
    frames = np.array(sorted(boxes))

    def at(t: float):
        if len(frames) == 0:
            return None
        frame = int(round(t * fps))
        nearest = int(frames[np.abs(frames - frame).argmin()])
        return boxes[nearest] if abs(nearest - frame) <= max_gap_frames else None

    return at


def build_montage(source: Path, t: float, rim_box, frame_w: int, frame_h: int,
                  tile: int = TILE, n: int = N_TILES, step_s: float = STEP_S, lead_s: float = LEAD_S,
                  rim_at=None) -> np.ndarray:
    """A grid of rim crops from t - lead_s onward, every step_s: the ball's fate after it reaches the rim.

    With `rim_at` (see rim_follower) each crop is centred on the rim where it is in that frame; when the
    rim is not tracked at some moment the crop stays where the rim was last seen."""
    box = montage_box(rim_box, frame_w, frame_h)
    start = max(0.0, t - lead_s)
    end = start + (n - 1) * step_s + step_s / 2
    tiles = []
    for frame_t, frame in iter_frames(source, fps=1 / step_s, start=start, end=end):
        here = rim_at(frame_t) if rim_at is not None else None
        if here is not None:
            box = montage_box(here, frame_w, frame_h)
        tiles.append(cv2.resize(crop(frame, box), (tile, tile), interpolation=cv2.INTER_AREA))
        if len(tiles) == n:
            break
    return montage_from_tiles(tiles, tile)
