"""The pictures a made-or-missed classifier looks at: sixteen small frames of the hoop and net, from two
frames before the ball arrives to 0.93 s after, laid out as one sheet."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from film.common.video import iter_frames

VIEW_FPS = 15.0
LEAD_FRAMES = 2
N_FRAMES = 16
GRID = 4
TILE = 112
# Measured on six reviewed games: a view 2.6 rim widths across leaves the ball about 10 px and the
# classifier blind; at 1.6 the ball and the net's movement are plain to see.
VIEW_WIDTHS = 1.6
DROP_WIDTHS = 0.35          # the view's centre sits this far below the rim box's centre: the net, not the glass
MIN_SIDE = 64


def view_square(rim_box, hoop_dx: float = 0.0) -> tuple[int, int, int]:
    """(centre x, centre y, side) of the view for a rim box; hoop_dx is the hoop's offset in rim widths."""
    w = max(rim_box[2] - rim_box[0], 8.0)
    side = int(round(max(VIEW_WIDTHS * w, MIN_SIDE)))
    cx = int(round((rim_box[0] + rim_box[2]) / 2 + hoop_dx * w))
    cy = int(round((rim_box[1] + rim_box[3]) / 2 + DROP_WIDTHS * w))
    return cx, cy, side


def read_rim_view(source: Path, t: float, rim_box, hoop_dx: float = 0.0, rim_at=None) -> np.ndarray:
    """The sheet for an arrival at time t. With `rim_at` (see rim_follower) the view follows a panning camera."""
    start = max(0.0, t - LEAD_FRAMES / VIEW_FPS)
    tiles: list[np.ndarray] = []
    for frame_t, frame in iter_frames(source, fps=VIEW_FPS, start=start, end=start + (N_FRAMES + 1) / VIEW_FPS):
        here = rim_at(frame_t) if rim_at is not None else None
        if here is not None:
            rim_box = here
        cx, cy, side = view_square(rim_box, hoop_dx)
        padded = cv2.copyMakeBorder(frame, side, side, side, side, cv2.BORDER_REPLICATE)
        x1, y1 = cx - side // 2 + side, cy - side // 2 + side
        x1 = max(0, min(x1, padded.shape[1] - side))
        y1 = max(0, min(y1, padded.shape[0] - side))
        tiles.append(cv2.resize(padded[y1:y1 + side, x1:x1 + side], (TILE, TILE), interpolation=cv2.INTER_AREA))
        if len(tiles) == N_FRAMES:
            break
    if not tiles:
        tiles = [np.zeros((TILE, TILE, 3), np.uint8)]
    while len(tiles) < N_FRAMES:
        tiles.append(tiles[-1].copy())
    return np.vstack([np.hstack(tiles[r * GRID:(r + 1) * GRID]) for r in range(GRID)])


def split_tiles(sheet: np.ndarray) -> list[np.ndarray]:
    return [sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for r in range(GRID) for c in range(GRID)]
