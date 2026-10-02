from __future__ import annotations

from pathlib import Path
from statistics import median

from film.common.video import iter_frames
from film.perceive.ball import crop

PATH_FPS = 15.0
LEAD_S = 0.2
AFTER_S = 1.4
CROP_SIZE = 320       # the size the ball detector was trained on
DOWN_SHIFT = 0.1      # fraction of the crop by which its centre sits below the rim: the ball's fate is below

# Thresholds are in rim-box widths. Measured on the Marcos and Cactus Shadows games (Hudl view).
COLUMN_HALF_WIDTH = 0.24          # a made ball stays this close to the hoop's centre line on its way down
NET_TOP, NET_BOTTOM = 0.2, 1.6    # the stretch below the rim where a made ball is seen dropping
COLUMN_UNTIL_S = 0.75
SIDEWAYS_FROM_S, SIDEWAYS_UNTIL_S = 0.2, 0.87
BACK_UP_FROM_S = 0.27
HOOP_MIN_CONF, HOOP_MIN_SAMPLES, HOOP_WINDOW_S = 0.4, 5, 0.2

FEATURES = ["column_frames", "depth", "sideways", "back_up", "seen", "peak_conf"]


def path_crop_box(rim_box, frame_w: int, frame_h: int, size: int = CROP_SIZE) -> tuple[int, int, int, int]:
    """A native-resolution square around the rim with more room below it than above, clamped to the frame."""
    size = min(size, frame_w, frame_h)
    cx, cy = (rim_box[0] + rim_box[2]) / 2, (rim_box[1] + rim_box[3]) / 2 + DOWN_SHIFT * size
    x1 = max(0, min(int(round(cx - size / 2)), frame_w - size))
    y1 = max(0, min(int(round(cy - size / 2)), frame_h - size))
    return x1, y1, x1 + size, y1 + size


def read_ball_path(source: Path, t: float, rim_box, frame_w: int, frame_h: int, detector, rim_at=None) -> dict:
    """Every ball the detector sees around the rim from just before an arrival to 1.4 s after it, at 15 fps.

    Each ball is [dx, dy, conf]: its centre relative to the rim box's centre, in rim-box widths, y downward.
    With `rim_at` (see rim_follower) the crop follows the rim as the camera pans."""
    start = max(0.0, t - LEAD_S)
    crops, rims, times = [], [], []
    for frame_t, frame in iter_frames(source, fps=PATH_FPS, start=start, end=t + AFTER_S + 0.5 / PATH_FPS):
        here = rim_at(frame_t) if rim_at is not None else None
        if here is not None:
            rim_box = here
        box = path_crop_box(rim_box, frame_w, frame_h)
        crops.append(crop(frame, box))
        rims.append((rim_box, box))
        times.append(frame_t)
    frames = []
    for frame_t, (rim, box), found in zip(times, rims, detector.detect_many(crops)):
        width = max(rim[2] - rim[0], 1.0)
        rcx, rcy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
        balls = [[round((b.cx + box[0] - rcx) / width, 3), round((b.cy + box[1] - rcy) / width, 3), round(b.conf, 3)]
                 for b in found if b.cls == "ball"]
        frames.append({"dt": round(frame_t - t, 3), "balls": balls})
    return {"rim_w": float(rim_box[2] - rim_box[0]), "frames": frames}


def hoop_offset(paths: list[dict]) -> float:
    """How far the hoop sits from the rim box's centre, in rim widths: the median x of confident balls as they
    arrive. Zero (the rim box's own centre) when too few arrivals were seen to say."""
    xs = [b[0] for p in paths for f in p["frames"] if 0 <= f["dt"] < HOOP_WINDOW_S
          for b in f["balls"] if b[2] >= HOOP_MIN_CONF]
    return float(median(xs)) if len(xs) >= HOOP_MIN_SAMPLES else 0.0


def path_features(path: dict, hoop_dx: float = 0.0) -> list[float]:
    """Six numbers that tell a make from a miss from nothing, in the order of FEATURES.

    column_frames  share of the frames after arrival with a ball in the net column under the hoop
    depth          how far below the rim the ball got while still in that column
    sideways       how far from the hoop's centre line the ball typically was afterwards
    back_up        share of the later frames with a ball above the rim again
    seen           share of all frames with any ball
    peak_conf      the detector's best confidence"""
    frames = path["frames"]
    after = [f for f in frames if f["dt"] >= -1e-6]
    early = [f for f in after if f["dt"] <= COLUMN_UNTIL_S]
    late = [f for f in after if f["dt"] >= BACK_UP_FROM_S]

    def in_column(b):
        return abs(b[0] - hoop_dx) < COLUMN_HALF_WIDTH

    column_frames = sum(any(in_column(b) and NET_TOP <= b[1] < NET_BOTTOM for b in f["balls"]) for f in early)
    depths = [b[1] for f in early for b in f["balls"] if in_column(b)]
    spread = [abs(b[0] - hoop_dx) for f in after if SIDEWAYS_FROM_S <= f["dt"] <= SIDEWAYS_UNTIL_S for b in f["balls"]]
    back_up = sum(any(b[1] < 0 for b in f["balls"]) for f in late)
    return [
        column_frames / max(len(early), 1),
        max(depths, default=-0.5),
        float(median(spread)) if spread else 1.3,
        back_up / max(len(late), 1),
        sum(bool(f["balls"]) for f in frames) / max(len(frames), 1),
        max((b[2] for f in frames for b in f["balls"]), default=0.0),
    ]
