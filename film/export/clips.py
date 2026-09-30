from __future__ import annotations

import subprocess
from pathlib import Path

import cv2

from film.common.ffmpeg import ffmpeg_exe
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.events import load_events


def cut_clip(source: Path, start: float, end: float, out_path: Path, width: int = 640) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}", "-i", str(source),
            "-t", f"{max(0.1, end - start):.3f}", "-vf", f"scale={width}:-2", "-an", "-movflags", "+faststart"]
    try:
        subprocess.run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", str(out_path)], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + ["-c:v", "mpeg4", "-q:v", "5", str(out_path)], check=True, capture_output=True)
    return out_path


def clip_path(game_dir_path: Path, event_id: str) -> Path:
    return Path(game_dir_path) / "clips" / f"{event_id}.mp4"


def clips_for_game(game_id: str, force: bool = False, log=print) -> int:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    n = 0
    for e in load_events(gd):
        out = clip_path(gd, e.id)
        if out.exists() and not force:
            continue
        cut_clip(Path(m.source), e.clip["start"], e.clip["end"], out)
        n += 1
    log(f"clips {game_id}: {n} cut")
    return n


def build_reel(clips: list[tuple[Path, str]], out_path: Path, size: tuple[int, int] = (640, 360)) -> Path:
    """Concatenate clips into one video with a label burned into each frame."""
    writer = None
    for path, label in clips:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, size)
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)
        cap.release()
    if writer is None:
        raise ValueError("no clips to build a reel from")
    writer.release()
    return Path(out_path)
