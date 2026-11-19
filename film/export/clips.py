from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2

from film.common.cloud import may_need_film
from film.common.ffmpeg import ffmpeg_exe
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.events import load_events


def cut_clip(source: Path, start: float, end: float, out_path: Path, width: int = 640) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(".part.mp4")
    base = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}", "-i", str(source),
            "-t", f"{max(0.1, end - start):.3f}", "-vf", f"scale={width}:-2", "-an", "-movflags", "+faststart"]

    def run(codec_args: list[str]) -> str | None:
        """Return None on success, else ffmpeg's decoded stderr."""
        result = subprocess.run(base + codec_args + [str(part)], capture_output=True)
        return None if result.returncode == 0 else result.stderr.decode(errors="replace").strip()

    err = run(["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"])
    if err is not None and ("Unknown encoder" in err or "libx264" in err):
        err = run(["-c:v", "mpeg4", "-q:v", "5"])
    if err is not None:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed cutting {source} {start:.2f}-{end:.2f}: {err}")
    os.replace(part, out_path)
    return out_path


REEL_WIDTH = 1920   # reels at the film's own size: 640 wide and three encodes is what made them look bad
REEL_CRF = 22       # about the bitrate of a Hudl 1080p download (~5 Mbps): ~11 MB per 18 s clip


def cut_labeled_clip(source: Path, start: float, end: float, label: str, out_path: Path,
                     width: int = REEL_WIDTH, crf: int = REEL_CRF) -> Path:
    """One play from the film with its label drawn on, in a single encode. Every clip gets the same codec
    settings, so a reel joins them without encoding again (`concat_clips`)."""
    out_path = Path(out_path)
    text = out_path.with_suffix(".txt")
    text.write_text(label)
    draw = (f"drawtext=textfile='{text}':x=24:y=24:fontsize=h/24:fontcolor=white:borderw=3:bordercolor=black")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}", "-i", str(source),
           "-t", f"{max(0.1, end - start):.3f}", "-vf", f"scale='min({width},iw)':-2,{draw}", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", "30",
           "-movflags", "+faststart", str(out_path)]
    result = subprocess.run(cmd, capture_output=True)
    text.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed cutting {source} {start:.2f}-{end:.2f}: "
                           f"{result.stderr.decode(errors='replace').strip()}")
    return out_path


def concat_clips(paths: list[Path], out_path: Path) -> Path:
    """Join clips cut by `cut_labeled_clip` into one video, copying the streams (no quality lost)."""
    if not paths:
        raise ValueError("no clips to build a reel from")
    out_path = Path(out_path)
    listing = out_path.with_suffix(".list.txt")
    listing.write_text("".join("file '" + str(Path(p).resolve()).replace("'", "'\\''") + "'\n" for p in paths))
    part = out_path.with_suffix(".part.mp4")
    result = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                             "-c", "copy", "-movflags", "+faststart", str(part)], capture_output=True)
    listing.unlink(missing_ok=True)
    if result.returncode != 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed joining {len(paths)} clips: {result.stderr.decode(errors='replace').strip()}")
    os.replace(part, out_path)
    return out_path


def clip_path(game_dir_path: Path, event_id: str) -> Path:
    return Path(game_dir_path) / "clips" / f"{event_id}.mp4"


@may_need_film
def clips_for_game(game_id: str, force: bool = False, log=print, *, film) -> int:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    n = 0
    for e in load_events(gd):
        out = clip_path(gd, e.id)
        if out.exists() and not force:
            continue
        cut_clip(film(), e.clip["start"], e.clip["end"], out)
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
