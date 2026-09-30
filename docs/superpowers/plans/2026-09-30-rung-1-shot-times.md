# Rung 1: Shot Times Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a Hudl Tactical game file, produce every shot attempt's time, made-or-missed, quarter and clock, a review queue where Garrett confirms each one with a keystroke, and a clip reel of the confirmed shots.

**Architecture:** A `film` command-line tool in Python. Ingest probes the file and picks a camera profile. Perceive streams the video once (no frames written to disk), finding rims and the ball near each rim with small detectors, computing motion energy at each rim, and reading the burned-in scoreboard once a second. Interpret turns those into shot events by a "ball crosses the rim plane downward" rule, calls made-or-missed with a small classifier cross-checked against the scoreboard, and writes `events.json`. A local FastAPI review page plays each shot's clip and records decisions. The same page labels boxes and clips so the detectors and classifier improve from Garrett's corrections; until they are trained, open-vocabulary fallbacks keep the pipeline running end to end.

**Tech Stack:** Python 3.12 managed by `uv`; PyAV for decoding and probing (no Homebrew, no system ffmpeg required); `imageio-ffmpeg`'s bundled ffmpeg binary for cutting clips; OpenCV; NumPy; pandas + pyarrow; PyTorch on the Metal backend; Ultralytics YOLO (`yolo11n`) and YOLO-World; torchvision ResNet-18; FastAPI + uvicorn; a single vanilla-JS page; pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-film-pipeline-design.md`. This plan implements rung 1 of section 13. Rung 2 (locations, teams, tracker export) and the tracker changes of section 9 get their own plans.

## Global Constraints

- Python `>=3.12`. Environment managed by `uv` (`uv sync --extra dev`, `uv run ...`). No Homebrew. No rented GPU.
- Device selection always goes through `film.common.device.pick_device()`: explicit argument, else `FILM_DEVICE`, else `mps` > `cuda` > `cpu`.
- Video is never written out as frames. Every stage decodes on the fly with `film.common.video.iter_frames`.
- Working frame rate default `10.0` fps (`--fps 5` on the laptop). Frame index is always `int(round(t * fps))`.
- Directory roots come from `film.common.paths` and honour `FILM_GAMES_DIR`, `FILM_MODELS_DIR`, `FILM_DATASETS_DIR`, `FILM_LABELS_DIR` (defaults `games/`, `models/`, `datasets/`, `labels/`, all git-ignored).
- Tests never need trained weights, the internet, or a GPU. Detectors and the classifier are injected; tests use `FakeDetector` and a `FakeClassifier`.
- Scoreboard reading in rung 1 exists for the `hudl-tactical` profile only. Other profiles produce no scoreboard timeline and the code paths must tolerate that.
- Court coordinates (`court_x`, `court_y`, `on_court`, `team`) are columns in the tracks file but are always null in rung 1.
- Every task ends with a commit whose message ends with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- **Human steps** are marked `**Human step (Garrett):**`. When an executor reaches one, it stops that task, reports what is needed, and continues with any later task that does not depend on it.

## File map

```
basketball-film-pipeline/
  pyproject.toml, .gitignore, README.md
  film/
    __init__.py                     __version__
    cli.py                          argparse entry; imports film.commands.*
    commands/                       one file per subcommand, each with register(sub)
      doctor.py ingest.py calibrate.py scoreboard.py perceive.py interpret.py
      prelabel.py train.py clips.py review.py export.py eval.py run.py
    common/
      device.py                     pick_device()
      ffmpeg.py                     ffmpeg_exe()
      paths.py                      games_dir(), game_dir(), game_id_for(), slugify(), models_dir(), datasets_dir(), labels_dir()
      manifest.py                   Manifest dataclass: load/save/mark
      registry.py                   latest(), new_version_dir(), write_manifest(), read_manifest()
      video.py                      VideoInfo, probe(), iter_frames()
    ingest/
      profile.py                    led_fraction(), detect_profile()
      run.py                        ingest()
    perceive/
      profiles/hudl_tactical.json   scoreboard cell geometry
      sevenseg.py                   led_mask(), decode_cell(), decode_number()
      scoreboard.py                 ScoreboardRead, load_profile(), read_scoreboard(), clean_timeline(), lookup(), save/load
      detectors.py                  Box, Detector, FakeDetector, YoloDetector, load_detector()
      tracks.py                     COLUMNS, rows_to_frame(), replace_kind(), read_tracks()
      rims.py                       RimTracker, detect_rims(), smooth_rims()
      ball.py                       rim_crop_box(), crop(), detect_ball_near_rim()
      motion.py                     MotionEnergy
      run.py                        perceive()
    interpret/
      arrivals.py                   Arrival, find_ball_arrivals(), find_motion_arrivals(), merge_arrivals()
      events.py                     Event, build_shot_events(), save_events(), load_events()
      run.py                        interpret()
    train/
      datasets.py                   pending labels, YOLO label writing, data.yaml, accept_*()
      prelabel.py                   prelabel_rims(), prelabel_balls()
      montage.py                    montage_box(), build_montage(), montage_from_tiles()
      rimclf.py                     build_model(), RimClassifier, train_rimclf()
      rimclf_dataset.py             build_rimclf_candidates()
      train_yolo.py                 train_detector(), train_rimclf_version()
    review/
      decisions.py                  load_decisions(), save_decision(), apply_decisions()
      app.py                        create_app()
      static/index.html             review + labeling UI
    export/
      clips.py                      cut_clip(), clips_for_game(), build_reel()
      shots.py                      export_shots()
      evaluate.py                   evaluate(), load_reference()
  fixtures/frames/*.jpg             committed frames from real film (already present)
  tests/                            one test file per module, conftest.py makes synthetic videos
```

---

### Task 1: Scaffold, environment, device selection, `film doctor`

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `film/__init__.py`, `film/cli.py`, `film/commands/__init__.py`, `film/commands/doctor.py`, `film/common/__init__.py`, `film/common/device.py`, `film/common/ffmpeg.py`
- Test: `tests/test_device.py`, `tests/test_ffmpeg.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `pick_device(prefer: str | None = None) -> str`; `ffmpeg_exe() -> str`; `film.cli.main(argv: list[str] | None = None) -> int`; every command module exposes `register(sub: argparse._SubParsersAction) -> None` and sets `parser.set_defaults(func=handler)` where `handler(args) -> int`.

- [ ] **Step 1: Install uv (no Homebrew)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```
Expected: a version line. If `curl` is blocked, `python3 -m pip install --user uv` and add `~/Library/Python/3.12/bin` to `PATH`.

- [ ] **Step 2: Write the project files**

`pyproject.toml`:
```toml
[project]
name = "basketball-film-pipeline"
version = "0.1.0"
description = "Turn downloaded game film into shot data for the shot tracker"
requires-python = ">=3.12"
dependencies = [
  "av>=14",
  "numpy>=2.0",
  "opencv-python>=4.10",
  "pillow>=10",
  "pandas>=2.2",
  "pyarrow>=17",
  "torch>=2.5",
  "torchvision>=0.20",
  "ultralytics>=8.3",
  "imageio-ffmpeg>=0.5",
  "fastapi>=0.115",
  "uvicorn>=0.30",
]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]

[project.scripts]
film = "film.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["film"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.DS_Store
games/
datasets/
labels/
models/
!models/**/manifest.json
fixtures/clips/
*.pt
runs/
```

`film/__init__.py`:
```python
__version__ = "0.1.0"
```

`film/common/__init__.py` and `film/commands/__init__.py`: empty files.

`film/common/device.py`:
```python
from __future__ import annotations

import os


def pick_device(prefer: str | None = None) -> str:
    """Torch device to use: an explicit preference, else FILM_DEVICE, else mps > cuda > cpu."""
    choice = prefer or os.environ.get("FILM_DEVICE")
    if choice:
        return choice
    try:
        import torch
    except ImportError:
        return "cpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

`film/common/ffmpeg.py`:
```python
from __future__ import annotations

import shutil


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: the system one if present, else the one bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
```

`film/cli.py`:
```python
from __future__ import annotations

import argparse
import importlib

from film import __version__

COMMANDS = [
    "film.commands.doctor",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="film", description="Basketball film pipeline")
    parser.add_argument("--version", action="version", version=f"film {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    for module_name in COMMANDS:
        importlib.import_module(module_name).register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
```

`film/commands/doctor.py`:
```python
from __future__ import annotations

import platform
import sys


def register(sub) -> None:
    parser = sub.add_parser("doctor", help="check the environment")
    parser.set_defaults(func=run)


def run(args) -> int:
    print(f"python {sys.version.split()[0]} {platform.machine()}")
    try:
        import torch

        mps = getattr(torch.backends, "mps", None)
        print(f"torch {torch.__version__} mps={bool(mps and mps.is_available())} cuda={torch.cuda.is_available()}")
    except ImportError:
        print("torch MISSING")
    for name in ("av", "cv2", "ultralytics", "pandas", "pyarrow", "fastapi"):
        try:
            module = __import__(name)
            print(f"{name} {getattr(module, '__version__', 'ok')}")
        except ImportError:
            print(f"{name} MISSING")
    from film.common.device import pick_device
    from film.common.ffmpeg import ffmpeg_exe
    from film.common.paths import datasets_dir, games_dir, labels_dir, models_dir

    print(f"ffmpeg {ffmpeg_exe()}")
    print(f"device {pick_device()}")
    print(f"games {games_dir()}")
    print(f"models {models_dir()}")
    print(f"datasets {datasets_dir()}")
    print(f"labels {labels_dir()}")
    return 0
```
Note: `film.common.paths` is written in Task 2. Until then `doctor` will fail on that import, which is why Task 2 is next and the doctor test lives in Task 2.

- [ ] **Step 3: Create the environment**

```bash
cd ~/basketball-film-pipeline
uv sync --extra dev
```
Expected: a `.venv` with torch, ultralytics and the rest. This downloads several hundred megabytes; it can take ten minutes.

- [ ] **Step 4: Write the failing tests**

`tests/test_device.py`:
```python
import sys
import types

from film.common.device import pick_device


def _fake_torch(mps: bool, cuda: bool):
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
    )


def test_explicit_preference_wins(monkeypatch):
    monkeypatch.setenv("FILM_DEVICE", "cpu")
    assert pick_device("mps") == "mps"


def test_env_variable_used(monkeypatch):
    monkeypatch.setenv("FILM_DEVICE", "cuda")
    assert pick_device() == "cuda"


def test_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("FILM_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=False, cuda=False))
    assert pick_device() == "cpu"


def test_prefers_mps(monkeypatch):
    monkeypatch.delenv("FILM_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=False))
    assert pick_device() == "mps"
```

`tests/test_ffmpeg.py`:
```python
import imageio_ffmpeg

from film.common import ffmpeg as ff


def test_prefers_system_ffmpeg(monkeypatch):
    monkeypatch.setattr(ff.shutil, "which", lambda name: "/usr/local/bin/ffmpeg")
    assert ff.ffmpeg_exe() == "/usr/local/bin/ffmpeg"


def test_falls_back_to_bundled_binary(monkeypatch):
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/bundled/ffmpeg")
    assert ff.ffmpeg_exe() == "/bundled/ffmpeg"
```

`tests/test_cli.py`:
```python
import pytest

from film.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "film 0.1.0" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out
```

- [ ] **Step 5: Run the tests to verify they pass** (the modules were written in Step 2)

Run: `uv run pytest tests/test_device.py tests/test_ffmpeg.py tests/test_cli.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore uv.lock film tests
git commit -m "Scaffold the film CLI with device selection and an ffmpeg resolver

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Paths and the game manifest

**Files:**
- Create: `film/common/paths.py`, `film/common/manifest.py`
- Test: `tests/test_paths.py`, `tests/test_manifest.py`, `tests/test_doctor.py`

**Interfaces:**
- Produces: `games_dir() -> Path`, `models_dir() -> Path`, `datasets_dir() -> Path`, `labels_dir() -> Path`, `game_dir(game_id: str) -> Path`, `slugify(text: str) -> str`, `game_id_for(source: Path, date: str) -> str`; `Manifest` dataclass with fields `game_id, source, date, duration_s, width, height, src_fps, fps, profile, stages: dict, model_versions: dict`, `Manifest.load(game_dir) -> Manifest`, `manifest.save(game_dir) -> Path`, `manifest.mark(stage: str, status: str, **info) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_paths.py`:
```python
from pathlib import Path

from film.common import paths


def test_games_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "g"))
    assert paths.games_dir() == (tmp_path / "g").resolve()
    assert paths.game_dir("abc") == (tmp_path / "g" / "abc").resolve()


def test_slugify():
    assert paths.slugify("JV vs Marcos - Tactical") == "jv-vs-marcos-tactical"
    assert paths.slugify("   ") == "game"


def test_game_id_for():
    src = Path("/x/JV vs Marcos - Tactical.mp4")
    assert paths.game_id_for(src, "2025-02-05") == "2025-02-05-jv-vs-marcos-tactical"
```

`tests/test_manifest.py`:
```python
from film.common.manifest import Manifest


def _manifest():
    return Manifest(
        game_id="2025-02-05-test", source="/x/test.mp4", date="2025-02-05",
        duration_s=3.0, width=320, height=240, src_fps=30.0, fps=10.0, profile="unknown",
    )


def test_roundtrip(tmp_path):
    m = _manifest()
    m.mark("ingest", "done", samples=5)
    m.model_versions["rim"] = "rim@v1"
    path = m.save(tmp_path / "game")
    assert path.name == "manifest.json"
    loaded = Manifest.load(tmp_path / "game")
    assert loaded == m
    assert loaded.stages["ingest"]["status"] == "done"
    assert loaded.stages["ingest"]["samples"] == 5
    assert "at" in loaded.stages["ingest"]
```

`tests/test_doctor.py`:
```python
from film.cli import main


def test_doctor_runs(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "device" in out and "ffmpeg" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_paths.py tests/test_manifest.py tests/test_doctor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.common.paths'`.

- [ ] **Step 3: Write the implementation**

`film/common/paths.py`:
```python
from __future__ import annotations

import os
import re
from pathlib import Path


def _root(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).resolve()


def games_dir() -> Path:
    return _root("FILM_GAMES_DIR", "games")


def models_dir() -> Path:
    return _root("FILM_MODELS_DIR", "models")


def datasets_dir() -> Path:
    return _root("FILM_DATASETS_DIR", "datasets")


def labels_dir() -> Path:
    return _root("FILM_LABELS_DIR", "labels")


def game_dir(game_id: str) -> Path:
    return games_dir() / game_id


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "game"


def game_id_for(source: Path, date: str) -> str:
    return f"{date}-{slugify(source.stem)}"
```

`film/common/manifest.py`:
```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Manifest:
    game_id: str
    source: str
    date: str
    duration_s: float
    width: int
    height: int
    src_fps: float
    fps: float
    profile: str
    stages: dict = field(default_factory=dict)
    model_versions: dict = field(default_factory=dict)

    @classmethod
    def load(cls, game_dir: Path) -> "Manifest":
        return cls(**json.loads((Path(game_dir) / "manifest.json").read_text()))

    def save(self, game_dir: Path) -> Path:
        game_dir = Path(game_dir)
        game_dir.mkdir(parents=True, exist_ok=True)
        path = game_dir / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    def mark(self, stage: str, status: str, **info) -> None:
        entry = {"status": status, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        entry.update(info)
        self.stages[stage] = entry
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_paths.py tests/test_manifest.py tests/test_doctor.py -v`
Expected: 5 passed. Also run `uv run film doctor` and confirm it prints `torch ... mps=True` on this Mac.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "Paths, the game manifest, and a working film doctor

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Video probe and on-the-fly frame iteration

**Files:**
- Create: `film/common/video.py`
- Test: `tests/conftest.py`, `tests/test_video.py`

**Interfaces:**
- Produces: `VideoInfo(duration_s: float, width: int, height: int, fps: float, codec: str)`; `probe(path: Path) -> VideoInfo`; `iter_frames(path: Path, fps: float, start: float = 0.0, end: float | None = None) -> Iterator[tuple[float, np.ndarray]]` yielding `(t_seconds, frame_bgr)` with frames spaced `1/fps` apart.
- Test fixtures: `make_test_video(path, seconds, fps, size, painter)` and the session fixture `synthetic_video` (3 s, 30 fps, 320x240, a white bar sweeping left to right).

- [ ] **Step 1: Write the test helpers and failing tests**

`tests/conftest.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Callable

import av
import numpy as np
import pytest


def make_test_video(
    path: Path,
    seconds: int = 3,
    fps: int = 30,
    size: tuple[int, int] = (320, 240),
    painter: Callable[[int, np.ndarray], None] | None = None,
) -> Path:
    """Encode a small mpeg4 video. `painter(i, img)` draws frame i in place; default draws a sweeping bar."""
    width, height = size
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    for i in range(seconds * fps):
        img = np.zeros((height, width, 3), np.uint8)
        if painter is None:
            x = (i * 5) % width
            img[:, x : x + 10] = 255
        else:
            painter(i, img)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    return make_test_video(tmp_path_factory.mktemp("video") / "synthetic.mp4")


@pytest.fixture
def video_factory(tmp_path):
    def _make(name="v.mp4", **kwargs) -> Path:
        return make_test_video(tmp_path / name, **kwargs)

    return _make
```

`tests/test_video.py`:
```python
import numpy as np

from film.common.video import iter_frames, probe


def test_probe(synthetic_video):
    info = probe(synthetic_video)
    assert info.width == 320 and info.height == 240
    assert abs(info.fps - 30) < 0.5
    assert 2.8 <= info.duration_s <= 3.2
    assert info.codec


def test_iter_frames_spacing(synthetic_video):
    frames = list(iter_frames(synthetic_video, fps=10))
    assert 28 <= len(frames) <= 31
    times = [t for t, _ in frames]
    assert times == sorted(times)
    gaps = np.diff(times)
    assert np.all(gaps > 0.05) and np.all(gaps < 0.15)
    t0, img = frames[0]
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8


def test_iter_frames_window(synthetic_video):
    frames = list(iter_frames(synthetic_video, fps=10, start=1.0, end=2.0))
    assert 9 <= len(frames) <= 12
    assert frames[0][0] >= 0.95
    assert frames[-1][0] <= 2.05
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.common.video'`.

- [ ] **Step 3: Write the implementation**

`film/common/video.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    codec: str


def probe(path: Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(container.duration / av.time_base)
        return VideoInfo(
            duration_s=duration,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(stream.average_rate),
            codec=stream.codec_context.name,
        )


def iter_frames(
    path: Path, fps: float, start: float = 0.0, end: float | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_seconds, frame_bgr) about `fps` times per second of video, decoding on the fly."""
    step = 1.0 / fps
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start > 0:
            container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
        next_t = start
        for frame in container.decode(stream):
            t = frame.time
            if t is None or t < start - 1e-6:
                continue
            if end is not None and t > end + 1e-6:
                break
            if t + 1e-6 >= next_t:
                yield t, frame.to_ndarray(format="bgr24")
                while next_t <= t + 1e-6:
                    next_t += step
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_video.py -v`
Expected: 3 passed. If `probe` returns a duration of 0 for the synthetic file, mpeg4-in-mp4 wrote no stream duration; the `container.duration` branch handles it, so check that branch is reached.

- [ ] **Step 5: Smoke test on real film**

```bash
uv run python -c "
from pathlib import Path; from film.common.video import probe, iter_frames
p = Path.home()/'Downloads'/'JV vs Marcos - Tactical.mp4'
print(probe(p))
import time; t0=time.time(); n=sum(1 for _ in iter_frames(p, fps=10, start=1500, end=1530)); print(n, 'frames in', round(time.time()-t0,1), 's')"
```
Expected: `VideoInfo(duration_s≈5404, width=1920, height=1080, fps≈29.97, codec='h264')` and about 300 frames in a few seconds.

- [ ] **Step 6: Commit**

```bash
git add film/common/video.py tests/conftest.py tests/test_video.py
git commit -m "Probe and stream video frames with PyAV

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Camera profile detection and `film ingest`

**Files:**
- Create: `film/ingest/__init__.py`, `film/ingest/profile.py`, `film/ingest/run.py`, `film/commands/ingest.py`
- Modify: `film/cli.py` (add `"film.commands.ingest"` to `COMMANDS`)
- Test: `tests/test_profile.py`, `tests/test_ingest.py`

**Interfaces:**
- Consumes: `probe`, `iter_frames` (Task 3); `Manifest`, `game_dir`, `game_id_for` (Task 2).
- Produces: `PROFILES = ("hudl-tactical", "nfhs-auto", "nfhs-broadcast", "handheld")`; `overlay_features(frame_bgr) -> tuple[float, float]` returning `(led_fraction, panel_fraction)` for the bottom-left overlay region; `detect_profile(frames: Sequence[np.ndarray]) -> str` returning `"hudl-tactical"` or `"unknown"`; `ingest(source: Path, date: str | None = None, profile: str | None = None, fps: float = 10.0) -> Manifest`.

- [ ] **Step 1: Write the failing tests**

`tests/test_profile.py`:
```python
from pathlib import Path

import cv2

from film.ingest.profile import detect_profile, overlay_features

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"


def load(name: str):
    img = cv2.imread(str(FIXTURES / name))
    assert img is not None, name
    return img


def test_hudl_frames_show_the_overlay():
    for name in ("hudl_tactical_t1500.jpg", "hudl_tactical_t2400.jpg", "hudl_tactical_t4800.jpg"):
        led, panel = overlay_features(load(name))
        assert led > 0.02, (name, led)
        assert panel > 0.35, (name, panel)


def test_other_frames_do_not():
    led, panel = overlay_features(load("nfhs_auto_t1500.jpg"))
    assert not (led > 0.02 and panel > 0.35), (led, panel)
    assert overlay_features(load("nfhs_broadcast_t1500.jpg")) == (0.0, 0.0)  # 1280x720, region out of frame


def test_detect_profile():
    hudl = [load("hudl_tactical_t1500.jpg"), load("hudl_tactical_t2400.jpg"), load("hudl_tactical_t600.jpg")]
    assert detect_profile(hudl) == "hudl-tactical"
    assert detect_profile([load("nfhs_auto_t1500.jpg")]) == "unknown"
    assert detect_profile([]) == "unknown"
```

`tests/test_ingest.py`:
```python
from film.ingest.run import ingest


def test_ingest_writes_manifest(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    assert m.game_id == "2026-01-01-synthetic"
    assert m.profile == "test"
    assert m.width == 320 and m.height == 240
    assert 2.8 <= m.duration_s <= 3.2
    assert m.fps == 10.0
    assert (tmp_path / "games" / m.game_id / "manifest.json").exists()
    assert m.stages["ingest"]["status"] == "done"


def test_ingest_profile_unknown_for_short_video(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01")
    assert m.profile == "unknown"
    assert m.stages["ingest"]["samples"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_profile.py tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.ingest'`.

- [ ] **Step 3: Write the implementation**

`film/ingest/__init__.py`: empty.

`film/ingest/profile.py`:
```python
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

PROFILES = ("hudl-tactical", "nfhs-auto", "nfhs-broadcast", "handheld")

# The Hudl Focus scoreboard overlay, in a 1920x1080 frame: x1, y1, x2, y2.
HUDL_OVERLAY_REGION = (0, 880, 340, 1080)
LED_MIN = 0.02
PANEL_MIN = 0.35


def overlay_features(frame_bgr: np.ndarray, region=HUDL_OVERLAY_REGION) -> tuple[float, float]:
    """(fraction of lit LED-coloured pixels, fraction of dark-blue panel pixels) inside the region."""
    x1, y1, x2, y2 = region
    h, w = frame_bgr.shape[:2]
    if h < y2 or w < x2:
        return 0.0, 0.0
    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    led = (sat > 90) & (val > 140) & ((hue <= 40) | (hue >= 165) | ((hue >= 50) & (hue <= 95)))
    panel = (hue >= 100) & (hue <= 130) & (sat > 100) & (val > 50) & (val < 235)
    return float(led.mean()), float(panel.mean())


def detect_profile(frames: Sequence[np.ndarray]) -> str:
    if not frames:
        return "unknown"
    features = np.array([overlay_features(f) for f in frames])
    led, panel = np.median(features[:, 0]), np.median(features[:, 1])
    if led > LED_MIN and panel > PANEL_MIN:
        return "hudl-tactical"
    return "unknown"
```

`film/ingest/run.py`:
```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from film.common.manifest import Manifest
from film.common.paths import game_dir, game_id_for
from film.common.video import iter_frames, probe
from film.ingest.profile import detect_profile

SAMPLE_TIMES = (60.0, 120.0, 180.0, 240.0, 300.0)


def sample_frames(source: Path, duration_s: float, times=SAMPLE_TIMES) -> list:
    frames = []
    for t in times:
        if t >= duration_s - 1:
            break
        for _, frame in iter_frames(source, fps=2.0, start=t, end=t + 0.6):
            frames.append(frame)
            break
    return frames


def ingest(source: Path, date: str | None = None, profile: str | None = None, fps: float = 10.0) -> Manifest:
    source = Path(source).resolve()
    if date is None:
        date = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d")
    info = probe(source)
    samples = []
    if profile is None:
        samples = sample_frames(source, info.duration_s)
        profile = detect_profile(samples)
    manifest = Manifest(
        game_id=game_id_for(source, date), source=str(source), date=date,
        duration_s=info.duration_s, width=info.width, height=info.height,
        src_fps=info.fps, fps=fps, profile=profile,
    )
    manifest.mark("ingest", "done", samples=len(samples))
    manifest.save(game_dir(manifest.game_id))
    return manifest
```

`film/commands/ingest.py`:
```python
from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("ingest", help="probe a game file, pick its camera profile, write the manifest")
    parser.add_argument("source", type=Path)
    parser.add_argument("--date", help="game date YYYY-MM-DD (default: file modification date)")
    parser.add_argument("--profile", help="override the camera profile (hudl-tactical, nfhs-auto, nfhs-broadcast, handheld)")
    parser.add_argument("--fps", type=float, default=10.0, help="working frame rate (use 5 on the laptop)")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.common.paths import game_dir
    from film.ingest.run import ingest

    m = ingest(args.source, date=args.date, profile=args.profile, fps=args.fps)
    print(f"game_id  {m.game_id}")
    print(f"profile  {m.profile}")
    print(f"duration {m.duration_s:.0f}s  {m.width}x{m.height} @ {m.src_fps:.2f} fps, working {m.fps} fps")
    print(f"manifest {game_dir(m.game_id) / 'manifest.json'}")
    if m.profile == "unknown":
        print("warning: camera profile not recognised; re-run with --profile to set it")
    return 0
```

In `film/cli.py`, change `COMMANDS` to:
```python
COMMANDS = [
    "film.commands.doctor",
    "film.commands.ingest",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_profile.py tests/test_ingest.py -v`
Expected: 5 passed. If `test_other_frames_do_not` fails on `nfhs_auto`, print the two features and raise `PANEL_MIN` until it separates from the three Hudl frames, which sit well above 0.5.

- [ ] **Step 5: Ingest the real game**

```bash
uv run film ingest ~/Downloads/"JV vs Marcos - Tactical.mp4" --date 2025-02-05
```
Expected: `profile  hudl-tactical` and a manifest under `games/2025-02-05-jv-vs-marcos-tactical/`.

- [ ] **Step 6: Commit**

```bash
git add film tests
git commit -m "Ingest: probe the file, detect the Hudl Tactical profile, write the manifest

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Seven-segment decoder, the Hudl scoreboard geometry, and `film calibrate`

**Files:**
- Create: `film/perceive/__init__.py`, `film/perceive/profiles/__init__.py`, `film/perceive/profiles/hudl_tactical.json`, `film/perceive/sevenseg.py`, `film/commands/calibrate.py`
- Modify: `film/cli.py` (add `"film.commands.calibrate"`)
- Test: `tests/test_sevenseg.py`

**Interfaces:**
- Produces: `SEGMENT_ZONES: dict[str, tuple[float, float, float, float]]`, `DIGIT_PATTERNS: dict[str, int]`; `led_mask(frame_bgr, color: str) -> np.ndarray[bool]` with `color in {"red", "amber", "green", "any"}`; `zone_fraction(mask, cell, zone) -> float`; `lit_segments(mask, cell, threshold=0.3) -> str`; `decode_cell(mask, cell, threshold=0.3) -> int | None` (None = blank, -1 = unrecognised); `decode_number(mask, cells, threshold=0.3) -> int | None`; `box_fraction(mask, box) -> float`. A cell is `(x, y, w, h)` in pixels; a box is `(x1, y1, x2, y2)`.
- The profile JSON (read by Task 6) has `fields.<name>.color`, `fields.<name>.cells` as lists of `[x, y, w, h]`, and for `clock` also `sep_top` and `sep_bottom` boxes `[x1, y1, x2, y2]`.

Geometry below was measured on 2026-09-29 from lit-pixel blobs in the three Hudl fixture frames. Digit cells are 23x41 px for scores and clock, 20x34 px for period and fouls.

- [ ] **Step 1: Write the failing tests**

`tests/test_sevenseg.py`:
```python
import json
from pathlib import Path

import cv2
import numpy as np

from film.perceive.sevenseg import (
    DIGIT_PATTERNS, SEGMENT_ZONES, box_fraction, decode_cell, decode_number, led_mask, lit_segments,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"
PROFILE = Path(__file__).resolve().parents[1] / "film" / "perceive" / "profiles" / "hudl_tactical.json"
RED = (0, 0, 255)


def paint(img, cell, segments, color=RED):
    x, y, w, h = cell
    for s in segments:
        zx1, zy1, zx2, zy2 = SEGMENT_ZONES[s]
        cv2.rectangle(img, (int(x + zx1 * w), int(y + zy1 * h)), (int(x + zx2 * w), int(y + zy2 * h)), color, -1)


def test_every_digit_decodes_from_its_pattern():
    cell = (10, 10, 23, 41)
    for pattern, digit in DIGIT_PATTERNS.items():
        img = np.zeros((64, 48, 3), np.uint8)
        paint(img, cell, pattern)
        mask = led_mask(img, "red")
        assert lit_segments(mask, cell) == pattern
        assert decode_cell(mask, cell) == digit


def test_blank_and_unknown_cells():
    cell = (10, 10, 23, 41)
    blank = np.zeros((64, 48, 3), np.uint8)
    assert decode_cell(led_mask(blank, "red"), cell) is None
    odd = np.zeros((64, 48, 3), np.uint8)
    paint(odd, cell, "ag")
    assert decode_cell(led_mask(odd, "red"), cell) == -1


def test_decode_number_is_right_aligned():
    cells = [(10, 10, 23, 41), (40, 10, 23, 41), (70, 10, 23, 41)]
    img = np.zeros((64, 110, 3), np.uint8)
    paint(img, cells[1], "abcdg")   # 3
    paint(img, cells[2], "abdeg")   # 2
    mask = led_mask(img, "red")
    assert decode_number(mask, cells) == 32
    assert decode_number(led_mask(np.zeros_like(img), "red"), cells) is None
    gap = np.zeros((64, 110, 3), np.uint8)
    paint(gap, cells[0], "abcdg")   # a digit followed by a blank is not a number
    assert decode_number(led_mask(gap, "red"), cells) is None


def test_colour_masks_are_selective():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:] = (47, 160, 211)   # BGR of the amber clock digits
    assert led_mask(img, "amber").all()
    assert not led_mask(img, "red").any()
    assert not led_mask(img, "green").any()
    img[:] = (159, 213, 48)   # BGR of the green period digit
    assert led_mask(img, "green").all()


def test_real_frame_home_score_and_period():
    profile = json.loads(PROFILE.read_text())
    frame = cv2.imread(str(FIXTURES / "hudl_tactical_t1500.jpg"))
    home = profile["fields"]["home"]
    period = profile["fields"]["period"]
    assert decode_number(led_mask(frame, home["color"]), [tuple(c) for c in home["cells"]]) == 32
    assert decode_number(led_mask(frame, period["color"]), [tuple(c) for c in period["cells"]]) == 2
    clock = profile["fields"]["clock"]
    amber = led_mask(frame, clock["color"])
    assert box_fraction(amber, clock["sep_top"]) >= 0.2
    assert box_fraction(amber, clock["sep_bottom"]) >= 0.2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sevenseg.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.perceive'`.

- [ ] **Step 3: Write the implementation**

`film/perceive/__init__.py` and `film/perceive/profiles/__init__.py`: empty.

`film/perceive/profiles/hudl_tactical.json`:
```json
{
  "name": "hudl-tactical",
  "frame": [1920, 1080],
  "fields": {
    "home":       {"color": "red",   "cells": [[13, 921, 23, 41], [37, 921, 23, 41], [61, 921, 23, 41]]},
    "away":       {"color": "red",   "cells": [[243, 921, 23, 41], [267, 921, 23, 41], [292, 921, 23, 41]]},
    "clock":      {"color": "amber", "cells": [[109, 897, 23, 41], [133, 897, 23, 41], [188, 897, 23, 41], [212, 897, 23, 41]],
                   "sep_top": [165, 901, 176, 910], "sep_bottom": [165, 923, 176, 932]},
    "period":     {"color": "green", "cells": [[189, 954, 20, 34]]},
    "home_fouls": {"color": "green", "cells": [[45, 1012, 20, 34]]},
    "away_fouls": {"color": "green", "cells": [[296, 1012, 20, 34]]}
  }
}
```

`film/perceive/sevenseg.py`:
```python
from __future__ import annotations

import cv2
import numpy as np

# Where each segment sits inside a digit cell, as fractions of the cell's width and height.
SEGMENT_ZONES: dict[str, tuple[float, float, float, float]] = {
    "a": (0.25, 0.03, 0.75, 0.17),
    "b": (0.76, 0.15, 0.98, 0.47),
    "c": (0.76, 0.53, 0.98, 0.85),
    "d": (0.25, 0.83, 0.75, 0.97),
    "e": (0.02, 0.53, 0.24, 0.85),
    "f": (0.02, 0.15, 0.24, 0.47),
    "g": (0.25, 0.42, 0.75, 0.58),
}
DIGIT_PATTERNS: dict[str, int] = {
    "abcdef": 0, "bc": 1, "abdeg": 2, "abcdg": 3, "bcfg": 4,
    "acdfg": 5, "acdefg": 6, "abc": 7, "abcdefg": 8, "abcdfg": 9,
}
# OpenCV hue runs 0..180.
HUE_RANGES = {"red": [(0, 12), (165, 180)], "amber": [(13, 40)], "green": [(50, 95)]}


def led_mask(frame_bgr: np.ndarray, color: str) -> np.ndarray:
    """Boolean mask of bright, saturated pixels of the given LED colour."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (sat > 90) & (val > 140)
    if color == "any":
        return bright
    in_hue = np.zeros_like(bright)
    for lo, hi in HUE_RANGES[color]:
        in_hue |= (hue >= lo) & (hue <= hi)
    return bright & in_hue


def box_fraction(mask: np.ndarray, box) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    patch = mask[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def zone_fraction(mask: np.ndarray, cell, zone) -> float:
    x, y, w, h = cell
    zx1, zy1, zx2, zy2 = zone
    return box_fraction(mask, (x + zx1 * w, y + zy1 * h, x + zx2 * w, y + zy2 * h))


def lit_segments(mask: np.ndarray, cell, threshold: float = 0.3) -> str:
    return "".join(s for s, zone in SEGMENT_ZONES.items() if zone_fraction(mask, cell, zone) >= threshold)


def decode_cell(mask: np.ndarray, cell, threshold: float = 0.3) -> int | None:
    """The digit in a cell: None when blank, -1 when the lit pattern is not a digit."""
    lit = lit_segments(mask, cell, threshold)
    if not lit:
        return None
    return DIGIT_PATTERNS.get(lit, -1)


def decode_number(mask: np.ndarray, cells, threshold: float = 0.3) -> int | None:
    """Right-aligned digits across cells. None if blank, unreadable, or a blank follows a digit."""
    digits = [decode_cell(mask, cell, threshold) for cell in cells]
    if all(d is None for d in digits) or any(d == -1 for d in digits):
        return None
    value, started = 0, False
    for d in digits:
        if d is None:
            if started:
                return None
            continue
        started = True
        value = value * 10 + d
    return value
```

`film/commands/calibrate.py`:
```python
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("calibrate", help="draw the scoreboard cells on a frame and print what each decodes to")
    parser.add_argument("frame", type=Path, help="a JPEG frame at the profile's native resolution")
    parser.add_argument("--profile", default="hudl-tactical")
    parser.add_argument("--out", type=Path, default=Path("calibrate.png"))
    parser.set_defaults(func=run)


def run(args) -> int:
    import cv2

    from film.perceive.sevenseg import SEGMENT_ZONES, box_fraction, decode_cell, led_mask, lit_segments

    profile = json.loads((files("film.perceive.profiles") / f"{args.profile.replace('-', '_')}.json").read_text())
    frame = cv2.imread(str(args.frame))
    if frame is None:
        print(f"could not read {args.frame}")
        return 1
    canvas = frame.copy()
    for name, field in profile["fields"].items():
        mask = led_mask(frame, field["color"])
        for cell in field["cells"]:
            x, y, w, h = cell
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
            for zone in SEGMENT_ZONES.values():
                zx1, zy1, zx2, zy2 = zone
                cv2.rectangle(canvas, (int(x + zx1 * w), int(y + zy1 * h)), (int(x + zx2 * w), int(y + zy2 * h)), (255, 255, 0), 1)
            print(f"{name:11s} cell {cell}: lit={lit_segments(mask, cell) or '-':8s} digit={decode_cell(mask, cell)}")
        for key in ("sep_top", "sep_bottom"):
            if key in field:
                x1, y1, x2, y2 = field[key]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
                print(f"{name:11s} {key}: fraction={box_fraction(mask, field[key]):.2f}")
    x1, y1, x2, y2 = 0, 860, 360, 1080
    zoom = cv2.resize(canvas[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(args.out), zoom)
    print(f"wrote {args.out}")
    return 0
```

In `film/cli.py`, add `"film.commands.calibrate",` to `COMMANDS` after `"film.commands.ingest",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sevenseg.py -v`
Expected: 5 passed. If `test_real_frame_home_score_and_period` fails, run
`uv run film calibrate fixtures/frames/hudl_tactical_t1500.jpg --out /tmp/cal.png`, open the PNG, and shift the offending cell in the JSON by a pixel or two until the green boxes hug the digits; the `lit=` column shows which segment is being missed.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "Seven-segment decoder and the Hudl scoreboard cell geometry

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Scoreboard reads, the cleaned timeline, and `film scoreboard`

**Files:**
- Create: `film/perceive/scoreboard.py`, `film/commands/scoreboard.py`
- Modify: `film/cli.py` (add `"film.commands.scoreboard"`)
- Test: `tests/test_scoreboard.py`

**Interfaces:**
- Consumes: `led_mask`, `decode_cell`, `decode_number`, `box_fraction` (Task 5); `iter_frames` (Task 3); `Manifest`, `game_dir` (Task 2).
- Produces: `ScoreboardRead` dataclass `(t, period, clock_s, home, away, home_fouls, away_fouls, conf)` with `to_json()`/`from_json()`; `load_profile(name: str) -> dict`; `has_scoreboard(name: str) -> bool`; `read_clock(mask, field) -> float | None`; `read_scoreboard(frame_bgr, profile: dict, t: float = 0.0) -> ScoreboardRead`; `build_timeline(source: Path, profile: dict, limit_s: float | None = None) -> list[ScoreboardRead]` (one read per second); `clean_timeline(reads) -> list[ScoreboardRead]`; `lookup(timeline, t) -> ScoreboardRead | None` (last read with `read.t <= t`); `save_timeline(reads, path)`, `load_timeline(path) -> list[ScoreboardRead]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_scoreboard.py`:
```python
from pathlib import Path

import cv2
import pytest

from film.perceive.scoreboard import (
    ScoreboardRead, clean_timeline, has_scoreboard, load_profile, load_timeline, lookup, read_scoreboard, save_timeline,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frames"
EXPECTED = {
    "hudl_tactical_t1500.jpg": dict(home=32, away=11, clock_s=278.0, period=2, home_fouls=0, away_fouls=0),
    "hudl_tactical_t2400.jpg": dict(home=35, away=23, clock_s=150.0, period=3, home_fouls=0, away_fouls=0),
    "hudl_tactical_t4800.jpg": dict(home=0, away=0, clock_s=0.0, period=1, home_fouls=0, away_fouls=0),
}


@pytest.mark.parametrize("name,expected", EXPECTED.items())
def test_reads_real_frames(name, expected):
    profile = load_profile("hudl-tactical")
    frame = cv2.imread(str(FIXTURES / name))
    read = read_scoreboard(frame, profile, t=1.0)
    for key, value in expected.items():
        assert getattr(read, key) == value, (name, key, getattr(read, key))
    assert read.conf == 1.0


def test_has_scoreboard():
    assert has_scoreboard("hudl-tactical")
    assert not has_scoreboard("handheld")


def R(t, period=1, clock=None, home=0, away=0, conf=1.0):
    return ScoreboardRead(t=t, period=period, clock_s=clock, home=home, away=away, home_fouls=0, away_fouls=0, conf=conf)


def test_clean_drops_clock_going_up_and_score_going_down():
    reads = [R(0, clock=480), R(1, clock=479), R(2, clock=488), R(3, clock=477), R(4, clock=476, home=2), R(5, clock=475, home=0), R(6, clock=474, home=2)]
    kept = clean_timeline(reads)
    assert [r.t for r in kept] == [0, 1, 3, 4, 6]


def test_clean_drops_low_confidence_and_period_going_back():
    reads = [R(0, period=2, clock=100), R(1, period=2, clock=99, conf=0.25), R(2, period=1, clock=98), R(3, period=2, clock=97)]
    assert [r.t for r in clean_timeline(reads)] == [0, 3]


def test_clean_accepts_big_jump_after_a_gap():
    reads = [R(0, clock=100, home=10), R(30, clock=70, home=18)]
    assert len(clean_timeline(reads)) == 2


def test_lookup_and_roundtrip(tmp_path):
    reads = [R(0, clock=100), R(1, clock=99), R(2, clock=98)]
    assert lookup(reads, 1.5).t == 1
    assert lookup(reads, -1) is None
    assert lookup([], 5) is None
    path = tmp_path / "scoreboard.jsonl"
    save_timeline(reads, path)
    assert load_timeline(path) == reads
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_scoreboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.perceive.scoreboard'`.

- [ ] **Step 3: Write the implementation**

`film/perceive/scoreboard.py`:
```python
from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from film.common.video import iter_frames
from film.perceive.sevenseg import box_fraction, decode_cell, decode_number, led_mask

RECENT_S = 10.0       # plausibility checks only apply against a read this recent
MAX_SCORE_JUMP = 3
MIN_CONF = 0.5


@dataclass
class ScoreboardRead:
    t: float
    period: int | None
    clock_s: float | None
    home: int | None
    away: int | None
    home_fouls: int | None
    away_fouls: int | None
    conf: float

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "ScoreboardRead":
        return cls(**data)


def _profile_path(name: str):
    return files("film.perceive.profiles") / f"{name.replace('-', '_')}.json"


def has_scoreboard(name: str) -> bool:
    try:
        return _profile_path(name).is_file()
    except (TypeError, ValueError):
        return False


def load_profile(name: str) -> dict:
    return json.loads(_profile_path(name).read_text())


def _cells(field: dict) -> list[tuple[int, int, int, int]]:
    return [tuple(c) for c in field["cells"]]


def read_clock(mask: np.ndarray, field: dict) -> float | None:
    """Clock cells read as M M : S S when both separator dots are lit, S S . t when only the bottom one is."""
    digits = [decode_cell(mask, cell) for cell in _cells(field)]
    if any(d == -1 for d in digits) or all(d is None for d in digits):
        return None
    d = [0 if x is None else x for x in digits]
    top = box_fraction(mask, field["sep_top"]) >= 0.2
    bottom = box_fraction(mask, field["sep_bottom"]) >= 0.2
    if top and bottom:
        return float(d[0] * 600 + d[1] * 60 + d[2] * 10 + d[3])
    if bottom:
        return float(d[0] * 10 + d[1] + d[2] / 10)
    return None


def read_scoreboard(frame_bgr: np.ndarray, profile: dict, t: float = 0.0) -> ScoreboardRead:
    fields = profile["fields"]
    masks: dict[str, np.ndarray] = {}

    def mask_for(color: str) -> np.ndarray:
        if color not in masks:
            masks[color] = led_mask(frame_bgr, color)
        return masks[color]

    def number(name: str) -> int | None:
        field = fields[name]
        return decode_number(mask_for(field["color"]), _cells(field))

    home, away, period = number("home"), number("away"), number("period")
    clock = read_clock(mask_for(fields["clock"]["color"]), fields["clock"])
    home_fouls = number("home_fouls") if "home_fouls" in fields else None
    away_fouls = number("away_fouls") if "away_fouls" in fields else None
    got = sum(v is not None for v in (home, away, clock, period))
    return ScoreboardRead(t=t, period=period, clock_s=clock, home=home, away=away,
                          home_fouls=home_fouls, away_fouls=away_fouls, conf=got / 4)


def build_timeline(source: Path, profile: dict, limit_s: float | None = None) -> list[ScoreboardRead]:
    return [read_scoreboard(frame, profile, t=t) for t, frame in iter_frames(source, fps=1.0, end=limit_s)]


def _plausible(read: ScoreboardRead, last: ScoreboardRead) -> bool:
    if read.period is not None and last.period is not None and read.period < last.period:
        return False
    if read.t - last.t > RECENT_S:
        return True
    same_period = read.period == last.period
    if same_period and read.clock_s is not None and last.clock_s is not None and read.clock_s > last.clock_s + 0.5:
        return False
    for now, before in ((read.home, last.home), (read.away, last.away)):
        if now is not None and before is not None and (now < before or now - before > MAX_SCORE_JUMP):
            return False
    return True


def clean_timeline(reads: list[ScoreboardRead]) -> list[ScoreboardRead]:
    kept: list[ScoreboardRead] = []
    for read in reads:
        if read.conf < MIN_CONF:
            continue
        if kept and not _plausible(read, kept[-1]):
            continue
        kept.append(read)
    return kept


def lookup(timeline: list[ScoreboardRead], t: float) -> ScoreboardRead | None:
    if not timeline:
        return None
    times = [r.t for r in timeline]
    i = bisect.bisect_right(times, t) - 1
    return timeline[i] if i >= 0 else None


def save_timeline(reads: list[ScoreboardRead], path: Path) -> Path:
    path = Path(path)
    path.write_text("".join(json.dumps(r.to_json()) + "\n" for r in reads))
    return path


def load_timeline(path: Path) -> list[ScoreboardRead]:
    path = Path(path)
    if not path.exists():
        return []
    return [ScoreboardRead.from_json(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
```

`film/commands/scoreboard.py`:
```python
from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("scoreboard", help="read the burned-in scoreboard once a second and write scoreboard.jsonl")
    parser.add_argument("game_id")
    parser.add_argument("--limit-s", type=float, default=None, help="stop after this many seconds of video")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.common.manifest import Manifest
    from film.common.paths import game_dir
    from film.perceive.scoreboard import build_timeline, clean_timeline, has_scoreboard, load_profile, save_timeline

    gd = game_dir(args.game_id)
    m = Manifest.load(gd)
    if not has_scoreboard(m.profile):
        print(f"profile {m.profile} has no scoreboard reader; nothing to do")
        return 1
    raw = build_timeline(Path(m.source), load_profile(m.profile), limit_s=args.limit_s)
    kept = clean_timeline(raw)
    out = save_timeline(kept, gd / "scoreboard.jsonl")
    m.mark("scoreboard", "done", raw=len(raw), kept=len(kept))
    m.save(gd)
    print(f"reads {len(raw)} kept {len(kept)} -> {out}")
    for r in kept[:3] + kept[-3:]:
        print(f"  t={r.t:7.1f} P{r.period} {r.clock_s} {r.home}-{r.away}")
    return 0
```

In `film/cli.py`, add `"film.commands.scoreboard",` to `COMMANDS` after `"film.commands.calibrate",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scoreboard.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run on two minutes of the real game and eyeball it**

```bash
uv run film scoreboard 2025-02-05-jv-vs-marcos-tactical --limit-s 120
```
Expected: about 120 reads, most kept, the clock counting down in the printed lines. The full-game run (about ten minutes of decoding) is not needed here; Task 11 folds scoreboard reading into the single Perceive pass.

- [ ] **Step 6: Commit**

```bash
git add film tests
git commit -m "Scoreboard reads, plausibility cleaning, and the timeline file

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Detector abstraction, model registry, and the tracks file

**Files:**
- Create: `film/common/registry.py`, `film/perceive/detectors.py`, `film/perceive/tracks.py`
- Test: `tests/test_registry.py`, `tests/test_detectors.py`, `tests/test_tracks.py`

**Interfaces:**
- Consumes: `models_dir()` (Task 2).
- Produces (registry): `latest(name: str) -> tuple[str, Path] | None` (highest `vN` folder containing `best.pt`), `new_version_dir(name) -> tuple[str, Path]`, `write_manifest(folder, **info) -> Path`, `read_manifest(folder) -> dict`.
- Produces (detectors): `Box(x1, y1, x2, y2, conf, cls)` frozen dataclass with `w, h, cx, cy` properties and `shifted(dx, dy) -> Box`; `Detector` protocol with `name: str` and `detect(image_bgr) -> list[Box]`; `FakeDetector(per_call=None, fn=None, name="fake@test")`; `boxes_from_result(result, cls_map=None) -> list[Box]`; `YoloDetector(model, device, imgsz, conf=0.25, cls_map=None, name="yolo")` with classmethods `from_weights(weights, device, imgsz, conf=0.25, name=...)` and `open_vocab(classes: dict[str, str], device, imgsz, conf=0.15, name=..., weights=WORLD_WEIGHTS)`; `load_detector(kind: "rim" | "ball", device) -> YoloDetector`; constants `IMGSZ = {"rim": 640, "ball": 320}`, `RIM_WORLD_CLASSES = {"basketball hoop": "rim", "backboard": "backboard"}`, `BALL_WORLD_CLASSES = {"basketball": "ball"}`, `WORLD_WEIGHTS = "yolov8s-worldv2.pt"`. Trained detectors use class names `rim`, `backboard`, `ball`.
- Produces (tracks): `COLUMNS` (the spec's tracks columns plus `ref_id`), `rows_to_frame(rows: list[dict]) -> pd.DataFrame`, `tracks_path(game_dir) -> Path`, `read_tracks(game_dir, kind=None) -> pd.DataFrame`, `replace_kind(game_dir, df_new, kind) -> Path`. Row dicts carry every column; `ball` rows put the rim's `track_id` in `ref_id`.

- [ ] **Step 1: Write the failing tests**

`tests/test_registry.py`:
```python
from film.common import registry


def test_latest_needs_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    assert registry.latest("rim") is None
    v1, d1 = registry.new_version_dir("rim")
    assert v1 == "v1" and d1.is_dir()
    assert registry.latest("rim") is None            # no best.pt yet
    (d1 / "best.pt").write_bytes(b"x")
    assert registry.latest("rim") == ("v1", d1)
    v2, d2 = registry.new_version_dir("rim")
    assert v2 == "v2"
    assert registry.latest("rim") == ("v1", d1)      # v2 has no weights
    (d2 / "best.pt").write_bytes(b"y")
    assert registry.latest("rim") == ("v2", d2)


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    _, d = registry.new_version_dir("ball")
    registry.write_manifest(d, name="ball", map50=0.91)
    info = registry.read_manifest(d)
    assert info["name"] == "ball" and info["map50"] == 0.91 and "written_at" in info
```

`tests/test_detectors.py`:
```python
import types

import numpy as np

from film.perceive import detectors as D
from film.perceive.detectors import Box, FakeDetector, YoloDetector, boxes_from_result, load_detector


def _result(rows, names):
    xyxy = [r[:4] for r in rows]
    return types.SimpleNamespace(
        names=names,
        boxes=types.SimpleNamespace(
            xyxy=types.SimpleNamespace(tolist=lambda: xyxy),
            conf=types.SimpleNamespace(tolist=lambda: [r[4] for r in rows]),
            cls=types.SimpleNamespace(tolist=lambda: [r[5] for r in rows]),
        ),
    )


def test_box_geometry():
    b = Box(10, 20, 30, 60, 0.9, "rim")
    assert (b.w, b.h, b.cx, b.cy) == (20, 40, 20, 40)
    s = b.shifted(5, -5)
    assert (s.x1, s.y1, s.x2, s.y2, s.cls) == (15, 15, 35, 55, "rim")


def test_fake_detector_per_call_and_fn():
    a, b = Box(0, 0, 1, 1, 1, "rim"), Box(1, 1, 2, 2, 1, "rim")
    d = FakeDetector(per_call=[[a], [b]])
    img = np.zeros((4, 4, 3), np.uint8)
    assert d.detect(img) == [a] and d.detect(img) == [b] and d.detect(img) == [b]
    f = FakeDetector(fn=lambda image, i: [a] if i % 2 == 0 else [])
    assert f.detect(img) == [a] and f.detect(img) == [] and f.calls == 2


def test_boxes_from_result_maps_class_names():
    res = _result([[1, 2, 3, 4, 0.8, 0], [5, 6, 7, 8, 0.4, 1]], names={0: "basketball hoop", 1: "backboard"})
    boxes = boxes_from_result(res, cls_map={"basketball hoop": "rim"})
    assert boxes == [Box(1, 2, 3, 4, 0.8, "rim"), Box(5, 6, 7, 8, 0.4, "backboard")]


def test_yolo_detector_uses_model_predict():
    seen = {}

    def predict(image, **kw):
        seen.update(kw)
        return [_result([[0, 0, 10, 10, 0.9, 0]], names={0: "ball"})]

    det = YoloDetector(types.SimpleNamespace(predict=predict), device="cpu", imgsz=320, conf=0.3, name="ball@v1")
    assert det.detect(np.zeros((8, 8, 3), np.uint8)) == [Box(0, 0, 10, 10, 0.9, "ball")]
    assert seen["imgsz"] == 320 and seen["conf"] == 0.3 and seen["device"] == "cpu" and seen["verbose"] is False


def test_load_detector_prefers_trained_then_falls_back(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(D.registry, "latest", lambda name: ("v3", tmp_path) if name == "rim" else None)
    monkeypatch.setattr(YoloDetector, "from_weights", classmethod(lambda cls, w, device, imgsz, conf=0.25, name="": calls.append(("weights", str(w), imgsz, name)) or "trained"))
    monkeypatch.setattr(YoloDetector, "open_vocab", classmethod(lambda cls, classes, device, imgsz, conf=0.15, name="", weights="": calls.append(("world", classes, imgsz, name)) or "world"))
    assert load_detector("rim", "cpu") == "trained"
    assert load_detector("ball", "cpu") == "world"
    assert calls[0] == ("weights", str(tmp_path / "best.pt"), 640, "rim@v3")
    assert calls[1] == ("world", D.BALL_WORLD_CLASSES, 320, "ball@world-fallback")
```

`tests/test_tracks.py`:
```python
import pandas as pd

from film.perceive.tracks import COLUMNS, read_tracks, replace_kind, rows_to_frame


def row(frame, kind, track_id=None, ref_id=None):
    return {"frame": frame, "t": frame / 10, "kind": kind, "track_id": track_id, "ref_id": ref_id,
            "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "conf": 0.5, "on_court": None,
            "court_x": None, "court_y": None, "team": None, "model_version": "fake@test"}


def test_rows_to_frame_has_every_column_and_types():
    df = rows_to_frame([row(0, "rim", track_id=1), {"frame": 1, "t": 0.1, "kind": "ball", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "conf": 0.2}])
    assert list(df.columns) == COLUMNS
    assert df["track_id"].dtype.name == "Int64" and pd.isna(df.loc[1, "track_id"])
    assert df["t"].dtype.name == "float64"
    assert rows_to_frame([]).empty and list(rows_to_frame([]).columns) == COLUMNS


def test_replace_kind_keeps_other_kinds(tmp_path):
    replace_kind(tmp_path, rows_to_frame([row(0, "rim", 1), row(1, "rim", 1)]), "rim")
    replace_kind(tmp_path, rows_to_frame([row(0, "ball", ref_id=1)]), "ball")
    assert len(read_tracks(tmp_path)) == 3
    replace_kind(tmp_path, rows_to_frame([row(5, "rim", 2)]), "rim")
    df = read_tracks(tmp_path)
    assert len(df) == 2 and set(df["kind"]) == {"rim", "ball"}
    assert read_tracks(tmp_path, "rim")["frame"].tolist() == [5]
    assert read_tracks(tmp_path / "missing").empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_registry.py tests/test_detectors.py tests/test_tracks.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`film/common/registry.py`:
```python
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from film.common.paths import models_dir


def _versions(name: str) -> list[Path]:
    root = models_dir() / name
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
    return sorted(dirs, key=lambda p: int(p.name[1:]))


def latest(name: str) -> tuple[str, Path] | None:
    """The newest version folder of a model that actually holds weights."""
    ready = [p for p in _versions(name) if (p / "best.pt").exists()]
    return (ready[-1].name, ready[-1]) if ready else None


def new_version_dir(name: str) -> tuple[str, Path]:
    existing = _versions(name)
    n = int(existing[-1].name[1:]) + 1 if existing else 1
    folder = models_dir() / name / f"v{n}"
    folder.mkdir(parents=True, exist_ok=False)
    return f"v{n}", folder


def write_manifest(folder: Path, **info) -> Path:
    info.setdefault("written_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path = Path(folder) / "manifest.json"
    path.write_text(json.dumps(info, indent=2))
    return path


def read_manifest(folder: Path) -> dict:
    return json.loads((Path(folder) / "manifest.json").read_text())
```

`film/perceive/detectors.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from film.common import registry

IMGSZ = {"rim": 640, "ball": 320}
RIM_WORLD_CLASSES = {"basketball hoop": "rim", "backboard": "backboard"}
BALL_WORLD_CLASSES = {"basketball": "ball"}
WORLD_WEIGHTS = "yolov8s-worldv2.pt"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: str

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def shifted(self, dx: float, dy: float) -> "Box":
        return replace(self, x1=self.x1 + dx, y1=self.y1 + dy, x2=self.x2 + dx, y2=self.y2 + dy)


class Detector(Protocol):
    name: str

    def detect(self, image_bgr: np.ndarray) -> list[Box]: ...


class FakeDetector:
    """Scripted detector for tests: per-call box lists (the last repeats), or a function of (image, call_index)."""

    def __init__(self, per_call: list[list[Box]] | None = None,
                 fn: Callable[[np.ndarray, int], list[Box]] | None = None, name: str = "fake@test"):
        self.per_call = per_call or [[]]
        self.fn = fn
        self.name = name
        self.calls = 0

    def detect(self, image_bgr: np.ndarray) -> list[Box]:
        i = self.calls
        self.calls += 1
        if self.fn is not None:
            return list(self.fn(image_bgr, i))
        return list(self.per_call[min(i, len(self.per_call) - 1)])


def boxes_from_result(result, cls_map: dict[str, str] | None = None) -> list[Box]:
    cls_map = cls_map or {}
    names = result.names
    out = []
    for xyxy, conf, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        raw = str(names[int(k)])
        out.append(Box(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf), cls_map.get(raw, raw)))
    return out


class YoloDetector:
    def __init__(self, model, device: str, imgsz: int, conf: float = 0.25,
                 cls_map: dict[str, str] | None = None, name: str = "yolo"):
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.cls_map = cls_map or {}
        self.name = name

    @classmethod
    def from_weights(cls, weights, device: str, imgsz: int, conf: float = 0.25, name: str = "yolo") -> "YoloDetector":
        from ultralytics import YOLO

        return cls(YOLO(str(weights)), device, imgsz, conf, None, name)

    @classmethod
    def open_vocab(cls, classes: dict[str, str], device: str, imgsz: int, conf: float = 0.15,
                   name: str = "world", weights: str = WORLD_WEIGHTS) -> "YoloDetector":
        """YOLO-World with text prompts; `classes` maps prompt -> our class name. Downloads weights on first use."""
        from ultralytics import YOLOWorld

        model = YOLOWorld(weights)
        model.set_classes(list(classes))
        return cls(model, device, imgsz, conf, dict(classes), name)

    def detect(self, image_bgr: np.ndarray) -> list[Box]:
        results = self.model.predict(image_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        return boxes_from_result(results[0], self.cls_map)


def load_detector(kind: str, device: str) -> YoloDetector:
    """The latest trained detector of this kind, else the open-vocabulary fallback so the pipeline still runs."""
    found = registry.latest(kind)
    if found is not None:
        version, folder = found
        return YoloDetector.from_weights(folder / "best.pt", device, IMGSZ[kind], name=f"{kind}@{version}")
    classes = RIM_WORLD_CLASSES if kind == "rim" else BALL_WORLD_CLASSES
    return YoloDetector.open_vocab(classes, device, IMGSZ[kind], name=f"{kind}@world-fallback")
```

`film/perceive/tracks.py`:
```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["frame", "t", "kind", "track_id", "ref_id", "x1", "y1", "x2", "y2", "conf",
           "on_court", "court_x", "court_y", "team", "model_version"]
FLOAT_COLS = ["t", "x1", "y1", "x2", "y2", "conf", "court_x", "court_y"]
INT_COLS = ["frame", "track_id", "ref_id"]
TEXT_COLS = ["kind", "team", "model_version"]


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["on_court"] = df["on_court"].astype("boolean")
    for col in TEXT_COLS:
        df[col] = df[col].astype("string")
    return df


def tracks_path(game_dir: Path) -> Path:
    return Path(game_dir) / "tracks.parquet"


def read_tracks(game_dir: Path, kind: str | None = None) -> pd.DataFrame:
    path = tracks_path(game_dir)
    df = pd.read_parquet(path) if path.exists() else rows_to_frame([])
    if kind is not None:
        df = df[df["kind"] == kind]
    return df.reset_index(drop=True)


def replace_kind(game_dir: Path, df_new: pd.DataFrame, kind: str) -> Path:
    """Write tracks.parquet with this kind's rows replaced and every other kind kept."""
    existing = read_tracks(game_dir)
    existing = existing[existing["kind"] != kind]
    incoming = rows_to_frame(df_new.to_dict("records"))
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.sort_values(["frame", "kind"], kind="stable").reset_index(drop=True)
    Path(game_dir).mkdir(parents=True, exist_ok=True)
    merged.to_parquet(tracks_path(game_dir), index=False)
    return tracks_path(game_dir)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_registry.py tests/test_detectors.py tests/test_tracks.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "Detector abstraction with an open-vocabulary fallback, model registry, tracks file

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Rim detection with stable ids and smoothing

**Files:**
- Create: `film/perceive/rims.py`
- Test: `tests/test_rims.py`

**Interfaces:**
- Consumes: `Box`, `Detector` (Task 7).
- Produces: `RimTracker(max_gap_frames=30, distance_factor=2.0)` with `update(boxes: list[Box], frame: int) -> list[tuple[int, Box]]`; `detect_rims(frame_bgr, detector, tracker, frame: int, t: float, model_version: str) -> list[dict]` (row dicts, `kind="rim"`, at most two rims, highest confidence first); `smooth_rims(df, window=5) -> pd.DataFrame` (rolling median of the box per `track_id`).

- [ ] **Step 1: Write the failing tests**

`tests/test_rims.py`:
```python
import numpy as np

from film.perceive.detectors import Box, FakeDetector
from film.perceive.rims import RimTracker, detect_rims, smooth_rims
from film.perceive.tracks import rows_to_frame


def rim(x, conf=0.9, cls="rim"):
    return Box(x, 100, x + 50, 130, conf, cls)


def test_tracker_keeps_id_while_panning_and_starts_new_far_away():
    tr = RimTracker()
    assert tr.update([rim(100)], 0) == [(1, rim(100))]
    assert tr.update([rim(130)], 1)[0][0] == 1          # 30 px pan, same rim
    assert tr.update([rim(160), rim(900)], 2) == [(1, rim(160)), (2, rim(900))]
    assert [tid for tid, _ in tr.update([rim(900)], 3)] == [2]


def test_tracker_expires_after_gap():
    tr = RimTracker(max_gap_frames=5)
    tr.update([rim(100)], 0)
    assert tr.update([rim(100)], 10)[0][0] == 2


def test_detect_rims_filters_and_caps():
    boxes = [rim(10, 0.5), rim(200, 0.9), rim(400, 0.7), Box(0, 0, 5, 5, 0.99, "backboard")]
    det = FakeDetector(per_call=[boxes])
    rows = detect_rims(np.zeros((4, 4, 3), np.uint8), det, RimTracker(), frame=3, t=0.3, model_version="rim@v1")
    assert [r["x1"] for r in rows] == [200, 400]
    assert all(r["kind"] == "rim" and r["model_version"] == "rim@v1" and r["frame"] == 3 for r in rows)
    assert rows[0]["track_id"] == 1 and rows[1]["track_id"] == 2


def test_smooth_rims_removes_a_spike():
    rows = []
    for f in range(9):
        b = rim(100 if f != 4 else 300)
        rows.append({"frame": f, "t": f / 10, "kind": "rim", "track_id": 1, "ref_id": None, "x1": b.x1, "y1": b.y1,
                     "x2": b.x2, "y2": b.y2, "conf": 0.9, "on_court": None, "court_x": None, "court_y": None,
                     "team": None, "model_version": "x"})
    df = smooth_rims(rows_to_frame(rows))
    assert df.loc[df["frame"] == 4, "x1"].item() == 100
    assert smooth_rims(rows_to_frame([])).empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rims.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.perceive.rims'`.

- [ ] **Step 3: Write the implementation**

`film/perceive/rims.py`:
```python
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from film.perceive.detectors import Box, Detector

MAX_RIMS = 2


class RimTracker:
    """Stable ids for rim boxes across frames: match by centre distance, expire after a gap.

    Distance rather than overlap, because the Hudl camera pans and a fast pan moves a rim
    by more than its own width between frames at 10 fps.
    """

    def __init__(self, max_gap_frames: int = 30, distance_factor: float = 2.0):
        self.max_gap = max_gap_frames
        self.factor = distance_factor
        self.next_id = 1
        self.active: dict[int, tuple[Box, int]] = {}

    def update(self, boxes: list[Box], frame: int) -> list[tuple[int, Box]]:
        self.active = {tid: (b, f) for tid, (b, f) in self.active.items() if frame - f <= self.max_gap}
        assigned: list[tuple[int, Box]] = []
        used: set[int] = set()
        for box in sorted(boxes, key=lambda b: -b.conf):
            best, best_d = None, math.inf
            for tid, (prev, _) in self.active.items():
                if tid in used:
                    continue
                d = math.hypot(box.cx - prev.cx, box.cy - prev.cy)
                if d < self.factor * max(prev.w, prev.h, 1.0) and d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.active[best] = (box, frame)
            assigned.append((best, box))
        return assigned


def rim_row(frame: int, t: float, track_id: int, box: Box, model_version: str) -> dict:
    return {"frame": frame, "t": t, "kind": "rim", "track_id": track_id, "ref_id": None,
            "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "conf": box.conf,
            "on_court": None, "court_x": None, "court_y": None, "team": None, "model_version": model_version}


def detect_rims(frame_bgr: np.ndarray, detector: Detector, tracker: RimTracker,
                frame: int, t: float, model_version: str) -> list[dict]:
    boxes = [b for b in detector.detect(frame_bgr) if b.cls == "rim"]
    boxes = sorted(boxes, key=lambda b: -b.conf)[:MAX_RIMS]
    return [rim_row(frame, t, tid, box, model_version) for tid, box in tracker.update(boxes, frame)]


def smooth_rims(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["track_id", "frame"]).copy()
    for col in ("x1", "y1", "x2", "y2"):
        df[col] = df.groupby("track_id")[col].transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
    return df.sort_values("frame", kind="stable").reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_rims.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add film/perceive/rims.py tests/test_rims.py
git commit -m "Rim detection with distance-based ids and median smoothing

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Rim crops and ball detection near the rim

**Files:**
- Create: `film/perceive/ball.py`
- Test: `tests/test_ball.py`

**Interfaces:**
- Consumes: `Box`, `Detector` (Task 7).
- Produces: `CROP_SIZE = 320`; `rim_crop_box(rim: Box, frame_w, frame_h, size=CROP_SIZE) -> tuple[int, int, int, int]` (a square crop centred a little above the rim, clamped inside the frame); `crop(frame_bgr, box) -> np.ndarray`; `detect_ball_near_rim(frame_bgr, rims: list[tuple[int, Box]], detector, frame, t, model_version) -> list[dict]` (at most one `ball` row per rim, in full-frame pixels, `ref_id` = the rim's track id).

- [ ] **Step 1: Write the failing tests**

`tests/test_ball.py`:
```python
import numpy as np

from film.perceive.ball import CROP_SIZE, crop, detect_ball_near_rim, rim_crop_box
from film.perceive.detectors import Box, FakeDetector


def test_crop_box_is_square_shifted_up_and_clamped():
    rim = Box(900, 500, 950, 530, 0.9, "rim")
    x1, y1, x2, y2 = rim_crop_box(rim, 1920, 1080)
    assert x2 - x1 == CROP_SIZE and y2 - y1 == CROP_SIZE
    assert x1 + CROP_SIZE / 2 == rim.cx
    assert y1 + CROP_SIZE / 2 < rim.cy               # more room above than below
    corner = rim_crop_box(Box(5, 5, 55, 35, 0.9, "rim"), 1920, 1080)
    assert corner[:2] == (0, 0)
    small = rim_crop_box(rim, 320, 240)
    assert small[2] - small[0] == 240 and small[3] - small[1] == 240


def test_crop_returns_the_region():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert crop(img, (10, 20, 30, 50)).shape == (30, 20, 3)


def test_detect_ball_maps_back_and_keeps_best():
    rim = Box(900, 500, 950, 530, 0.9, "rim")
    cb = rim_crop_box(rim, 1920, 1080)
    det = FakeDetector(per_call=[[Box(10, 10, 20, 20, 0.4, "ball"), Box(100, 100, 112, 112, 0.8, "ball"), Box(0, 0, 9, 9, 0.99, "rim")]])
    rows = detect_ball_near_rim(np.zeros((1080, 1920, 3), np.uint8), [(7, rim)], det, frame=12, t=1.2, model_version="ball@v1")
    assert len(rows) == 1
    r = rows[0]
    assert (r["x1"], r["y1"]) == (100 + cb[0], 100 + cb[1])
    assert r["kind"] == "ball" and r["ref_id"] == 7 and r["track_id"] is None and r["conf"] == 0.8


def test_no_ball_no_row():
    det = FakeDetector(per_call=[[]])
    assert detect_ball_near_rim(np.zeros((1080, 1920, 3), np.uint8), [(1, Box(900, 500, 950, 530, 0.9, "rim"))], det, 0, 0.0, "x") == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ball.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.perceive.ball'`.

- [ ] **Step 3: Write the implementation**

`film/perceive/ball.py`:
```python
from __future__ import annotations

import numpy as np

from film.perceive.detectors import Box, Detector

CROP_SIZE = 320
UP_SHIFT = 0.15   # fraction of the crop by which its centre sits above the rim


def rim_crop_box(rim: Box, frame_w: int, frame_h: int, size: int = CROP_SIZE) -> tuple[int, int, int, int]:
    """A native-resolution square around the rim with more room above it than below, clamped to the frame."""
    size = min(size, frame_w, frame_h)
    cx, cy = rim.cx, rim.cy - UP_SHIFT * size
    x1 = int(round(cx - size / 2))
    y1 = int(round(cy - size / 2))
    x1 = max(0, min(x1, frame_w - size))
    y1 = max(0, min(y1, frame_h - size))
    return x1, y1, x1 + size, y1 + size


def crop(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return frame_bgr[y1:y2, x1:x2]


def ball_row(frame: int, t: float, rim_id: int, box: Box, model_version: str) -> dict:
    return {"frame": frame, "t": t, "kind": "ball", "track_id": None, "ref_id": rim_id,
            "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "conf": box.conf,
            "on_court": None, "court_x": None, "court_y": None, "team": None, "model_version": model_version}


def detect_ball_near_rim(frame_bgr: np.ndarray, rims: list[tuple[int, Box]], detector: Detector,
                         frame: int, t: float, model_version: str) -> list[dict]:
    h, w = frame_bgr.shape[:2]
    rows = []
    for rim_id, rim in rims:
        cb = rim_crop_box(rim, w, h)
        balls = [b for b in detector.detect(crop(frame_bgr, cb)) if b.cls == "ball"]
        if not balls:
            continue
        best = max(balls, key=lambda b: b.conf).shifted(cb[0], cb[1])
        rows.append(ball_row(frame, t, rim_id, best, model_version))
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ball.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add film/perceive/ball.py tests/test_ball.py
git commit -m "Ball detection inside a native-resolution crop around each rim

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Motion energy at the rim

**Files:**
- Create: `film/perceive/motion.py`
- Test: `tests/test_motion.py`

**Interfaces:**
- Consumes: `Box` (Task 7).
- Produces: `zone_box(rim: Box, frame_w, frame_h) -> tuple[int, int, int, int]` (a square three rim-widths wide, centred on the rim); `gray_tile(frame_bgr, box, size) -> np.ndarray`; `MotionEnergy()` with `update(frame_bgr, rims: list[tuple[int, Box]], frame, t) -> list[dict]` rows `{frame, t, rim_id, energy, energy_global}`. Energy is the mean absolute difference from the previous tile of the same rim; `energy_global` the same for a thumbnail of the whole frame, so camera pans can be discounted.

- [ ] **Step 1: Write the failing tests**

`tests/test_motion.py`:
```python
import numpy as np

from film.perceive.detectors import Box
from film.perceive.motion import MotionEnergy, zone_box


def test_zone_box_is_three_rim_widths():
    rim = Box(900, 500, 950, 530, 0.9, "rim")
    x1, y1, x2, y2 = zone_box(rim, 1920, 1080)
    assert x2 - x1 == 150 and y2 - y1 == 150
    assert (x1 + x2) / 2 == rim.cx and (y1 + y2) / 2 == rim.cy


def test_energy_is_zero_when_static_and_positive_when_something_moves_at_the_rim():
    rim = Box(900, 500, 950, 530, 0.9, "rim")
    base = np.full((1080, 1920, 3), 40, np.uint8)
    me = MotionEnergy()
    first = me.update(base, [(1, rim)], 0, 0.0)
    assert first[0]["energy"] == 0.0 and first[0]["energy_global"] == 0.0
    second = me.update(base.copy(), [(1, rim)], 1, 0.1)
    assert second[0]["energy"] == 0.0
    moved = base.copy()
    moved[505:525, 915:935] = 255                    # a bright blob at the rim
    third = me.update(moved, [(1, rim)], 2, 0.2)
    assert third[0]["energy"] > 5.0
    assert third[0]["energy"] > third[0]["energy_global"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_motion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.perceive.motion'`.

- [ ] **Step 3: Write the implementation**

`film/perceive/motion.py`:
```python
from __future__ import annotations

import cv2
import numpy as np

from film.perceive.detectors import Box

TILE = 96
GLOBAL_SIZE = (96, 54)


def zone_box(rim: Box, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    side = int(round(3 * max(rim.w, rim.h, 8.0)))
    side = min(side, frame_w, frame_h)
    x1 = int(round(rim.cx - side / 2))
    y1 = int(round(rim.cy - side / 2))
    x1 = max(0, min(x1, frame_w - side))
    y1 = max(0, min(y1, frame_h - side))
    return x1, y1, x1 + side, y1 + side


def gray_tile(frame_bgr: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


class MotionEnergy:
    """Mean absolute frame difference in a tight box around each rim, alongside the whole-frame difference."""

    def __init__(self):
        self.prev_rim: dict[int, np.ndarray] = {}
        self.prev_global: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray, rims: list[tuple[int, Box]], frame: int, t: float) -> list[dict]:
        h, w = frame_bgr.shape[:2]
        whole = gray_tile(frame_bgr, (0, 0, w, h), GLOBAL_SIZE).astype(np.int16)
        energy_global = float(np.abs(whole - self.prev_global).mean()) if self.prev_global is not None else 0.0
        self.prev_global = whole
        rows = []
        for rim_id, rim in rims:
            tile = gray_tile(frame_bgr, zone_box(rim, w, h), (TILE, TILE)).astype(np.int16)
            prev = self.prev_rim.get(rim_id)
            energy = float(np.abs(tile - prev).mean()) if prev is not None else 0.0
            self.prev_rim[rim_id] = tile
            rows.append({"frame": frame, "t": t, "rim_id": rim_id, "energy": energy, "energy_global": energy_global})
        return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add film/perceive/motion.py tests/test_motion.py
git commit -m "Motion energy at each rim as the fallback shot signal

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: The single streaming Perceive pass and `film perceive`

**Files:**
- Create: `film/perceive/run.py`, `film/commands/perceive.py`
- Modify: `film/cli.py` (add `"film.commands.perceive"`)
- Test: `tests/test_perceive_run.py`

**Interfaces:**
- Consumes: everything from Tasks 3 to 10.
- Produces: `perceive(game_id: str, device: str | None = None, rim_detector=None, ball_detector=None, limit_s: float | None = None, log=print) -> dict` which writes `tracks.parquet` (rim rows smoothed, ball rows), `motion.parquet` (columns `frame, t, rim_id, energy, energy_global`), `scoreboard.jsonl` when the profile has a reader, and marks the `perceive` stage with `frames, rims, balls, scoreboard, seconds, device`.

- [ ] **Step 1: Write the failing test**

`tests/test_perceive_run.py`:
```python
import pandas as pd

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.perceive.detectors import Box, FakeDetector
from film.perceive.run import perceive
from film.perceive.tracks import read_tracks


def test_perceive_writes_tracks_motion_and_marks_stage(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    rim = Box(100, 60, 150, 90, 0.9, "rim")
    rims = FakeDetector(fn=lambda img, i: [rim], name="rim@fake")
    balls = FakeDetector(fn=lambda img, i: [Box(10, 10, 20, 20, 0.7, "ball")] if i % 3 == 0 else [], name="ball@fake")
    info = perceive(m.game_id, device="cpu", rim_detector=rims, ball_detector=balls, log=lambda *a: None)
    gd = game_dir(m.game_id)
    rim_rows = read_tracks(gd, "rim")
    ball_rows = read_tracks(gd, "ball")
    assert 28 <= len(rim_rows) <= 31
    assert rim_rows["track_id"].nunique() == 1
    assert 9 <= len(ball_rows) <= 11
    assert (ball_rows["ref_id"] == 1).all()
    assert ball_rows["x1"].iloc[0] == 15 and ball_rows["y1"].iloc[0] == 10   # crop origin is (5, 0), see note
    motion = pd.read_parquet(gd / "motion.parquet")
    assert list(motion.columns) == ["frame", "t", "rim_id", "energy", "energy_global"]
    assert len(motion) == len(rim_rows)
    assert not (gd / "scoreboard.jsonl").exists()
    saved = Manifest.load(gd)
    assert saved.stages["perceive"]["status"] == "done"
    assert saved.model_versions == {"rim": "rim@fake", "ball": "ball@fake"}
    assert info["frames"] == len(rim_rows)
```
Note on the crop origin: the synthetic frame is 320x240, so the crop is 240 px square. The rim's centre is (125, 75), the crop centre sits 0.15 x 240 = 36 px above it, so the unclamped origin is (5, -81), which clamps to (5, 0). The fake ball box at (10, 10) therefore lands at (15, 10) in frame coordinates.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_perceive_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.perceive.run'`.

- [ ] **Step 3: Write the implementation**

`film/perceive/run.py`:
```python
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.video import iter_frames
from film.perceive.ball import detect_ball_near_rim
from film.perceive.detectors import Box, load_detector
from film.perceive.motion import MotionEnergy
from film.perceive.rims import RimTracker, detect_rims, smooth_rims
from film.perceive.scoreboard import clean_timeline, has_scoreboard, load_profile, read_scoreboard, save_timeline
from film.perceive.tracks import replace_kind, rows_to_frame

MOTION_COLUMNS = ["frame", "t", "rim_id", "energy", "energy_global"]


def perceive(game_id: str, device: str | None = None, rim_detector=None, ball_detector=None,
             limit_s: float | None = None, log=print) -> dict:
    """One decode pass: rims, the ball near each rim, motion energy, and the scoreboard once a second."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    device = pick_device(device)
    rim_detector = rim_detector or load_detector("rim", device)
    ball_detector = ball_detector or load_detector("ball", device)
    profile = load_profile(m.profile) if has_scoreboard(m.profile) else None
    tracker, motion = RimTracker(), MotionEnergy()
    rim_rows, ball_rows, motion_rows, reads = [], [], [], []
    last_second, n_frames, started = -1, 0, time.time()
    report_every = max(1, int(60 * m.fps))
    log(f"perceive {game_id}: rims={rim_detector.name} ball={ball_detector.name} device={device} scoreboard={'yes' if profile else 'no'}")
    for t, frame in iter_frames(Path(m.source), fps=m.fps, end=limit_s):
        fi = int(round(t * m.fps))
        n_frames += 1
        rims = detect_rims(frame, rim_detector, tracker, fi, t, rim_detector.name)
        rim_rows.extend(rims)
        pairs = [(r["track_id"], Box(r["x1"], r["y1"], r["x2"], r["y2"], r["conf"], "rim")) for r in rims]
        ball_rows.extend(detect_ball_near_rim(frame, pairs, ball_detector, fi, t, ball_detector.name))
        motion_rows.extend(motion.update(frame, pairs, fi, t))
        if profile is not None and int(t) != last_second:
            last_second = int(t)
            reads.append(read_scoreboard(frame, profile, t=t))
        if n_frames % report_every == 0:
            elapsed = time.time() - started
            log(f"  t={t:7.1f}s  {elapsed:6.0f}s elapsed  {n_frames / elapsed:5.1f} frames/s")
    replace_kind(gd, smooth_rims(rows_to_frame(rim_rows)), "rim")
    replace_kind(gd, rows_to_frame(ball_rows), "ball")
    pd.DataFrame(motion_rows, columns=MOTION_COLUMNS).to_parquet(gd / "motion.parquet", index=False)
    n_scoreboard = 0
    if profile is not None:
        kept = clean_timeline(reads)
        save_timeline(kept, gd / "scoreboard.jsonl")
        n_scoreboard = len(kept)
    m.model_versions.update({"rim": rim_detector.name, "ball": ball_detector.name})
    m.mark("perceive", "done", frames=n_frames, rims=len(rim_rows), balls=len(ball_rows),
           scoreboard=n_scoreboard, seconds=round(time.time() - started, 1), device=device)
    m.save(gd)
    log(f"done: {n_frames} frames, {len(rim_rows)} rim rows, {len(ball_rows)} ball rows, {n_scoreboard} scoreboard reads in {m.stages['perceive']['seconds']}s")
    return m.stages["perceive"]
```

`film/commands/perceive.py`:
```python
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("perceive", help="one pass over the film: rims, ball, motion energy, scoreboard")
    parser.add_argument("game_id")
    parser.add_argument("--device", default=None, help="mps, cuda or cpu (default: auto)")
    parser.add_argument("--limit-s", type=float, default=None, help="stop after this many seconds of video")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.perceive.run import perceive

    perceive(args.game_id, device=args.device, limit_s=args.limit_s)
    return 0
```

In `film/cli.py`, add `"film.commands.perceive",` to `COMMANDS` after `"film.commands.scoreboard",`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_perceive_run.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the fallback detectors on two minutes of real film**

```bash
uv run film perceive 2025-02-05-jv-vs-marcos-tactical --limit-s 120
```
The first run downloads `yolov8s-worldv2.pt` and the CLIP text encoder that YOLO-World needs (Ultralytics installs it from GitHub on first `set_classes`; this is the one step that needs the internet and `git`). Expected: a progress line per minute of video, then a `done:` line with a rim row count near the frame count (a rim is visible most of the time) and some ball rows. Check the frames-per-second figure; on the M3 laptop with YOLO-World expect single digits, which is why the trained `yolo11n` models in Task 16 matter.

- [ ] **Step 6: Commit**

```bash
git add film tests
git commit -m "Perceive: one streaming pass for rims, ball, motion energy and scoreboard

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Shot arrivals from ball tracks, with motion-energy fallback

**Files:**
- Create: `film/interpret/__init__.py`, `film/interpret/arrivals.py`
- Test: `tests/test_arrivals.py`

**Interfaces:**
- Consumes: tracks frames from Task 7 (`rim` rows with `track_id`, `ball` rows with `ref_id`), motion rows from Task 10.
- Produces: `Arrival(t, frame, rim_id, rim_box: tuple[float, float, float, float], source: "ball" | "motion", conf)`; `find_ball_arrivals(ball_df, rim_df, fps, min_gap_s=1.5) -> list[Arrival]`; `find_motion_arrivals(motion_df, rim_df, fps, z=3.0, min_gap_s=1.5) -> list[Arrival]`; `merge_arrivals(primary, secondary, window_s=2.0) -> list[Arrival]`.

The ball rule: an arrival is the frame where the ball's centre goes from above the rim's top edge to at or below it, while horizontally within one rim-width of the rim, with the previous ball detection at most three frames earlier. This catches jump shots, layups and putbacks alike, and ignores passes, which never cross the rim line downward near the rim.

- [ ] **Step 1: Write the failing tests**

`tests/test_arrivals.py`:
```python
import numpy as np
import pandas as pd

from film.interpret.arrivals import Arrival, find_ball_arrivals, find_motion_arrivals, merge_arrivals
from film.perceive.tracks import rows_to_frame

RIM = (900.0, 500.0, 950.0, 530.0)
FPS = 10.0


def rim_rows(frames, rim=RIM, track_id=1):
    return [{"frame": f, "t": f / FPS, "kind": "rim", "track_id": track_id, "ref_id": None, "x1": rim[0], "y1": rim[1],
             "x2": rim[2], "y2": rim[3], "conf": 0.9, "on_court": None, "court_x": None, "court_y": None,
             "team": None, "model_version": "x"} for f in frames]


def ball_rows(points, rim_id=1):
    """points: iterable of (frame, cx, cy) ball centres, 12 px boxes."""
    return [{"frame": f, "t": f / FPS, "kind": "ball", "track_id": None, "ref_id": rim_id, "x1": cx - 6, "y1": cy - 6,
             "x2": cx + 6, "y2": cy + 6, "conf": 0.7, "on_court": None, "court_x": None, "court_y": None,
             "team": None, "model_version": "x"} for f, cx, cy in points]


def arc(start_frame, cx=925.0, top=300.0, bottom=560.0, n=12):
    ys = np.linspace(top, bottom, n)
    return [(start_frame + i, cx, float(y)) for i, y in enumerate(ys)]


def test_ball_crossing_the_rim_line_is_one_arrival():
    rims = rows_to_frame(rim_rows(range(0, 60)))
    balls = rows_to_frame(ball_rows(arc(0)))
    found = find_ball_arrivals(balls, rims, FPS)
    assert len(found) == 1
    a = found[0]
    assert a.source == "ball" and a.rim_id == 1 and a.rim_box == RIM
    assert 0.8 <= a.t <= 1.1                       # crosses y=500 between frames 8 and 9


def test_pass_under_the_rim_and_ball_rising_are_not_arrivals():
    rims = rows_to_frame(rim_rows(range(0, 60)))
    under = rows_to_frame(ball_rows([(f, 700.0 + 20 * f, 560.0) for f in range(12)]))
    assert find_ball_arrivals(under, rims, FPS) == []
    rising = rows_to_frame(ball_rows([(f, 925.0, 560.0 - 25 * f) for f in range(12)]))
    assert find_ball_arrivals(rising, rims, FPS) == []
    wide = rows_to_frame(ball_rows([(f, 700.0, y) for f, _, y in arc(0)]))   # far to the left of the rim
    assert find_ball_arrivals(wide, rims, FPS) == []


def test_min_gap_and_repeat_arrivals():
    rims = rows_to_frame(rim_rows(range(0, 100)))
    close = rows_to_frame(ball_rows(arc(0) + arc(8)))          # second crossing within 1.5 s
    assert len(find_ball_arrivals(close, rims, FPS)) == 1
    apart = rows_to_frame(ball_rows(arc(0) + arc(40)))         # 4 s later
    assert len(find_ball_arrivals(apart, rims, FPS)) == 2


def test_detection_gap_breaks_the_crossing():
    rims = rows_to_frame(rim_rows(range(0, 60)))
    pts = arc(0)
    pts = [p for p in pts if p[0] not in (6, 7, 8, 9)]         # ball lost for four frames around the rim line
    assert find_ball_arrivals(rows_to_frame(ball_rows(pts)), rims, FPS) == []


def test_motion_peak_becomes_a_low_confidence_arrival():
    rims = rows_to_frame(rim_rows(range(0, 200)))
    energy = np.full(200, 2.0)
    energy[120] = 40.0
    motion = pd.DataFrame({"frame": range(200), "t": np.arange(200) / FPS, "rim_id": 1, "energy": energy, "energy_global": 1.0})
    found = find_motion_arrivals(motion, rims, FPS)
    assert len(found) == 1 and found[0].source == "motion" and found[0].t == 12.0 and found[0].conf < 0.5


def test_merge_keeps_ball_arrivals_and_adds_distant_motion_ones():
    ball = [Arrival(t=10.0, frame=100, rim_id=1, rim_box=RIM, source="ball", conf=0.8)]
    motion = [Arrival(t=11.0, frame=110, rim_id=1, rim_box=RIM, source="motion", conf=0.4),
              Arrival(t=30.0, frame=300, rim_id=1, rim_box=RIM, source="motion", conf=0.4),
              Arrival(t=11.0, frame=110, rim_id=2, rim_box=RIM, source="motion", conf=0.4)]
    merged = merge_arrivals(ball, motion)
    assert [(a.t, a.rim_id, a.source) for a in merged] == [(10.0, 1, "ball"), (11.0, 2, "motion"), (30.0, 1, "motion")]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_arrivals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.interpret'`.

- [ ] **Step 3: Write the implementation**

`film/interpret/__init__.py`: empty.

`film/interpret/arrivals.py`:
```python
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

MIN_GAP_S = 1.5
MAX_FRAME_GAP = 3
MOTION_Z = 3.0
MERGE_WINDOW_S = 2.0


@dataclass
class Arrival:
    t: float
    frame: int
    rim_id: int
    rim_box: tuple[float, float, float, float]
    source: str
    conf: float


def _rim_lookup(rim_df: pd.DataFrame) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    return {(int(r.frame), int(r.track_id)): (float(r.x1), float(r.y1), float(r.x2), float(r.y2)) for r in rim_df.itertuples()}


def find_ball_arrivals(ball_df: pd.DataFrame, rim_df: pd.DataFrame, fps: float, min_gap_s: float = MIN_GAP_S) -> list[Arrival]:
    """The ball's centre crossing the rim's top edge downward while horizontally over the rim."""
    if ball_df.empty or rim_df.empty:
        return []
    rims = _rim_lookup(rim_df)
    out: list[Arrival] = []
    for rim_id, group in ball_df.sort_values("frame").groupby("ref_id"):
        rim_id = int(rim_id)
        prev: tuple[int, float] | None = None
        last_t = -math.inf
        for b in group.itertuples():
            rim = rims.get((int(b.frame), rim_id))
            if rim is None:
                prev = None
                continue
            rx1, ry1, rx2, _ = rim
            width = max(rx2 - rx1, 1.0)
            cx, cy = (float(b.x1) + float(b.x2)) / 2, (float(b.y1) + float(b.y2)) / 2
            over_rim = rx1 - width <= cx <= rx2 + width
            crossed = prev is not None and int(b.frame) - prev[0] <= MAX_FRAME_GAP and prev[1] < ry1 <= cy
            if crossed and over_rim and float(b.t) - last_t >= min_gap_s:
                out.append(Arrival(t=float(b.t), frame=int(b.frame), rim_id=rim_id, rim_box=rim, source="ball", conf=0.8))
                last_t = float(b.t)
            prev = (int(b.frame), cy)
    return sorted(out, key=lambda a: a.t)


def find_motion_arrivals(motion_df: pd.DataFrame, rim_df: pd.DataFrame, fps: float,
                         z: float = MOTION_Z, min_gap_s: float = MIN_GAP_S) -> list[Arrival]:
    """Local peaks of rim motion energy (above the whole-frame energy) more than z standard deviations out."""
    if motion_df.empty or rim_df.empty:
        return []
    rims = _rim_lookup(rim_df)
    half = max(1, int(round(0.5 * fps)))
    out: list[Arrival] = []
    for rim_id, group in motion_df.sort_values("frame").groupby("rim_id"):
        rim_id = int(rim_id)
        rel = (group["energy"] - group["energy_global"]).to_numpy(dtype=float)
        frames = group["frame"].to_numpy()
        times = group["t"].to_numpy(dtype=float)
        if len(rel) < 3 or rel.std() == 0:
            continue
        threshold = rel.mean() + z * rel.std()
        last_t = -math.inf
        for i in range(len(rel)):
            if rel[i] < threshold or rel[i] < rel[max(0, i - half): i + half + 1].max():
                continue
            if times[i] - last_t < min_gap_s:
                continue
            rim = rims.get((int(frames[i]), rim_id))
            if rim is None:
                continue
            out.append(Arrival(t=float(times[i]), frame=int(frames[i]), rim_id=rim_id, rim_box=rim, source="motion", conf=0.4))
            last_t = float(times[i])
    return sorted(out, key=lambda a: a.t)


def merge_arrivals(primary: list[Arrival], secondary: list[Arrival], window_s: float = MERGE_WINDOW_S) -> list[Arrival]:
    """Every primary arrival, plus secondary ones not within window_s of a primary at the same rim."""
    out = list(primary)
    for s in secondary:
        if any(p.rim_id == s.rim_id and abs(p.t - s.t) <= window_s for p in primary):
            continue
        out.append(s)
    return sorted(out, key=lambda a: (a.t, a.rim_id))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_arrivals.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add film/interpret tests/test_arrivals.py
git commit -m "Shot arrivals: the ball crossing the rim line, with motion peaks as fallback

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Rim montages and the made-or-missed classifier

**Files:**
- Create: `film/train/__init__.py`, `film/train/montage.py`, `film/train/rimclf.py`
- Test: `tests/test_montage.py`, `tests/test_rimclf.py`

**Interfaces:**
- Consumes: `iter_frames` (Task 3), `crop` (Task 9), `registry` (Task 7).
- Produces (montage): `TILE = 96`, `GRID = 3`, `STEP_S = 0.2`, `LEAD_S = 0.4`; `montage_box(rim_box, frame_w, frame_h) -> tuple[int, int, int, int]`; `montage_from_tiles(tiles, tile=TILE, grid=GRID) -> np.ndarray` (288x288x3, pads by repeating the last tile); `build_montage(source, t, rim_box, frame_w, frame_h) -> np.ndarray`.
- Produces (rimclf): `CLASSES = ["made", "missed", "none"]`; `build_model(arch="resnet18", pretrained=True, num_classes=3) -> nn.Module` (`arch="tiny"` is a small CNN for tests); `preprocess(montage_bgr) -> torch.Tensor`; `RimClassifier(model, device, version)` with `predict(montage_bgr) -> tuple[str, float, dict[str, float]]` and classmethod `load(device) -> RimClassifier | None` (latest `rimclf` version in the registry, else None); `FakeClassifier(labels: list[tuple[str, float]])` with the same `predict` and a `version`; `train_rimclf(dataset_dir, out_dir, device, epochs=8, batch=32, lr=1e-3, arch="resnet18", pretrained=True) -> dict` expecting `dataset_dir/{train,val}/{made,missed,none}/*.jpg` and saving `out_dir/best.pt`.

- [ ] **Step 1: Write the failing tests**

`tests/test_montage.py`:
```python
import numpy as np

from film.train.montage import GRID, TILE, build_montage, montage_box, montage_from_tiles


def test_montage_box_is_three_widths_and_looks_below_the_rim():
    x1, y1, x2, y2 = montage_box((900, 500, 950, 530), 1920, 1080)
    assert x2 - x1 == 150 and y2 - y1 == 150
    assert (x1 + x2) / 2 == 925
    assert (y1 + y2) / 2 > 515


def test_montage_from_tiles_pads_by_repeating_the_last_tile():
    tiles = [np.full((TILE, TILE, 3), i, np.uint8) for i in range(4)]
    m = montage_from_tiles(tiles)
    assert m.shape == (TILE * GRID, TILE * GRID, 3)
    assert m[0, 0, 0] == 0 and m[0, TILE, 0] == 1 and m[TILE, 0, 0] == 3 and m[2 * TILE, 2 * TILE, 0] == 3


def test_build_montage_from_video(synthetic_video):
    m = build_montage(synthetic_video, t=1.0, rim_box=(100, 60, 150, 90), frame_w=320, frame_h=240)
    assert m.shape == (TILE * GRID, TILE * GRID, 3) and m.dtype == np.uint8
```

`tests/test_rimclf.py`:
```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from film.train.rimclf import CLASSES, FakeClassifier, RimClassifier, build_model, train_rimclf

COLOURS = {"made": (0, 0, 255), "missed": (0, 255, 0), "none": (255, 0, 0)}


def make_dataset(root: Path, per_split=(6, 2)):
    for split, n in zip(("train", "val"), per_split):
        for cls, colour in COLOURS.items():
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(n):
                img = np.zeros((96, 96, 3), np.uint8)
                img[:] = colour
                cv2.imwrite(str(folder / f"{i}.jpg"), img)


def test_fake_classifier_cycles_labels():
    f = FakeClassifier([("made", 0.9), ("missed", 0.8)])
    assert f.predict(None)[0] == "made" and f.predict(None)[0] == "missed" and f.predict(None)[0] == "missed"


def test_train_tiny_and_predict(tmp_path):
    make_dataset(tmp_path / "ds")
    out = tmp_path / "out"
    info = train_rimclf(tmp_path / "ds", out, device="cpu", epochs=40, batch=8, lr=0.05, arch="tiny", pretrained=False)
    assert info["best_val_acc"] == 1.0 and (out / "best.pt").exists()
    assert info["train_images"] == 18 and info["val_images"] == 6
    import torch

    model = build_model("tiny", pretrained=False)
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    clf = RimClassifier(model, "cpu", version="rimclf@test")
    red = np.zeros((288, 288, 3), np.uint8)
    red[:] = COLOURS["made"]
    label, prob, scores = clf.predict(red)
    assert label == "made" and prob > 0.5 and set(scores) == set(CLASSES)


def test_load_returns_none_without_a_trained_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path))
    assert RimClassifier.load("cpu") is None


def test_train_refuses_wrong_class_folders(tmp_path):
    (tmp_path / "ds" / "train" / "hit").mkdir(parents=True)
    (tmp_path / "ds" / "val" / "hit").mkdir(parents=True)
    with pytest.raises(ValueError):
        train_rimclf(tmp_path / "ds", tmp_path / "out", device="cpu", epochs=1, arch="tiny", pretrained=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_montage.py tests/test_rimclf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.train'`.

- [ ] **Step 3: Write the implementation**

`film/train/__init__.py`: empty.

`film/train/montage.py`:
```python
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
```

`film/train/rimclf.py`:
```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from film.common import registry

CLASSES = ["made", "missed", "none"]
INPUT = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(arch: str = "resnet18", pretrained: bool = True, num_classes: int = len(CLASSES)) -> nn.Module:
    if arch == "tiny":
        return nn.Sequential(nn.Conv2d(3, 8, 5, stride=4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, num_classes))
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def preprocess(montage_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(montage_bgr, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)


class RimClassifier:
    def __init__(self, model: nn.Module, device: str, version: str = "rimclf@test"):
        self.model = model.to(device).eval()
        self.device = device
        self.version = version

    @classmethod
    def load(cls, device: str) -> "RimClassifier | None":
        found = registry.latest("rimclf")
        if found is None:
            return None
        version, folder = found
        info = registry.read_manifest(folder) if (folder / "manifest.json").exists() else {}
        model = build_model(arch=info.get("arch", "resnet18"), pretrained=False)
        model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu"))
        return cls(model, device, version=f"rimclf@{version}")

    @torch.no_grad()
    def predict(self, montage_bgr: np.ndarray) -> tuple[str, float, dict[str, float]]:
        logits = self.model(preprocess(montage_bgr).to(self.device))
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        scores = dict(zip(CLASSES, probs))
        label = max(scores, key=scores.get)
        return label, scores[label], scores


class FakeClassifier:
    """Scripted classifier for tests: returns the given (label, prob) per call, the last one repeating."""

    def __init__(self, labels: list[tuple[str, float]] | None = None, version: str = "rimclf@fake"):
        self.labels = labels or [("made", 0.9)]
        self.calls = 0
        self.version = version

    def predict(self, montage_bgr) -> tuple[str, float, dict[str, float]]:
        label, prob = self.labels[min(self.calls, len(self.labels) - 1)]
        self.calls += 1
        scores = {c: (prob if c == label else (1 - prob) / 2) for c in CLASSES}
        return label, prob, scores


def train_rimclf(dataset_dir: Path, out_dir: Path, device: str, epochs: int = 8, batch: int = 32,
                 lr: float = 1e-3, arch: str = "resnet18", pretrained: bool = True) -> dict:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.Resize((INPUT, INPUT)), transforms.ToTensor(),
                             transforms.Normalize(MEAN.tolist(), STD.tolist())])
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    for split in ("train", "val"):
        found = sorted(p.name for p in (dataset_dir / split).iterdir() if p.is_dir())
        if found != CLASSES:
            raise ValueError(f"{dataset_dir / split} must hold exactly the folders {CLASSES}, found {found}")
    train_ds = datasets.ImageFolder(str(dataset_dir / "train"), transform=tf)
    val_ds = datasets.ImageFolder(str(dataset_dir / "val"), transform=tf)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch)
    model = build_model(arch, pretrained).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, history = -1.0, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(y)
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                n += len(y)
        acc = correct / max(n, 1)
        history.append({"epoch": epoch + 1, "loss": round(total / max(len(train_ds), 1), 4), "val_acc": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_dir / "best.pt")
    return {"arch": arch, "epochs": epochs, "best_val_acc": best_acc, "train_images": len(train_ds),
            "val_images": len(val_ds), "history": history}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_montage.py tests/test_rimclf.py -v`
Expected: 7 passed. `test_train_tiny_and_predict` trains a four-layer network on eighteen solid-colour images; it takes a few seconds. If it lands short of 1.0 accuracy, raise `epochs` to 80 in the test rather than touching the trainer.

- [ ] **Step 5: Commit**

```bash
git add film/train tests/test_montage.py tests/test_rimclf.py
git commit -m "Rim montages and the made-or-missed classifier with a fake for tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Shot events, `interpret`, and `film interpret`

**Files:**
- Create: `film/interpret/events.py`, `film/interpret/run.py`, `film/commands/interpret.py`
- Modify: `film/cli.py` (add `"film.commands.interpret"`)
- Test: `tests/test_events.py`, `tests/test_interpret_run.py`

**Interfaces:**
- Consumes: `Arrival` and finders (Task 12), `ScoreboardRead`, `lookup`, `load_timeline` (Task 6), `build_montage`, `RimClassifier`, `FakeClassifier` (Task 13), tracks (Task 7), `MOTION_COLUMNS` (Task 11).
- Produces: `Event(id, type, t, rim_id, quarter, clock_s, team, made, confidence, clip: {"start", "end"}, signals: dict, flags: list[str], needs_location: bool = True)` with `to_json()`/`from_json()`; `score_change(timeline, t, window_s) -> tuple[int, int] | None`; `event_id(t, rim_id) -> str` (`shot-<t*10 zero-padded 6>-r<rim_id>`); `build_shot_events(arrivals, timeline, clf_preds: list[tuple[str, float] | None], game_id) -> list[Event]`; `save_events(game_dir, events) -> Path`, `load_events(game_dir) -> list[Event]`; `interpret(game_id, classifier=None, use_classifier=True, device=None, log=print) -> list[Event]`.
- Decision rules (from spec section 7): the classifier is the primary made-or-missed signal, the scoreboard the check. Agreement gives 0.95 (made) or 0.9 (missed). Disagreement takes the scoreboard's answer at 0.5 with flag `clf-scoreboard-disagree`. No scoreboard: the classifier alone at `0.7 * prob` with flag `no-scoreboard`. No classifier: the scoreboard alone at 0.6/0.5 with flag `no-classifier`, or `None` at 0.3 when neither exists. A classifier `none` on a motion-only arrival makes the event `type="unknown"`. Motion-only arrivals are capped at 0.5 and flagged `motion-only`. A one-point score change adds flag `free-throw?`. The score window for an arrival ends at the next arrival at the same rim (minus half a second) when that is sooner than six seconds, so a miss followed by a quick putback is not credited with the putback's points.

- [ ] **Step 1: Write the failing tests**

`tests/test_events.py`:
```python
from film.interpret.arrivals import Arrival
from film.interpret.events import Event, build_shot_events, event_id, load_events, save_events, score_change
from film.perceive.scoreboard import ScoreboardRead

RIM = (900.0, 500.0, 950.0, 530.0)


def R(t, home=0, away=0, period=2, clock=300.0):
    return ScoreboardRead(t=t, period=period, clock_s=clock - t, home=home, away=away, home_fouls=0, away_fouls=0, conf=1.0)


def timeline_with_home_score_at(t_score, points=2, length=60):
    return [R(t, home=(points if t >= t_score else 0)) for t in range(length)]


def A(t, source="ball", rim_id=1):
    return Arrival(t=t, frame=int(t * 10), rim_id=rim_id, rim_box=RIM, source=source, conf=0.8 if source == "ball" else 0.4)


def test_event_id_format():
    assert event_id(1500.44, 2) == "shot-015004-r2"


def test_score_change_window():
    tl = timeline_with_home_score_at(14)
    assert score_change(tl, 10.0, 6.0) == (2, 0)
    assert score_change(tl, 10.0, 2.0) == (0, 0)
    assert score_change([], 10.0, 6.0) is None


def test_agreement_gives_high_confidence_and_team():
    tl = timeline_with_home_score_at(13)
    [e] = build_shot_events([A(10.0)], tl, [("made", 0.97)], "g")
    assert e.type == "shot" and e.made is True and e.team == "home" and e.confidence == 0.95 and e.flags == []
    assert e.quarter == 2 and e.clock_s == 290.0
    assert e.clip == {"start": 7.0, "end": 11.5}
    assert e.signals["score_delta"] == [2, 0] and e.signals["points"] == 2
    [m] = build_shot_events([A(30.0)], tl, [("missed", 0.9)], "g")
    assert m.made is False and m.team is None and m.confidence == 0.9


def test_disagreement_takes_the_scoreboard_and_flags():
    tl = timeline_with_home_score_at(13)
    [e] = build_shot_events([A(10.0)], tl, [("missed", 0.8)], "g")
    assert e.made is True and e.confidence == 0.5 and "clf-scoreboard-disagree" in e.flags
    [e2] = build_shot_events([A(30.0)], tl, [("made", 0.8)], "g")
    assert e2.made is False and e2.confidence == 0.5


def test_without_scoreboard_or_classifier():
    [e] = build_shot_events([A(10.0)], [], [("made", 0.8)], "g")
    assert e.made is True and e.confidence == 0.56 and "no-scoreboard" in e.flags and e.quarter is None
    [f] = build_shot_events([A(10.0)], timeline_with_home_score_at(13), [None], "g")
    assert f.made is True and f.confidence == 0.6 and "no-classifier" in f.flags
    [g] = build_shot_events([A(10.0)], [], [None], "g")
    assert g.made is None and g.confidence == 0.3


def test_motion_only_and_none_class():
    tl = timeline_with_home_score_at(13)
    [e] = build_shot_events([A(10.0, source="motion")], tl, [("made", 0.9)], "g")
    assert e.confidence == 0.5 and "motion-only" in e.flags
    [u] = build_shot_events([A(10.0, source="motion")], tl, [("none", 0.9)], "g")
    assert u.type == "unknown" and u.made is None
    [s] = build_shot_events([A(10.0)], tl, [("none", 0.9)], "g")
    assert s.type == "shot" and s.made is True and s.confidence == 0.4 and "clf-none" in s.flags


def test_free_throw_flag_and_putback_window():
    tl = [R(t, home=(1 if t >= 13 else 0)) for t in range(60)]
    [e] = build_shot_events([A(10.0)], tl, [("made", 0.9)], "g")
    assert "free-throw?" in e.flags and e.signals["points"] == 1
    tl2 = timeline_with_home_score_at(15)
    miss, putback = build_shot_events([A(10.0), A(12.0)], tl2, [("missed", 0.9), ("made", 0.9)], "g")
    assert miss.made is False and miss.confidence == 0.9
    assert putback.made is True and putback.confidence == 0.95


def test_save_and_load_roundtrip(tmp_path):
    events = build_shot_events([A(10.0)], timeline_with_home_score_at(13), [("made", 0.9)], "g")
    save_events(tmp_path, events)
    assert load_events(tmp_path) == events
    assert load_events(tmp_path / "missing") == []
```

`tests/test_interpret_run.py`:
```python
import numpy as np

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import load_events
from film.interpret.run import interpret
from film.perceive.tracks import replace_kind, rows_to_frame
from film.train.rimclf import FakeClassifier


def _tracks(gd):
    rim = (100.0, 60.0, 150.0, 90.0)
    rim_rows = [{"frame": f, "t": f / 10, "kind": "rim", "track_id": 1, "ref_id": None, "x1": rim[0], "y1": rim[1],
                 "x2": rim[2], "y2": rim[3], "conf": 0.9, "on_court": None, "court_x": None, "court_y": None,
                 "team": None, "model_version": "x"} for f in range(30)]
    ys = np.linspace(20, 100, 12)
    ball_rows = [{"frame": 4 + i, "t": (4 + i) / 10, "kind": "ball", "track_id": None, "ref_id": 1, "x1": 119.0,
                  "y1": float(y) - 6, "x2": 131.0, "y2": float(y) + 6, "conf": 0.7, "on_court": None,
                  "court_x": None, "court_y": None, "team": None, "model_version": "x"} for i, y in enumerate(ys)]
    replace_kind(gd, rows_to_frame(rim_rows), "rim")
    replace_kind(gd, rows_to_frame(ball_rows), "ball")


def test_interpret_writes_events(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    gd = game_dir(m.game_id)
    _tracks(gd)
    events = interpret(m.game_id, classifier=FakeClassifier([("made", 0.9)]), log=lambda *a: None)
    assert len(events) == 1
    e = events[0]
    assert e.type == "shot" and e.made is True and e.confidence == 0.63 and "no-scoreboard" in e.flags
    assert load_events(gd) == events
    saved = Manifest.load(gd)
    assert saved.stages["interpret"]["status"] == "done" and saved.stages["interpret"]["shots"] == 1
    assert saved.model_versions["rimclf"] == "rimclf@fake"


def test_interpret_without_classifier(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test", fps=10.0)
    _tracks(game_dir(m.game_id))
    [e] = interpret(m.game_id, use_classifier=False, log=lambda *a: None)
    assert e.made is None and e.confidence == 0.3 and "no-classifier" in e.flags
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events.py tests/test_interpret_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.interpret.events'`.

- [ ] **Step 3: Write the implementation**

`film/interpret/events.py`:
```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from film.interpret.arrivals import Arrival
from film.perceive.scoreboard import ScoreboardRead, lookup

CLIP_BEFORE_S = 3.0
CLIP_AFTER_S = 1.5
SCORE_WINDOW_S = 6.0
NEXT_ARRIVAL_MARGIN_S = 0.5


@dataclass
class Event:
    id: str
    type: str
    t: float
    rim_id: int
    quarter: int | None
    clock_s: float | None
    team: str | None
    made: bool | None
    confidence: float
    clip: dict
    signals: dict
    flags: list = field(default_factory=list)
    needs_location: bool = True

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Event":
        return cls(**data)


def event_id(t: float, rim_id: int) -> str:
    return f"shot-{int(round(t * 10)):06d}-r{rim_id}"


def score_change(timeline: list[ScoreboardRead], t: float, window_s: float) -> tuple[int, int] | None:
    """(home delta, away delta) between the read just before t and the read at t + window_s."""
    before = lookup(timeline, t - 1.0)
    after = lookup(timeline, t + window_s)
    if before is None or after is None:
        return None
    if None in (before.home, before.away, after.home, after.away):
        return None
    return after.home - before.home, after.away - before.away


def _window_for(i: int, arrivals: list[Arrival]) -> float:
    this = arrivals[i]
    for nxt in arrivals[i + 1:]:
        if nxt.rim_id == this.rim_id:
            return max(0.5, min(SCORE_WINDOW_S, nxt.t - this.t - NEXT_ARRIVAL_MARGIN_S))
    return SCORE_WINDOW_S


def build_shot_events(arrivals: list[Arrival], timeline: list[ScoreboardRead],
                      clf_preds: list[tuple[str, float] | None], game_id: str) -> list[Event]:
    events: list[Event] = []
    for i, (arrival, pred) in enumerate(zip(arrivals, clf_preds)):
        delta = score_change(timeline, arrival.t, _window_for(i, arrivals))
        d_home, d_away = delta if delta is not None else (0, 0)
        home_scored, away_scored = d_home in (1, 2, 3), d_away in (1, 2, 3)
        score_made = delta is not None and (home_scored != away_scored)
        team = "home" if score_made and home_scored else ("away" if score_made and away_scored else None)
        points = (d_home if home_scored else d_away) if score_made else 0
        read = lookup(timeline, arrival.t)
        flags: list[str] = []
        kind = "shot"
        signals = {"source": arrival.source, "arrival_conf": arrival.conf, "clf": list(pred) if pred else None,
                   "score_delta": list(delta) if delta is not None else None, "points": points}
        if pred is None:
            flags.append("no-classifier")
            if delta is None:
                made, conf = None, 0.3
                flags.append("no-scoreboard")
            else:
                made, conf = score_made, (0.6 if score_made else 0.5)
        else:
            label, prob = pred
            if label == "none":
                flags.append("clf-none")
                if arrival.source == "motion":
                    kind, made, conf = "unknown", None, 0.2
                else:
                    made, conf = (score_made if delta is not None else False), 0.4
            else:
                clf_made = label == "made"
                if delta is None:
                    made, conf = clf_made, 0.7 * prob
                    flags.append("no-scoreboard")
                elif clf_made == score_made:
                    made, conf = clf_made, (0.95 if clf_made else 0.9)
                else:
                    made, conf = score_made, 0.5
                    flags.append("clf-scoreboard-disagree")
        if points == 1:
            flags.append("free-throw?")
        if arrival.source == "motion":
            conf = min(conf, 0.5)
            flags.append("motion-only")
        events.append(Event(
            id=event_id(arrival.t, arrival.rim_id), type=kind, t=arrival.t, rim_id=arrival.rim_id,
            quarter=read.period if read else None, clock_s=read.clock_s if read else None,
            team=team, made=made, confidence=round(conf, 2),
            clip={"start": round(max(0.0, arrival.t - CLIP_BEFORE_S), 2), "end": round(arrival.t + CLIP_AFTER_S, 2)},
            signals=signals, flags=flags,
        ))
    return events


def save_events(game_dir: Path, events: list[Event]) -> Path:
    path = Path(game_dir) / "events.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.to_json() for e in events], indent=1))
    return path


def load_events(game_dir: Path) -> list[Event]:
    path = Path(game_dir) / "events.json"
    if not path.exists():
        return []
    return [Event.from_json(d) for d in json.loads(path.read_text())]
```

`film/interpret/run.py`:
```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.arrivals import find_ball_arrivals, find_motion_arrivals, merge_arrivals
from film.interpret.events import Event, build_shot_events, save_events
from film.perceive.run import MOTION_COLUMNS
from film.perceive.scoreboard import load_timeline
from film.perceive.tracks import read_tracks
from film.train.montage import build_montage
from film.train.rimclf import RimClassifier


def interpret(game_id: str, classifier=None, use_classifier: bool = True, device: str | None = None, log=print) -> list[Event]:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    ball, rims = read_tracks(gd, "ball"), read_tracks(gd, "rim")
    motion_path = gd / "motion.parquet"
    motion = pd.read_parquet(motion_path) if motion_path.exists() else pd.DataFrame(columns=MOTION_COLUMNS)
    arrivals = merge_arrivals(find_ball_arrivals(ball, rims, m.fps), find_motion_arrivals(motion, rims, m.fps))
    timeline = load_timeline(gd / "scoreboard.jsonl")
    clf = classifier
    if clf is None and use_classifier:
        clf = RimClassifier.load(pick_device(device))
    preds: list[tuple[str, float] | None] = []
    for arrival in arrivals:
        if clf is None:
            preds.append(None)
            continue
        montage = build_montage(Path(m.source), arrival.t, arrival.rim_box, m.width, m.height)
        label, prob, _ = clf.predict(montage)
        preds.append((label, prob))
    events = build_shot_events(arrivals, timeline, preds, game_id)
    save_events(gd, events)
    version = clf.version if clf is not None else "none"
    m.model_versions["rimclf"] = version
    m.mark("interpret", "done", arrivals=len(arrivals), events=len(events),
           shots=sum(e.type == "shot" for e in events), classifier=version, scoreboard_reads=len(timeline))
    m.save(gd)
    log(f"interpret {game_id}: {len(arrivals)} arrivals -> {len(events)} events ({m.stages['interpret']['shots']} shots), classifier={version}")
    return events
```

`film/commands/interpret.py`:
```python
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("interpret", help="turn tracks and the scoreboard into shot events")
    parser.add_argument("game_id")
    parser.add_argument("--no-classifier", action="store_true", help="skip the rim classifier even if one is trained")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.interpret.run import interpret

    events = interpret(args.game_id, use_classifier=not args.no_classifier, device=args.device)
    for e in events[:10]:
        print(f"  {e.id}  t={e.t:8.1f}  Q{e.quarter} {e.clock_s}  {e.type:7s} made={e.made} team={e.team} conf={e.confidence} {' '.join(e.flags)}")
    if len(events) > 10:
        print(f"  ... {len(events) - 10} more")
    return 0
```

In `film/cli.py`, add `"film.commands.interpret",` to `COMMANDS` after `"film.commands.perceive",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_events.py tests/test_interpret_run.py -v`
Expected: 10 passed.

- [ ] **Step 5: Run on the two-minute perceive output**

```bash
uv run film interpret 2025-02-05-jv-vs-marcos-tactical
```
Expected: a handful of events with `no-classifier` flags (nothing trained yet) and quarters and clocks filled in from the scoreboard. Sanity check one by opening the film at its `t`.

- [ ] **Step 6: Commit**

```bash
git add film tests
git commit -m "Interpret: shot events from arrivals, the classifier and the scoreboard

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Label datasets, pre-labeling, and the classifier dataset builder

**Files:**
- Create: `film/train/datasets.py`, `film/train/prelabel.py`, `film/train/rimclf_dataset.py`, `film/commands/prelabel.py`
- Modify: `film/cli.py` (add `"film.commands.prelabel"`)
- Test: `tests/test_datasets.py`, `tests/test_prelabel.py`, `tests/test_rimclf_dataset.py`

**Interfaces:**
- Consumes: `datasets_dir`, `labels_dir` (Task 2), `Box`, detectors (Task 7), `rim_crop_box`, `crop` (Task 9), `build_montage` (Task 13), `load_events` (Task 14), `read_tracks` (Task 7).
- Produces (datasets): `CLASS_NAMES = {"rim": ["rim", "backboard"], "ball": ["ball"]}`; pending queue under `labels/<dataset>/pending/<image_id>.json` + `labels/<dataset>/images/<image_id>.jpg` with `write_pending(dataset, image_id, image_bgr, boxes, extra=None) -> Path`, `list_pending(dataset) -> list[str]`, `read_pending(dataset, image_id) -> dict`, `remove_pending(dataset, image_id)`; YOLO datasets under `datasets/<dataset>/{images,labels}/{train,val}` with `yolo_line(cls_idx, box, w, h) -> str`, `data_yaml(dataset) -> Path`, `split_for(index, val_every=5) -> str`, `count_accepted(dataset) -> int`, `accept_boxes(dataset, image_id, boxes) -> str`; class datasets under `datasets/<dataset>/{train,val}/<cls>/` with `store_class_image(dataset, image_id, image_bgr, cls) -> Path` and `accept_class(dataset, image_id, cls) -> str`.
- Produces (prelabel): `sample_times(duration_s, n, margin_s=30.0) -> list[float]`, `first_frame_at(source, t, fps) -> np.ndarray | None`, `prelabel_rims(game_id, n=150, detector=None, device=None, margin_s=30.0, log=print) -> int`, `prelabel_balls(game_id, n=300, rim_detector=None, ball_detector=None, device=None, margin_s=30.0, log=print) -> int` (samples around known arrivals when `events.json` exists, else evenly).
- Produces (rimclf dataset): `rim_box_at(rims_df, frame, rim_id=None, max_gap=10) -> tuple | None`; `build_rimclf_candidates(game_id, n_none=40, seed=0, margin_s=30.0, exclusion_s=3.0, log=print) -> dict` with counts `made_auto`, `none_auto`, `pending`. A shot whose scoreboard points are 2 or 3 is stored straight into `made`; every other event becomes a pending montage with a `suggested` class; random rim montages away from any event become `none`.

- [ ] **Step 1: Write the failing tests**

`tests/test_datasets.py`:
```python
import numpy as np

from film.perceive.detectors import Box
from film.train import datasets as ds


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def test_pending_roundtrip(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    img = np.zeros((100, 200, 3), np.uint8)
    ds.write_pending("rim", "g_00010", img, [Box(10, 20, 30, 40, 0.5, "rim")], extra={"t": 10.0})
    assert ds.list_pending("rim") == ["g_00010"]
    data = ds.read_pending("rim", "g_00010")
    assert data["width"] == 200 and data["height"] == 100 and data["t"] == 10.0
    assert data["boxes"] == [{"cls": "rim", "x1": 10, "y1": 20, "x2": 30, "y2": 40, "conf": 0.5}]
    ds.remove_pending("rim", "g_00010")
    assert ds.list_pending("rim") == [] and ds.list_pending("nothing") == []


def test_yolo_line_and_split():
    assert ds.yolo_line(1, Box(10, 20, 30, 40, 0.5, "backboard"), 200, 100) == "1 0.100000 0.300000 0.100000 0.200000"
    assert [ds.split_for(i) for i in range(6)] == ["val", "train", "train", "train", "train", "val"]


def test_accept_boxes_writes_yolo_dataset(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    img = np.zeros((100, 200, 3), np.uint8)
    for i in range(3):
        ds.write_pending("rim", f"g_{i}", img, [])
    split0 = ds.accept_boxes("rim", "g_0", [Box(10, 20, 30, 40, 0.5, "rim"), Box(0, 0, 5, 5, 0.9, "person")])
    assert split0 == "val"
    root = tmp_path / "datasets" / "rim"
    assert (root / "images" / "val" / "g_0.jpg").exists()
    assert (root / "labels" / "val" / "g_0.txt").read_text() == "0 0.100000 0.300000 0.100000 0.200000\n"
    assert ds.accept_boxes("rim", "g_1", []) == "train"
    assert (root / "labels" / "train" / "g_1.txt").read_text() == ""
    assert ds.count_accepted("rim") == 2 and ds.list_pending("rim") == ["g_2"]
    yaml = ds.data_yaml("rim").read_text()
    assert "train: images/train" in yaml and "0: rim" in yaml and "1: backboard" in yaml


def test_class_dataset_guarantees_val_has_each_class(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    img = np.zeros((288, 288, 3), np.uint8)
    p1 = ds.store_class_image("rimclf", "a", img, "made")
    assert p1.parts[-3:] == ("val", "made", "a.jpg")            # first of its class lands in val
    assert ds.store_class_image("rimclf", "b", img, "made").parts[-3] == "train"
    ds.write_pending("rimclf", "c", img, [], extra={"suggested": "missed"})
    assert ds.accept_class("rimclf", "c", "missed") == "val"
    assert ds.list_pending("rimclf") == []
```

`tests/test_prelabel.py`:
```python
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.perceive.detectors import Box, FakeDetector
from film.train import datasets as ds
from film.train.prelabel import prelabel_balls, prelabel_rims, sample_times


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def test_sample_times():
    assert sample_times(100, 3) == [30.0, 50.0, 70.0]
    assert sample_times(50, 3) == []


def test_prelabel_rims_writes_pending(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    det = FakeDetector(fn=lambda img, i: [Box(100, 60, 150, 90, 0.8, "rim"), Box(0, 0, 9, 9, 0.9, "person")])
    n = prelabel_rims(m.game_id, n=4, detector=det, margin_s=0.5, log=lambda *a: None)
    assert n == 4
    ids = ds.list_pending("rim")
    assert len(ids) == 4 and all(i.startswith(m.game_id) for i in ids)
    assert [b["cls"] for b in ds.read_pending("rim", ids[0])["boxes"]] == ["rim"]


def test_prelabel_balls_samples_around_arrivals(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    ev = Event(id="shot-000010-r1", type="shot", t=1.0, rim_id=1, quarter=None, clock_s=None, team=None, made=None,
               confidence=0.3, clip={"start": 0.0, "end": 2.5}, signals={}, flags=[])
    save_events(game_dir(m.game_id), [ev])
    rims = FakeDetector(fn=lambda img, i: [Box(100, 60, 150, 90, 0.8, "rim")])
    balls = FakeDetector(fn=lambda img, i: [Box(5, 5, 15, 15, 0.6, "ball")])
    n = prelabel_balls(m.game_id, n=10, rim_detector=rims, ball_detector=balls, margin_s=0.5, log=lambda *a: None)
    assert n == 5                                              # five offsets around the one arrival
    data = ds.read_pending("ball", ds.list_pending("ball")[0])
    assert data["width"] == 240 and data["height"] == 240        # the rim crop, not the full frame
    assert data["boxes"][0]["cls"] == "ball"
```

`tests/test_rimclf_dataset.py`:
```python
from film.common.paths import game_dir
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.perceive.tracks import replace_kind, rows_to_frame
from film.train import datasets as ds
from film.train.rimclf_dataset import build_rimclf_candidates, rim_box_at


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def _rims(gd, frames=range(30)):
    rows = [{"frame": f, "t": f / 10, "kind": "rim", "track_id": 1, "ref_id": None, "x1": 100.0, "y1": 60.0, "x2": 150.0,
             "y2": 90.0, "conf": 0.9, "on_court": None, "court_x": None, "court_y": None, "team": None, "model_version": "x"}
            for f in frames]
    replace_kind(gd, rows_to_frame(rows), "rim")


def _event(eid, t, kind="shot", points=0):
    return Event(id=eid, type=kind, t=t, rim_id=1, quarter=None, clock_s=None, team=None, made=None, confidence=0.5,
                 clip={"start": t - 1, "end": t + 1}, signals={"points": points, "score_delta": [points, 0] if points else None}, flags=[])


def test_rim_box_at_uses_nearest_frame(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd, frames=[0, 1, 2, 20])
    from film.perceive.tracks import read_tracks

    rims = read_tracks(gd, "rim")
    assert rim_box_at(rims, 2) == (100.0, 60.0, 150.0, 90.0)
    assert rim_box_at(rims, 6) == (100.0, 60.0, 150.0, 90.0)     # frame 2 is within max_gap
    assert rim_box_at(rims, 40) is None                          # nearest is frame 20, too far
    assert rim_box_at(rims, 20, rim_id=1) == (100.0, 60.0, 150.0, 90.0) and rim_box_at(rims, 20, rim_id=9) is None


def test_build_candidates_routes_made_pending_and_none(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    _rims(gd)
    save_events(gd, [_event("shot-000010-r1", 1.0, points=2), _event("shot-000020-r1", 2.0), _event("shot-000025-r1", 2.5, kind="unknown")])
    counts = build_rimclf_candidates(m.game_id, n_none=2, seed=1, margin_s=0.2, exclusion_s=0.1, log=lambda *a: None)
    assert counts == {"made_auto": 1, "pending": 2, "none_auto": 2}
    root = tmp_path / "datasets" / "rimclf"
    assert (root / "val" / "made" / "shot-000010-r1.jpg").exists()
    assert sorted(p.parent.name for p in root.glob("*/none/*.jpg")) == ["none", "none"]
    pending = {i: ds.read_pending("rimclf", i) for i in ds.list_pending("rimclf")}
    assert pending["shot-000020-r1"]["suggested"] == "missed"
    assert pending["shot-000025-r1"]["suggested"] == "none"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_datasets.py tests/test_prelabel.py tests/test_rimclf_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.train.datasets'`.

- [ ] **Step 3: Write the implementation**

`film/train/datasets.py`:
```python
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from film.common.paths import datasets_dir, labels_dir
from film.perceive.detectors import Box

CLASS_NAMES = {"rim": ["rim", "backboard"], "ball": ["ball"]}
VAL_EVERY = 5


# ---- pending queue: what the labeling page shows ----

def pending_dir(dataset: str) -> Path:
    return labels_dir() / dataset / "pending"


def pending_images_dir(dataset: str) -> Path:
    return labels_dir() / dataset / "images"


def write_pending(dataset: str, image_id: str, image_bgr: np.ndarray, boxes: list[Box], extra: dict | None = None) -> Path:
    pending_dir(dataset).mkdir(parents=True, exist_ok=True)
    pending_images_dir(dataset).mkdir(parents=True, exist_ok=True)
    image_path = pending_images_dir(dataset) / f"{image_id}.jpg"
    cv2.imwrite(str(image_path), np.ascontiguousarray(image_bgr))
    h, w = image_bgr.shape[:2]
    data = {"image_id": image_id, "image": str(image_path), "width": w, "height": h,
            "boxes": [{"cls": b.cls, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "conf": b.conf} for b in boxes]}
    if extra:
        data.update(extra)
    path = pending_dir(dataset) / f"{image_id}.json"
    path.write_text(json.dumps(data, indent=1))
    return path


def list_pending(dataset: str) -> list[str]:
    folder = pending_dir(dataset)
    return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []


def read_pending(dataset: str, image_id: str) -> dict:
    return json.loads((pending_dir(dataset) / f"{image_id}.json").read_text())


def remove_pending(dataset: str, image_id: str) -> None:
    for path in (pending_dir(dataset) / f"{image_id}.json", pending_images_dir(dataset) / f"{image_id}.jpg"):
        if path.exists():
            path.unlink()


# ---- YOLO box datasets ----

def yolo_line(cls_idx: int, box: Box, w: int, h: int) -> str:
    cx, cy = (box.x1 + box.x2) / 2 / w, (box.y1 + box.y2) / 2 / h
    return f"{cls_idx} {cx:.6f} {cy:.6f} {box.w / w:.6f} {box.h / h:.6f}"


def data_yaml(dataset: str) -> Path:
    root = datasets_dir() / dataset
    root.mkdir(parents=True, exist_ok=True)
    names = CLASS_NAMES[dataset]
    text = f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    path = root / "data.yaml"
    path.write_text(text)
    return path


def split_for(index: int, val_every: int = VAL_EVERY) -> str:
    return "val" if index % val_every == 0 else "train"


def count_accepted(dataset: str) -> int:
    root = datasets_dir() / dataset / "images"
    return sum(1 for split in ("train", "val") for _ in (root / split).glob("*.jpg")) if root.exists() else 0


def accept_boxes(dataset: str, image_id: str, boxes: list[Box]) -> str:
    """Move a pending image into the YOLO dataset with its final boxes. Returns the split it went to."""
    data = read_pending(dataset, image_id)
    names = CLASS_NAMES[dataset]
    split = split_for(count_accepted(dataset))
    root = datasets_dir() / dataset
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy(data["image"], root / "images" / split / f"{image_id}.jpg")
    lines = [yolo_line(names.index(b.cls), b, data["width"], data["height"]) for b in boxes if b.cls in names]
    (root / "labels" / split / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    remove_pending(dataset, image_id)
    return split


# ---- class datasets (ImageFolder layout) ----

def _class_split(dataset: str, cls: str) -> str:
    root = datasets_dir() / dataset
    if not any((root / "val" / cls).glob("*.jpg")):
        return "val"
    total = sum(1 for _ in root.glob("*/*/*.jpg"))
    return split_for(total)


def store_class_image(dataset: str, image_id: str, image_bgr: np.ndarray, cls: str) -> Path:
    split = _class_split(dataset, cls)
    folder = datasets_dir() / dataset / split / cls
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{image_id}.jpg"
    cv2.imwrite(str(path), np.ascontiguousarray(image_bgr))
    return path


def accept_class(dataset: str, image_id: str, cls: str) -> str:
    data = read_pending(dataset, image_id)
    image = cv2.imread(data["image"])
    path = store_class_image(dataset, image_id, image, cls)
    remove_pending(dataset, image_id)
    return path.parts[-3]
```

`film/train/prelabel.py`:
```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from film.common.device import pick_device
from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.common.video import iter_frames
from film.interpret.events import load_events
from film.perceive.ball import crop, rim_crop_box
from film.perceive.detectors import IMGSZ, RIM_WORLD_CLASSES, YoloDetector, load_detector
from film.train.datasets import CLASS_NAMES, write_pending

ARRIVAL_OFFSETS_S = (-0.6, -0.4, -0.2, 0.0, 0.2)


def sample_times(duration_s: float, n: int, margin_s: float = 30.0) -> list[float]:
    if n <= 0 or duration_s <= 2 * margin_s:
        return []
    return [float(x) for x in np.linspace(margin_s, duration_s - margin_s, n)]


def first_frame_at(source: Path, t: float, fps: float) -> np.ndarray | None:
    for _, frame in iter_frames(source, fps=fps, start=t, end=t + 1.0):
        return frame
    return None


def prelabel_rims(game_id: str, n: int = 150, detector=None, device: str | None = None,
                  margin_s: float = 30.0, log=print) -> int:
    """Propose rim and backboard boxes on evenly spaced frames for the coach to confirm."""
    m = Manifest.load(game_dir(game_id))
    det = detector or YoloDetector.open_vocab(RIM_WORLD_CLASSES, pick_device(device), IMGSZ["rim"], name="rim@world")
    count = 0
    for t in sample_times(m.duration_s, n, margin_s):
        frame = first_frame_at(Path(m.source), t, m.fps)
        if frame is None:
            continue
        boxes = [b for b in det.detect(frame) if b.cls in CLASS_NAMES["rim"]]
        write_pending("rim", f"{game_id}_{int(round(t * 10)):06d}", frame, boxes, extra={"t": t})
        count += 1
    log(f"prelabel rim: {count} frames queued for {game_id}")
    return count


def prelabel_balls(game_id: str, n: int = 300, rim_detector=None, ball_detector=None, device: str | None = None,
                   margin_s: float = 30.0, log=print) -> int:
    """Propose ball boxes inside rim crops, sampled around known arrivals when events exist."""
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    device = pick_device(device)
    rim_det = rim_detector or load_detector("rim", device)
    ball_det = ball_detector or load_detector("ball", device)
    times = [e.t + d for e in load_events(gd) for d in ARRIVAL_OFFSETS_S]
    if not times:
        times = sample_times(m.duration_s, n, margin_s)
    times = [t for t in times if 0.0 <= t < m.duration_s][:n]
    count = 0
    for t in times:
        frame = first_frame_at(Path(m.source), t, m.fps)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        rims = sorted((b for b in rim_det.detect(frame) if b.cls == "rim"), key=lambda b: -b.conf)[:2]
        for i, rim in enumerate(rims):
            tile = crop(frame, rim_crop_box(rim, w, h))
            boxes = [b for b in ball_det.detect(tile) if b.cls == "ball"]
            write_pending("ball", f"{game_id}_{int(round(t * 10)):06d}_r{i}", tile, boxes, extra={"t": t})
            count += 1
    log(f"prelabel ball: {count} crops queued for {game_id}")
    return count
```

`film/train/rimclf_dataset.py`:
```python
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from film.common.manifest import Manifest
from film.common.paths import game_dir
from film.interpret.events import load_events
from film.perceive.tracks import read_tracks
from film.train.datasets import store_class_image, write_pending
from film.train.montage import build_montage

EVENT_EXCLUSION_S = 3.0


def rim_box_at(rims: pd.DataFrame, frame: int, rim_id: int | None = None, max_gap: int = 10) -> tuple | None:
    """The rim box at the nearest tracked frame within max_gap frames, for a given rim id or any rim."""
    if rims.empty:
        return None
    df = rims if rim_id is None else rims[rims["track_id"] == rim_id]
    if df.empty:
        return None
    gaps = (df["frame"].astype(int) - frame).abs()
    i = gaps.idxmin()
    if int(gaps.loc[i]) > max_gap:
        return None
    row = df.loc[i]
    return float(row.x1), float(row.y1), float(row.x2), float(row.y2)


def build_rimclf_candidates(game_id: str, n_none: int = 40, seed: int = 0, margin_s: float = 30.0,
                            exclusion_s: float = EVENT_EXCLUSION_S, log=print) -> dict:
    gd = game_dir(game_id)
    m = Manifest.load(gd)
    source = Path(m.source)
    events = load_events(gd)
    rims = read_tracks(gd, "rim")
    counts = {"made_auto": 0, "pending": 0, "none_auto": 0}
    for e in events:
        box = rim_box_at(rims, int(round(e.t * m.fps)), e.rim_id)
        if box is None:
            continue
        montage = build_montage(source, e.t, box, m.width, m.height)
        if e.type == "shot" and e.signals.get("points") in (2, 3):
            store_class_image("rimclf", e.id, montage, "made")
            counts["made_auto"] += 1
        else:
            suggested = "missed" if e.type == "shot" else "none"
            write_pending("rimclf", e.id, montage, [], extra={"t": e.t, "suggested": suggested, "event_id": e.id})
            counts["pending"] += 1
    rng = random.Random(seed)
    event_times = [e.t for e in events]
    tries = 0
    while counts["none_auto"] < n_none and tries < n_none * 20 and m.duration_s > 2 * margin_s:
        tries += 1
        t = rng.uniform(margin_s, m.duration_s - margin_s)
        if any(abs(t - et) < exclusion_s for et in event_times):
            continue
        box = rim_box_at(rims, int(round(t * m.fps)))
        if box is None:
            continue
        store_class_image("rimclf", f"none-{game_id}-{int(round(t * 10)):06d}", build_montage(source, t, box, m.width, m.height), "none")
        counts["none_auto"] += 1
    log(f"rimclf candidates for {game_id}: {counts}")
    return counts
```

`film/commands/prelabel.py`:
```python
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("prelabel", help="queue frames, rim crops or rim montages for labeling, with proposed labels")
    parser.add_argument("what", choices=["rim", "ball", "rimclf"])
    parser.add_argument("game_id")
    parser.add_argument("--n", type=int, default=None, help="how many to queue (rim 150, ball 300, rimclf none-samples 40)")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    if args.what == "rim":
        from film.train.prelabel import prelabel_rims

        prelabel_rims(args.game_id, n=args.n or 150, device=args.device)
    elif args.what == "ball":
        from film.train.prelabel import prelabel_balls

        prelabel_balls(args.game_id, n=args.n or 300, device=args.device)
    else:
        from film.train.rimclf_dataset import build_rimclf_candidates

        build_rimclf_candidates(args.game_id, n_none=args.n or 40)
    print("next: uv run film review, then open the Label tab")
    return 0
```

In `film/cli.py`, add `"film.commands.prelabel",` to `COMMANDS` after `"film.commands.interpret",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_datasets.py tests/test_prelabel.py tests/test_rimclf_dataset.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "Label queues, YOLO and class datasets, and pre-labeling with the fallback models

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: Training commands for the detectors and the classifier

**Files:**
- Create: `film/train/train_yolo.py`, `film/commands/train.py`
- Modify: `film/cli.py` (add `"film.commands.train"`)
- Test: `tests/test_train_yolo.py`

**Interfaces:**
- Consumes: `registry` (Task 7), `data_yaml`, `datasets_dir` (Tasks 15, 2), `train_rimclf` (Task 13), `IMGSZ` (Task 7).
- Produces: `BASE_WEIGHTS = {"rim": "yolo11n.pt", "ball": "yolo11n.pt"}`; `dataset_counts(name) -> dict[str, int]`; `train_detector(name, device, epochs=50, batch=16, yolo_cls=None, log=print) -> tuple[str, Path]` writing `models/<name>/vN/best.pt` and a manifest with `map50`; `class_counts(dataset) -> dict[str, dict[str, int]]`; `train_rimclf_version(device, epochs=8, log=print) -> tuple[str, Path]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_train_yolo.py`:
```python
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from film.common import registry
from film.train.train_yolo import class_counts, dataset_counts, train_detector, train_rimclf_version


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))


def _yolo_dataset(root: Path, train=24, val=6):
    for split, n in (("train", train), ("val", val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


class StubYOLO:
    created = []

    def __init__(self, weights):
        StubYOLO.created.append(weights)
        self.kwargs = None

    def train(self, **kwargs):
        self.kwargs = kwargs
        weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"weights")
        return types.SimpleNamespace(box=types.SimpleNamespace(map50=0.91))


def test_train_detector_registers_a_version(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    _yolo_dataset(tmp_path / "datasets" / "rim")
    assert dataset_counts("rim") == {"train": 24, "val": 6}
    version, folder = train_detector("rim", device="cpu", epochs=3, yolo_cls=StubYOLO, log=lambda *a: None)
    assert version == "v1" and (folder / "best.pt").read_bytes() == b"weights"
    assert StubYOLO.created[-1] == "yolo11n.pt"
    info = registry.read_manifest(folder)
    assert info["map50"] == 0.91 and info["epochs"] == 3 and info["imgsz"] == 640 and info["train_images"] == 24
    assert registry.latest("rim") == ("v1", folder)


def test_train_detector_refuses_tiny_dataset(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    _yolo_dataset(tmp_path / "datasets" / "ball", train=5, val=1)
    with pytest.raises(ValueError):
        train_detector("ball", device="cpu", yolo_cls=StubYOLO)


def test_train_rimclf_version(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    root = tmp_path / "datasets" / "rimclf"
    colours = {"made": (0, 0, 255), "missed": (0, 255, 0), "none": (255, 0, 0)}
    for split, n in (("train", 6), ("val", 2)):
        for cls, colour in colours.items():
            (root / split / cls).mkdir(parents=True)
            for i in range(n):
                img = np.zeros((96, 96, 3), np.uint8)
                img[:] = colour
                cv2.imwrite(str(root / split / cls / f"{i}.jpg"), img)
    assert class_counts("rimclf") == {"train": {"made": 6, "missed": 6, "none": 6}, "val": {"made": 2, "missed": 2, "none": 2}}
    version, folder = train_rimclf_version("cpu", epochs=40, arch="tiny", pretrained=False, lr=0.05, batch=8, log=lambda *a: None)
    assert version == "v1" and (folder / "best.pt").exists()
    assert registry.read_manifest(folder)["arch"] == "tiny"
    assert registry.latest("rimclf") == ("v1", folder)


def test_train_rimclf_refuses_empty_class(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    root = tmp_path / "datasets" / "rimclf"
    for split in ("train", "val"):
        for cls in ("made", "missed", "none"):
            (root / split / cls).mkdir(parents=True)
    cv2.imwrite(str(root / "train" / "made" / "0.jpg"), np.zeros((96, 96, 3), np.uint8))
    with pytest.raises(ValueError, match="missed"):
        train_rimclf_version("cpu", epochs=1, arch="tiny", pretrained=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train_yolo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.train.train_yolo'`.

- [ ] **Step 3: Write the implementation**

`film/train/train_yolo.py`:
```python
from __future__ import annotations

import shutil
from pathlib import Path

from film.common import registry
from film.common.paths import datasets_dir
from film.perceive.detectors import IMGSZ
from film.train.datasets import data_yaml
from film.train.rimclf import CLASSES, train_rimclf

BASE_WEIGHTS = {"rim": "yolo11n.pt", "ball": "yolo11n.pt"}
MIN_TRAIN, MIN_VAL = 20, 4


def dataset_counts(name: str) -> dict[str, int]:
    root = datasets_dir() / name / "images"
    return {split: (sum(1 for _ in (root / split).glob("*.jpg")) if (root / split).exists() else 0) for split in ("train", "val")}


def _metric(results, key: str):
    box = getattr(results, "box", None)
    value = getattr(box, key, None)
    return float(value) if value is not None else None


def train_detector(name: str, device: str, epochs: int = 50, batch: int = 16, yolo_cls=None, log=print) -> tuple[str, Path]:
    """Fine-tune a small YOLO on datasets/<name> and register the result as models/<name>/vN."""
    counts = dataset_counts(name)
    if counts["train"] < MIN_TRAIN or counts["val"] < MIN_VAL:
        raise ValueError(f"{name}: need at least {MIN_TRAIN} train and {MIN_VAL} val images, have {counts}; label more first")
    yaml_path = data_yaml(name)
    version, out = registry.new_version_dir(name)
    if yolo_cls is None:
        from ultralytics import YOLO as yolo_cls
    log(f"train {name} {version}: {counts} images, {epochs} epochs at {IMGSZ[name]} px on {device}")
    model = yolo_cls(BASE_WEIGHTS[name])
    results = model.train(data=str(yaml_path), epochs=epochs, imgsz=IMGSZ[name], device=device, batch=batch,
                          project=str(out), name="train", exist_ok=True, verbose=False, plots=False)
    best = out / "train" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training produced no weights at {best}")
    shutil.copy(best, out / "best.pt")
    registry.write_manifest(out, name=name, version=version, base=BASE_WEIGHTS[name], epochs=epochs, imgsz=IMGSZ[name],
                            device=device, map50=_metric(results, "map50"), train_images=counts["train"], val_images=counts["val"])
    log(f"registered {name} {version}: map50={_metric(results, 'map50')}")
    return version, out


def class_counts(dataset: str) -> dict[str, dict[str, int]]:
    root = datasets_dir() / dataset
    return {split: {cls: sum(1 for _ in (root / split / cls).glob("*.jpg")) for cls in CLASSES} for split in ("train", "val")}


def train_rimclf_version(device: str, epochs: int = 8, log=print, **kwargs) -> tuple[str, Path]:
    counts = class_counts("rimclf")
    empty = [f"{split}/{cls}" for split, per in counts.items() for cls, n in per.items() if n == 0]
    if empty:
        raise ValueError(f"rimclf: these class folders are empty: {', '.join(empty)}; label more first")
    version, out = registry.new_version_dir("rimclf")
    log(f"train rimclf {version}: {counts}, {epochs} epochs on {device}")
    info = train_rimclf(datasets_dir() / "rimclf", out, device, epochs=epochs, **kwargs)
    registry.write_manifest(out, name="rimclf", version=version, device=device, **info)
    log(f"registered rimclf {version}: best val acc {info['best_val_acc']:.3f}")
    return version, out
```

`film/commands/train.py`:
```python
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("train", help="fine-tune a model from the labeled datasets and register a new version")
    parser.add_argument("what", choices=["rim", "ball", "rimclf"])
    parser.add_argument("--epochs", type=int, default=None, help="detectors 50, classifier 8")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.common.device import pick_device

    device = pick_device(args.device)
    if args.what == "rimclf":
        from film.train.train_yolo import train_rimclf_version

        train_rimclf_version(device, epochs=args.epochs or 8)
    else:
        from film.train.train_yolo import train_detector

        train_detector(args.what, device, epochs=args.epochs or 50)
    return 0
```

In `film/cli.py`, add `"film.commands.train",` to `COMMANDS` after `"film.commands.prelabel",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_train_yolo.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "Training commands that register detector and classifier versions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Human step (Garrett), after Task 19 gives you the labeling page:** label at least 150 rim frames (`film prelabel rim <game>` on two or three different games, then the Label tab), run `uv run film train rim` overnight, then `film prelabel ball` on a game whose events exist, label at least 300 crops, and run `uv run film train ball`. The registry picks up each new version automatically; `film doctor` lists them.

---

### Task 17: Clips and the reel

**Files:**
- Create: `film/export/__init__.py`, `film/export/clips.py`, `film/commands/clips.py`
- Modify: `film/cli.py` (add `"film.commands.clips"`)
- Test: `tests/test_clips.py`

**Interfaces:**
- Consumes: `ffmpeg_exe` (Task 1), `probe` (Task 3), `load_events` (Task 14).
- Produces: `cut_clip(source, start, end, out_path, width=640) -> Path` (H.264 if the ffmpeg build has it, else mpeg4; no audio); `clip_path(game_dir, event_id) -> Path` (`<game_dir>/clips/<event_id>.mp4`); `clips_for_game(game_id, force=False, log=print) -> int`; `build_reel(clips: list[tuple[Path, str]], out_path, size=(640, 360)) -> Path` (OpenCV re-encode with the label burned in at top-left).

- [ ] **Step 1: Write the failing tests**

`tests/test_clips.py`:
```python
import cv2

from film.common.paths import game_dir
from film.common.video import probe
from film.export.clips import build_reel, clip_path, clips_for_game, cut_clip
from film.ingest.run import ingest
from film.interpret.events import Event, save_events


def test_cut_clip_duration(synthetic_video, tmp_path):
    out = cut_clip(synthetic_video, 0.5, 1.5, tmp_path / "c" / "clip.mp4")
    assert out.exists()
    assert 0.8 <= probe(out).duration_s <= 1.3


def test_clips_for_game_and_reel(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i * 0.5, rim_id=1, quarter=1, clock_s=100.0, team=None,
                    made=bool(i % 2), confidence=0.5, clip={"start": 0.5 + i * 0.5, "end": 1.5 + i * 0.5}, signals={}, flags=[])
              for i in range(2)]
    save_events(gd, events)
    assert clips_for_game(m.game_id, log=lambda *a: None) == 2
    assert clip_path(gd, "shot-000000-r1").exists()
    assert clips_for_game(m.game_id, log=lambda *a: None) == 0          # already cut
    reel = build_reel([(clip_path(gd, e.id), f"Q{e.quarter} {'MADE' if e.made else 'MISS'}") for e in events], tmp_path / "reel.mp4")
    cap = cv2.VideoCapture(str(reel))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 10
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_clips.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.export'`.

- [ ] **Step 3: Write the implementation**

`film/export/__init__.py`: empty.

`film/export/clips.py`:
```python
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
```

`film/commands/clips.py`:
```python
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("clips", help="cut a short clip for every event, for the review page")
    parser.add_argument("game_id")
    parser.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.export.clips import clips_for_game

    clips_for_game(args.game_id, force=args.force)
    return 0
```

In `film/cli.py`, add `"film.commands.clips",` to `COMMANDS` after `"film.commands.train",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_clips.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "Clips per event and a labelled reel

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 18: Review decisions and the review/label API

**Files:**
- Create: `film/review/__init__.py`, `film/review/decisions.py`, `film/review/app.py`, `film/review/static/index.html` (placeholder; Task 19 replaces it)
- Test: `tests/test_decisions.py`, `tests/test_review_app.py`

**Interfaces:**
- Consumes: `load_events` (Task 14), `clip_path` (Task 17), datasets pending queue and accept functions (Task 15), `Manifest`, `games_dir`, `game_dir` (Task 2).
- Produces (decisions): `STATUSES = ("confirmed", "rejected", "unsure")`; `load_decisions(game_dir) -> dict`; `save_decision(game_dir, event_id, decision: dict) -> dict` (stores `status`, optional `made` and `team`, stamps `decided_at`; raises `ValueError` on a bad status); `apply_decisions(events: list[dict], decisions) -> list[dict]` (adds `decision` and overrides `made`/`team` when the coach set them).
- Produces (app): `create_app() -> FastAPI` with `GET /api/games`, `GET /api/games/{game_id}/events`, `POST /api/games/{game_id}/decisions/{event_id}` (body `{status, made?, team?}`), `GET /clips/{game_id}/{event_id}.mp4`, `GET /video/{game_id}` (byte-range streaming of the source file), `GET /api/label/{dataset}/next`, `GET /labels/{dataset}/{image_id}.jpg`, `POST /api/label/{dataset}/{image_id}/boxes` (body `{boxes: [{cls, x1, y1, x2, y2}]}`), `POST /api/label/{dataset}/{image_id}/class` (body `{cls}`), `POST /api/label/{dataset}/{image_id}/skip`, and the static page at `/`.

- [ ] **Step 1: Write the failing tests**

`tests/test_decisions.py`:
```python
import pytest

from film.review.decisions import apply_decisions, load_decisions, save_decision


def test_save_and_apply(tmp_path):
    assert load_decisions(tmp_path) == {}
    entry = save_decision(tmp_path, "shot-1", {"status": "confirmed", "made": False})
    assert entry["status"] == "confirmed" and entry["made"] is False and "decided_at" in entry
    save_decision(tmp_path, "shot-2", {"status": "rejected"})
    decisions = load_decisions(tmp_path)
    assert set(decisions) == {"shot-1", "shot-2"}
    events = [{"id": "shot-1", "made": True, "team": "home"}, {"id": "shot-2", "made": None, "team": None}, {"id": "shot-3", "made": True, "team": None}]
    merged = apply_decisions(events, decisions)
    assert merged[0]["made"] is False and merged[0]["team"] == "home" and merged[0]["decision"]["status"] == "confirmed"
    assert merged[1]["decision"]["status"] == "rejected"
    assert merged[2]["decision"] is None and merged[2]["made"] is True


def test_bad_status_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_decision(tmp_path, "shot-1", {"status": "maybe"})
```

`tests/test_review_app.py`:
```python
import numpy as np
from fastapi.testclient import TestClient

from film.common.paths import game_dir
from film.export.clips import clip_path, cut_clip
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.perceive.detectors import Box
from film.review.app import create_app
from film.train import datasets as ds


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("FILM_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FILM_LABELS_DIR", str(tmp_path / "labels"))


def _game(synthetic_video):
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i * 0.5, rim_id=1, quarter=1, clock_s=100.0, team=None,
                    made=bool(i), confidence=0.4 + 0.5 * i, clip={"start": 0.5 + i * 0.5, "end": 1.5 + i * 0.5}, signals={}, flags=[])
              for i in range(2)]
    save_events(gd, events)
    cut_clip(synthetic_video, 0.5, 1.5, clip_path(gd, events[0].id))
    return m.game_id, gd


def test_games_events_and_decisions(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    client = TestClient(create_app())
    games = client.get("/api/games").json()
    assert games[0]["game_id"] == game_id and games[0]["events"] == 2 and games[0]["decided"] == 0
    events = client.get(f"/api/games/{game_id}/events").json()
    assert [e["id"] for e in events] == ["shot-000000-r1", "shot-000001-r1"] and events[0]["decision"] is None
    r = client.post(f"/api/games/{game_id}/decisions/shot-000001-r1", json={"status": "confirmed", "made": False})
    assert r.status_code == 200 and r.json()["status"] == "confirmed"
    events = client.get(f"/api/games/{game_id}/events").json()
    assert events[1]["made"] is False and events[1]["decision"]["status"] == "confirmed"
    assert client.get("/api/games").json()[0]["decided"] == 1
    assert client.post(f"/api/games/{game_id}/decisions/nope", json={"status": "confirmed"}).status_code == 404
    assert client.post(f"/api/games/{game_id}/decisions/shot-000000-r1", json={"status": "maybe"}).status_code == 400
    assert client.get("/api/games/missing/events").status_code == 404


def test_clip_and_ranged_video(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    game_id, gd = _game(synthetic_video)
    client = TestClient(create_app())
    assert client.get(f"/clips/{game_id}/shot-000000-r1.mp4").status_code == 200
    assert client.get(f"/clips/{game_id}/shot-000001-r1.mp4").status_code == 404
    whole = client.get(f"/video/{game_id}")
    assert whole.status_code == 200 and whole.headers["accept-ranges"] == "bytes"
    part = client.get(f"/video/{game_id}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    assert part.headers["content-range"].startswith("bytes 0-99/")


def test_label_flow(synthetic_video, tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    assert client.get("/api/label/rim/next").json() == {"done": True, "remaining": 0}
    img = np.zeros((100, 200, 3), np.uint8)
    ds.write_pending("rim", "g_1", img, [Box(10, 20, 30, 40, 0.5, "rim")])
    ds.write_pending("rim", "g_2", img, [])
    nxt = client.get("/api/label/rim/next").json()
    assert nxt["image_id"] == "g_1" and nxt["remaining"] == 2 and nxt["boxes"][0]["cls"] == "rim"
    assert client.get(nxt["image_url"]).status_code == 200
    r = client.post("/api/label/rim/g_1/boxes", json={"boxes": [{"cls": "rim", "x1": 10, "y1": 20, "x2": 30, "y2": 40}]})
    assert r.status_code == 200 and r.json()["split"] == "val"
    assert client.post("/api/label/rim/g_2/skip").status_code == 200
    assert client.get("/api/label/rim/next").json()["done"] is True
    ds.write_pending("rimclf", "m_1", np.zeros((288, 288, 3), np.uint8), [], extra={"suggested": "missed"})
    assert client.get("/api/label/rimclf/next").json()["suggested"] == "missed"
    assert client.post("/api/label/rimclf/m_1/class", json={"cls": "hit"}).status_code == 400
    assert client.post("/api/label/rimclf/m_1/class", json={"cls": "missed"}).json()["split"] == "val"
    assert client.get("/").status_code == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_decisions.py tests/test_review_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.review'`.

- [ ] **Step 3: Write the implementation**

`film/review/__init__.py`: empty.

`film/review/static/index.html` (placeholder, replaced in Task 19):
```html
<!doctype html><meta charset="utf-8"><title>Film review</title><p>The review page arrives in Task 19.</p>
```

`film/review/decisions.py`:
```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("confirmed", "rejected", "unsure")


def decisions_path(game_dir: Path) -> Path:
    return Path(game_dir) / "review.json"


def load_decisions(game_dir: Path) -> dict:
    path = decisions_path(game_dir)
    return json.loads(path.read_text()) if path.exists() else {}


def save_decision(game_dir: Path, event_id: str, decision: dict) -> dict:
    status = decision.get("status")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    entry = {"status": status, "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key in ("made", "team"):
        if key in decision:
            entry[key] = decision[key]
    everything = load_decisions(game_dir)
    everything[event_id] = entry
    Path(game_dir).mkdir(parents=True, exist_ok=True)
    decisions_path(game_dir).write_text(json.dumps(everything, indent=1))
    return entry


def apply_decisions(events: list[dict], decisions: dict) -> list[dict]:
    """Events with the coach's decision attached and their made/team overridden where the coach set them."""
    out = []
    for event in events:
        merged = dict(event)
        decision = decisions.get(event["id"])
        merged["decision"] = decision
        if decision:
            for key in ("made", "team"):
                if key in decision:
                    merged[key] = decision[key]
        out.append(merged)
    return out
```

`film/review/app.py`:
```python
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from film.common.manifest import Manifest
from film.common.paths import game_dir, games_dir
from film.export.clips import clip_path
from film.interpret.events import load_events
from film.perceive.detectors import Box
from film.review.decisions import apply_decisions, load_decisions, save_decision
from film.train import datasets as ds
from film.train.rimclf import CLASSES

STATIC = Path(__file__).parent / "static"


class Decision(BaseModel):
    status: str
    made: bool | None = None
    team: str | None = None


class BoxIn(BaseModel):
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float


class BoxLabel(BaseModel):
    boxes: list[BoxIn]


class ClassLabel(BaseModel):
    cls: str


def _gd(game_id: str) -> Path:
    gd = game_dir(game_id)
    if not (gd / "manifest.json").exists():
        raise HTTPException(404, f"no game {game_id}")
    return gd


def _summary(gd: Path) -> dict:
    m = Manifest.load(gd)
    events = load_events(gd)
    decisions = load_decisions(gd)
    return {"game_id": m.game_id, "profile": m.profile, "date": m.date, "events": len(events),
            "decided": sum(1 for e in events if e.id in decisions),
            "stages": {name: info.get("status") for name, info in m.stages.items()}}


def _range_response(path: Path, range_header: str | None):
    size = path.stat().st_size
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    match = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if not match:
        raise HTTPException(416, "bad range")
    start = int(match.group(1) or 0)
    end = min(int(match.group(2) or size - 1), size - 1)
    if start > end:
        raise HTTPException(416, "bad range")

    def body():
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)}
    return StreamingResponse(body(), status_code=206, media_type="video/mp4", headers=headers)


def create_app() -> FastAPI:
    app = FastAPI(title="film review")

    @app.get("/api/games")
    def games():
        root = games_dir()
        if not root.exists():
            return []
        return [_summary(p) for p in sorted(root.iterdir()) if (p / "manifest.json").exists()]

    @app.get("/api/games/{game_id}/events")
    def events(game_id: str):
        gd = _gd(game_id)
        return apply_decisions([e.to_json() for e in load_events(gd)], load_decisions(gd))

    @app.post("/api/games/{game_id}/decisions/{event_id}")
    def decide(game_id: str, event_id: str, decision: Decision):
        gd = _gd(game_id)
        if event_id not in {e.id for e in load_events(gd)}:
            raise HTTPException(404, "no such event")
        try:
            return save_decision(gd, event_id, decision.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/clips/{game_id}/{event_id}.mp4")
    def clip(game_id: str, event_id: str):
        path = clip_path(_gd(game_id), event_id)
        if not path.exists():
            raise HTTPException(404, "clip not cut yet; run film clips")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/video/{game_id}")
    def video(game_id: str, request: Request):
        m = Manifest.load(_gd(game_id))
        return _range_response(Path(m.source), request.headers.get("range"))

    @app.get("/api/label/{dataset}/next")
    def label_next(dataset: str):
        ids = ds.list_pending(dataset)
        if not ids:
            return {"done": True, "remaining": 0}
        data = ds.read_pending(dataset, ids[0])
        data["remaining"] = len(ids)
        data["image_url"] = f"/labels/{dataset}/{ids[0]}.jpg"
        return data

    @app.get("/labels/{dataset}/{image_id}.jpg")
    def label_image(dataset: str, image_id: str):
        path = ds.pending_images_dir(dataset) / f"{image_id}.jpg"
        if not path.exists():
            raise HTTPException(404, "no such pending image")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/label/{dataset}/{image_id}/boxes")
    def label_boxes(dataset: str, image_id: str, body: BoxLabel):
        if dataset not in ds.CLASS_NAMES:
            raise HTTPException(400, f"{dataset} is not a box dataset")
        boxes = [Box(b.x1, b.y1, b.x2, b.y2, 1.0, b.cls) for b in body.boxes]
        return {"split": ds.accept_boxes(dataset, image_id, boxes)}

    @app.post("/api/label/{dataset}/{image_id}/class")
    def label_class(dataset: str, image_id: str, body: ClassLabel):
        if body.cls not in CLASSES:
            raise HTTPException(400, f"cls must be one of {CLASSES}")
        return {"split": ds.accept_class(dataset, image_id, body.cls)}

    @app.post("/api/label/{dataset}/{image_id}/skip")
    def label_skip(dataset: str, image_id: str):
        ds.remove_pending(dataset, image_id)
        return {"skipped": image_id}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_decisions.py tests/test_review_app.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "Review decisions and the review/label API

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 19: The review and labeling page, and `film review`

**Files:**
- Create: `film/commands/review.py`
- Modify: `film/review/static/index.html` (replace the placeholder), `film/cli.py` (add `"film.commands.review"`)
- Test: `tests/test_review_command.py`, plus a manual check in the browser

**Interfaces:**
- Consumes: every endpoint from Task 18.
- Produces: `film review [--port 8765] [--no-open]` serving the page on `127.0.0.1`. The page has a game list, a Shots tab (the shot queue with the keys from spec section 8: `Enter` confirm, `M` toggle made, `T` cycle team, `X` reject, `U` unsure, `L` open the film at the shot, `J`/`K` previous/next, and a "Confirm all above 0.9" button), and three Label tabs (rim boxes, ball boxes, clip classes).

- [ ] **Step 1: Write the failing test**

`tests/test_review_command.py`:
```python
from pathlib import Path

from film.cli import build_parser


def test_review_command_registered_and_page_has_the_tabs():
    args = build_parser().parse_args(["review", "--port", "9999", "--no-open"])
    assert args.port == 9999 and args.no_open is True and args.func.__module__ == "film.commands.review"
    html = (Path("film") / "review" / "static" / "index.html").read_text()
    for needle in ('data-tab="shots"', 'data-tab="rim"', 'data-tab="ball"', 'data-tab="rimclf"', "/api/games", "Confirm all above 0.9"):
        assert needle in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_review_command.py -v`
Expected: FAIL (`review` is not a valid choice).

- [ ] **Step 3: Write the command and the page**

`film/commands/review.py`:
```python
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("review", help="serve the review and labeling page in your browser")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    parser.set_defaults(func=run)


def run(args) -> int:
    import threading
    import webbrowser

    import uvicorn

    from film.review.app import create_app

    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"review page at {url}  (Ctrl+C to stop)")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0
```

In `film/cli.py`, add `"film.commands.review",` to `COMMANDS` after `"film.commands.clips",`.

`film/review/static/index.html` (replace the whole file):
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Film review</title>
<style>
  :root { --bg:#0f1218; --panel:#171c26; --line:#2a3242; --text:#e6e9ef; --muted:#8b95a7; --ok:#3ddc84; --bad:#ff5c5c; --accent:#4c8dff; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--text); display:grid; grid-template-columns:260px 1fr; height:100vh; }
  aside { border-right:1px solid var(--line); padding:12px; overflow:auto; }
  main { padding:16px; overflow:auto; }
  h1 { font-size:16px; margin:0 0 12px; }
  .game { padding:8px; border:1px solid var(--line); border-radius:6px; margin-bottom:8px; cursor:pointer; }
  .game.sel { border-color:var(--accent); }
  .game small { color:var(--muted); display:block; }
  nav button { background:var(--panel); color:var(--text); border:1px solid var(--line); padding:6px 12px; border-radius:6px; cursor:pointer; margin-right:6px; }
  nav button.sel { border-color:var(--accent); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; margin-top:12px; max-width:960px; }
  video { width:100%; max-width:640px; background:#000; border-radius:6px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-weight:600; margin-right:6px; }
  .made { background:var(--ok); color:#062; } .miss { background:var(--bad); color:#400; } .none { background:var(--line); }
  .muted { color:var(--muted); }
  .keys { color:var(--muted); font-size:12px; margin-top:8px; }
  kbd { background:var(--line); padding:1px 5px; border-radius:4px; font-family:inherit; }
  canvas { max-width:100%; border:1px solid var(--line); border-radius:6px; cursor:crosshair; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .btn { background:var(--accent); color:#fff; border:0; padding:6px 12px; border-radius:6px; cursor:pointer; }
  .list { margin-top:12px; max-height:240px; overflow:auto; font-size:12px; max-width:960px; }
  .list div { padding:3px 6px; border-bottom:1px solid var(--line); cursor:pointer; }
  .list div.sel { background:#22304a; }
</style>
</head>
<body>
<aside>
  <h1>Games</h1>
  <div id="games"></div>
</aside>
<main>
  <nav>
    <button data-tab="shots" class="sel">Shots</button>
    <button data-tab="rim">Label rim</button>
    <button data-tab="ball">Label ball</button>
    <button data-tab="rimclf">Label clips</button>
  </nav>
  <section id="shots"></section>
  <section id="label" hidden></section>
</main>
<script>
const state = { game: null, tab: "shots", events: [], index: 0, label: null, boxes: [], selected: -1, drawing: null, img: null };
const CLASSES = { rim: ["rim", "backboard"], ball: ["ball"] };
const $ = (sel) => document.querySelector(sel);
const api = async (url, opts) => { const r = await fetch(url, opts); if (!r.ok) throw new Error(await r.text()); return r.json(); };
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
const clock = (s) => s == null ? "--" : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function loadGames() {
  const games = await api("/api/games");
  $("#games").innerHTML = games.map(g => `<div class="game ${g.game_id === state.game ? "sel" : ""}" data-id="${esc(g.game_id)}"><b>${esc(g.game_id)}</b><small>${esc(g.profile)} · ${g.decided}/${g.events} decided</small></div>`).join("")
    || "<div class='muted'>No games yet. Run: film run &lt;file&gt;</div>";
  document.querySelectorAll(".game").forEach(el => el.onclick = () => selectGame(el.dataset.id));
  if (!state.game && games.length) selectGame(games[0].game_id);
}

async function selectGame(id) {
  state.game = id;
  state.events = await api(`/api/games/${id}/events`);
  state.events.sort((a, b) => (!!a.decision - !!b.decision) || (a.confidence - b.confidence));
  state.index = 0;
  await loadGames();
  render();
}

function render() {
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("sel", b.dataset.tab === state.tab));
  $("#shots").hidden = state.tab !== "shots";
  $("#label").hidden = state.tab === "shots";
  if (state.tab === "shots") renderShots(); else loadLabel();
}

function renderShots() {
  const el = $("#shots");
  if (!state.game) { el.innerHTML = "<p class='muted'>Pick a game.</p>"; return; }
  const decided = state.events.filter(e => e.decision).length;
  const e = state.events[state.index];
  if (!e) { el.innerHTML = `<p>No events for this game yet.</p>`; return; }
  const madeBadge = e.made === true ? "<span class='badge made'>MADE</span>" : e.made === false ? "<span class='badge miss'>MISSED</span>" : "<span class='badge none'>UNKNOWN</span>";
  const dec = e.decision ? `<span class='badge none'>${esc(e.decision.status)}</span>` : "";
  el.innerHTML = `
    <div class="row"><b>${decided}/${state.events.length} decided</b><span class="muted">${state.index + 1} of ${state.events.length}</span>
      <button class="btn" id="confirm-high">Confirm all above 0.9</button></div>
    <div class="card">
      <video src="/clips/${esc(state.game)}/${esc(e.id)}.mp4" autoplay loop muted playsinline></video>
      <div style="margin-top:8px">${madeBadge}<span class="badge none">${esc(e.type)}</span>${dec}
        <b>Q${e.quarter ?? "?"} ${clock(e.clock_s)}</b> <span class="muted">t=${e.t.toFixed(1)}s · team ${esc(e.team ?? "?")} · conf ${e.confidence} · rim ${e.rim_id}</span></div>
      <div class="muted" style="margin-top:4px">${esc((e.flags || []).join(" ") || "no flags")} · ${esc(JSON.stringify(e.signals))}</div>
      <div class="keys"><kbd>Enter</kbd> confirm <kbd>M</kbd> toggle made <kbd>T</kbd> team <kbd>X</kbd> reject <kbd>U</kbd> unsure <kbd>L</kbd> open film <kbd>J</kbd>/<kbd>K</kbd> prev/next</div>
    </div>
    <div class="list">${state.events.map((x, i) => `<div class="${i === state.index ? "sel" : ""}" data-i="${i}">${x.decision ? "✓" : "·"} ${esc(x.id)} Q${x.quarter ?? "?"} ${clock(x.clock_s)} ${x.made === true ? "made" : x.made === false ? "missed" : "?"} ${x.confidence}</div>`).join("")}</div>`;
  $("#confirm-high").onclick = confirmHigh;
  document.querySelectorAll(".list div").forEach(d => d.onclick = () => { state.index = +d.dataset.i; renderShots(); });
}

async function decide(status) {
  const e = state.events[state.index];
  if (!e) return;
  e.decision = await post(`/api/games/${state.game}/decisions/${e.id}`, { status, made: e.made, team: e.team });
  state.index = Math.min(state.index + 1, state.events.length - 1);
  renderShots();
  loadGames();
}

async function confirmHigh() {
  for (const e of state.events) {
    if (!e.decision && e.type === "shot" && e.confidence >= 0.9) {
      e.decision = await post(`/api/games/${state.game}/decisions/${e.id}`, { status: "confirmed", made: e.made, team: e.team });
    }
  }
  renderShots();
  loadGames();
}

function shotKeys(ev) {
  const e = state.events[state.index];
  if (!e) return;
  const k = ev.key.toLowerCase();
  if (ev.key === "Enter") decide("confirmed");
  else if (k === "m") { e.made = e.made !== true; renderShots(); }
  else if (k === "t") { e.team = e.team === "home" ? "away" : e.team === "away" ? null : "home"; renderShots(); }
  else if (k === "x") decide("rejected");
  else if (k === "u") decide("unsure");
  else if (k === "l") window.open(`/video/${state.game}#t=${Math.max(0, e.t - 5).toFixed(1)}`, "_blank");
  else if (k === "j") { state.index = Math.max(0, state.index - 1); renderShots(); }
  else if (k === "k") { state.index = Math.min(state.events.length - 1, state.index + 1); renderShots(); }
  else return;
  ev.preventDefault();
}

async function loadLabel() {
  const dataset = state.tab;
  state.label = await api(`/api/label/${dataset}/next`);
  if (state.label.done) {
    $("#label").innerHTML = `<p>Nothing pending for <b>${esc(dataset)}</b>. Queue more with: film prelabel ${esc(dataset)} &lt;game_id&gt;</p>`;
    return;
  }
  if (dataset === "rimclf") renderClipLabel(); else renderBoxLabel();
}

function renderClipLabel() {
  const l = state.label;
  $("#label").innerHTML = `<div class="row"><b>${l.remaining} pending</b><span class="muted">${esc(l.image_id)} · suggested: ${esc(l.suggested || "?")}</span></div>
    <div class="card"><img src="${l.image_url}" style="width:576px;max-width:100%">
    <div class="keys">Nine frames, left to right then top to bottom, 0.2 s apart, starting 0.4 s before the ball reaches the rim.
    <kbd>1</kbd> made <kbd>2</kbd> missed <kbd>3</kbd> not a shot <kbd>K</kbd> skip</div></div>`;
}

async function clipKeys(ev) {
  const l = state.label;
  if (!l || l.done) return;
  const map = { "1": "made", "2": "missed", "3": "none" };
  if (map[ev.key]) await post(`/api/label/rimclf/${l.image_id}/class`, { cls: map[ev.key] });
  else if (ev.key.toLowerCase() === "k") await post(`/api/label/rimclf/${l.image_id}/skip`);
  else return;
  ev.preventDefault();
  loadLabel();
}

function renderBoxLabel() {
  const l = state.label;
  const names = CLASSES[state.tab];
  state.boxes = l.boxes.map(b => ({ cls: b.cls, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 }));
  state.selected = -1;
  state.img = null;
  $("#label").innerHTML = `<div class="row"><b>${l.remaining} pending</b><span class="muted">${esc(l.image_id)}</span></div>
    <div class="card"><canvas id="cv" width="${l.width}" height="${l.height}"></canvas>
    <div class="keys">Drag to draw a box. Click a box to select it. <kbd>Delete</kbd> remove
    ${names.map((n, i) => `<kbd>${i + 1}</kbd> ${esc(n)}`).join(" ")} <kbd>S</kbd> save <kbd>K</kbd> skip</div></div>`;
  const img = new Image();
  img.onload = () => { state.img = img; drawBoxes(); };
  img.src = l.image_url;
  const cv = $("#cv");
  const pos = (ev) => { const r = cv.getBoundingClientRect(); return { x: (ev.clientX - r.left) * cv.width / r.width, y: (ev.clientY - r.top) * cv.height / r.height }; };
  cv.onmousedown = (ev) => {
    const p = pos(ev);
    const hit = state.boxes.findIndex(b => p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2);
    if (hit >= 0) { state.selected = hit; drawBoxes(); return; }
    state.drawing = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
  };
  cv.onmousemove = (ev) => { if (!state.drawing) return; const p = pos(ev); state.drawing.x2 = p.x; state.drawing.y2 = p.y; drawBoxes(); };
  cv.onmouseup = () => {
    const d = state.drawing;
    state.drawing = null;
    if (!d) return;
    const b = { cls: names[0], x1: Math.min(d.x1, d.x2), y1: Math.min(d.y1, d.y2), x2: Math.max(d.x1, d.x2), y2: Math.max(d.y1, d.y2) };
    if (b.x2 - b.x1 > 3 && b.y2 - b.y1 > 3) { state.boxes.push(b); state.selected = state.boxes.length - 1; }
    drawBoxes();
  };
}

function drawBoxes() {
  const cv = $("#cv");
  if (!cv || !state.img) return;
  const ctx = cv.getContext("2d");
  ctx.drawImage(state.img, 0, 0);
  const colours = { rim: "#3ddc84", backboard: "#4c8dff", ball: "#ffc857" };
  ctx.font = "14px sans-serif";
  state.boxes.forEach((b, i) => {
    ctx.lineWidth = i === state.selected ? 4 : 2;
    ctx.strokeStyle = colours[b.cls] || "#fff";
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(b.cls, b.x1 + 2, Math.max(12, b.y1 - 4));
  });
  if (state.drawing) {
    const d = state.drawing;
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#fff";
    ctx.strokeRect(Math.min(d.x1, d.x2), Math.min(d.y1, d.y2), Math.abs(d.x2 - d.x1), Math.abs(d.y2 - d.y1));
    ctx.setLineDash([]);
  }
}

async function boxKeys(ev) {
  const l = state.label;
  if (!l || l.done) return;
  const names = CLASSES[state.tab];
  const k = ev.key.toLowerCase();
  if (ev.key === "Delete" || ev.key === "Backspace") {
    if (state.selected >= 0) { state.boxes.splice(state.selected, 1); state.selected = -1; drawBoxes(); }
  } else if (names[+ev.key - 1]) {
    if (state.selected >= 0) { state.boxes[state.selected].cls = names[+ev.key - 1]; drawBoxes(); }
  } else if (k === "s") {
    await post(`/api/label/${state.tab}/${l.image_id}/boxes`, { boxes: state.boxes });
    loadLabel();
  } else if (k === "k") {
    await post(`/api/label/${state.tab}/${l.image_id}/skip`);
    loadLabel();
  } else return;
  ev.preventDefault();
}

document.addEventListener("keydown", (ev) => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (state.tab === "shots") shotKeys(ev);
  else if (state.tab === "rimclf") clipKeys(ev);
  else boxKeys(ev);
});
document.querySelectorAll("nav button").forEach(b => b.onclick = () => { state.tab = b.dataset.tab; render(); });
loadGames();
</script>
</body>
</html>
```

- [ ] **Step 4: Run the test, then check the page by hand**

Run: `uv run pytest tests/test_review_command.py tests/test_review_app.py -v`
Expected: 4 passed.

Then, with the two-minute game from Task 14 (`film clips 2025-02-05-jv-vs-marcos-tactical` first):
```bash
uv run film review
```
Check in the browser: the game appears in the sidebar; the Shots tab shows a looping clip with quarter and clock; `M` flips the badge; `Enter` advances and the sidebar count rises; `L` opens the film in a new tab near the shot; the Label rim tab (after `uv run film prelabel rim 2025-02-05-jv-vs-marcos-tactical --n 10`) shows proposed boxes, dragging draws a new one, `S` saves and moves on; the Label clips tab (after `uv run film prelabel rimclf 2025-02-05-jv-vs-marcos-tactical`) shows a montage and `1`/`2`/`3` files it.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "The review and labeling page

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Human step (Garrett):** the labeling loop in Task 16's note can start now.

---

### Task 20: Export the shot list and reel, and evaluate

**Files:**
- Create: `film/export/shots.py`, `film/export/evaluate.py`, `film/commands/export.py`, `film/commands/eval.py`
- Modify: `film/cli.py` (add `"film.commands.export"` and `"film.commands.eval"`)
- Test: `tests/test_export_shots.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `load_events` (Task 14), decisions (Task 18), `clip_path`, `build_reel` (Task 17).
- Produces: `export_shots(game_id) -> tuple[Path, Path]` writing `export/<game_id>.shots.csv` and `.shots.json` with confirmed and unsure shots (coach overrides applied); `export_reel(game_id) -> Path | None` writing `export/<game_id>.reel.mp4` from confirmed shots in time order; `load_reference(path) -> list[float]` (one time per line: seconds, `M:SS` of video time, or `t=<seconds>`; `#` comments allowed); `match_times(machine, reference, tol_s=2.0) -> tuple[int, int, int]` = (matched, unmatched machine, unmatched reference); `evaluate(game_id, reference_times=None, tol_s=2.0) -> dict` with `events, shots, confirmed, rejected, unsure, undecided, precision, made_agreement, made_compared` and, when a reference is given, `reference, matched, recall, precision_vs_reference`.

- [ ] **Step 1: Write the failing tests**

`tests/test_export_shots.py`:
```python
import csv
import json

from film.common.paths import game_dir
from film.export.clips import clip_path, cut_clip
from film.export.shots import export_reel, export_shots
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.review.decisions import save_decision


def _game(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=1.0 + i * 0.4, rim_id=1, quarter=2, clock_s=125.0 - i, team="home",
                    made=True, confidence=0.9, clip={"start": 0.5, "end": 1.5}, signals={}, flags=["x"]) for i in range(3)]
    save_events(gd, events)
    save_decision(gd, "shot-000000-r1", {"status": "confirmed", "made": False})
    save_decision(gd, "shot-000001-r1", {"status": "rejected"})
    save_decision(gd, "shot-000002-r1", {"status": "unsure"})
    return m.game_id, gd


def test_export_shots(synthetic_video, tmp_path, monkeypatch):
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    csv_path, json_path = export_shots(game_id)
    rows = list(csv.DictReader(csv_path.open()))
    assert [r["id"] for r in rows] == ["shot-000000-r1", "shot-000002-r1"]
    assert rows[0]["made"] == "False" and rows[0]["status"] == "confirmed" and rows[1]["status"] == "unsure"
    data = json.loads(json_path.read_text())
    assert data["format"] == "film-shot-times" and len(data["shots"]) == 2 and data["shots"][0]["made"] is False


def test_export_reel_uses_confirmed_only(synthetic_video, tmp_path, monkeypatch):
    game_id, gd = _game(synthetic_video, tmp_path, monkeypatch)
    for i in range(3):
        cut_clip(synthetic_video, 0.5, 1.5, clip_path(gd, f"shot-{i:06d}-r1"))
    reel = export_reel(game_id)
    assert reel is not None and reel.exists() and reel.name.endswith(".reel.mp4")
```

`tests/test_evaluate.py`:
```python
from film.common.paths import game_dir
from film.export.evaluate import evaluate, load_reference, match_times
from film.ingest.run import ingest
from film.interpret.events import Event, save_events
from film.review.decisions import save_decision


def test_load_reference(tmp_path):
    path = tmp_path / "ref.txt"
    path.write_text("# shots\n90\n25:00\nt=1500.4\n\n")
    assert load_reference(path) == [90.0, 1500.0, 1500.4]


def test_match_times():
    assert match_times([10.0, 20.0, 30.0], [11.0, 29.0, 50.0]) == (2, 1, 1)
    assert match_times([], [1.0]) == (0, 0, 1)


def test_evaluate(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    m = ingest(synthetic_video, date="2026-01-01", profile="test")
    gd = game_dir(m.game_id)
    made = [True, True, False, None]
    events = [Event(id=f"shot-{i:06d}-r1", type="shot", t=10.0 * (i + 1), rim_id=1, quarter=1, clock_s=None, team=None,
                    made=made[i], confidence=0.5, clip={"start": 0, "end": 1}, signals={}, flags=[]) for i in range(4)]
    events.append(Event(id="unknown-1", type="unknown", t=99.0, rim_id=1, quarter=1, clock_s=None, team=None, made=None,
                        confidence=0.2, clip={"start": 0, "end": 1}, signals={}, flags=[]))
    save_events(gd, events)
    save_decision(gd, "shot-000000-r1", {"status": "confirmed"})                  # machine made=True, coach agrees
    save_decision(gd, "shot-000001-r1", {"status": "confirmed", "made": False})   # coach flipped it
    save_decision(gd, "shot-000002-r1", {"status": "rejected"})
    save_decision(gd, "shot-000003-r1", {"status": "unsure"})
    result = evaluate(m.game_id)
    assert result["shots"] == 4 and result["confirmed"] == 2 and result["rejected"] == 1 and result["unsure"] == 1 and result["undecided"] == 0
    assert result["precision"] == 2 / 3
    assert result["made_compared"] == 2 and result["made_agreement"] == 0.5
    with_ref = evaluate(m.game_id, reference_times=[10.5, 20.0, 60.0])
    assert with_ref["reference"] == 3 and with_ref["matched"] == 2
    assert with_ref["recall"] == 2 / 3 and with_ref["precision_vs_reference"] == 2 / 3      # machine = confirmed + unsure = 10, 20, 40
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_export_shots.py tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'film.export.shots'`.

- [ ] **Step 3: Write the implementation**

`film/export/shots.py`:
```python
from __future__ import annotations

import csv
import json
from pathlib import Path

from film.common.paths import game_dir
from film.export.clips import build_reel, clip_path
from film.interpret.events import load_events
from film.review.decisions import apply_decisions, load_decisions

FIELDS = ["id", "t", "quarter", "clock_s", "team", "made", "confidence", "status", "flags"]


def reviewed_shots(game_id: str, statuses=("confirmed", "unsure")) -> list[dict]:
    gd = game_dir(game_id)
    merged = apply_decisions([e.to_json() for e in load_events(gd)], load_decisions(gd))
    rows = [e for e in merged if e["type"] == "shot" and e.get("decision") and e["decision"]["status"] in statuses]
    return sorted(rows, key=lambda e: e["t"])


def export_shots(game_id: str) -> tuple[Path, Path]:
    gd = game_dir(game_id)
    rows = reviewed_shots(game_id)
    out = gd / "export"
    out.mkdir(exist_ok=True)
    csv_path, json_path = out / f"{game_id}.shots.csv", out / f"{game_id}.shots.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for e in rows:
            writer.writerow({"id": e["id"], "t": e["t"], "quarter": e["quarter"], "clock_s": e["clock_s"], "team": e["team"],
                             "made": e["made"], "confidence": e["confidence"], "status": e["decision"]["status"],
                             "flags": " ".join(e.get("flags", []))})
    json_path.write_text(json.dumps({"format": "film-shot-times", "schemaVersion": 1, "gameId": game_id, "shots": rows}, indent=1))
    return csv_path, json_path


def _label(e: dict) -> str:
    clock = "--" if e["clock_s"] is None else f"{int(e['clock_s'] // 60)}:{int(e['clock_s'] % 60):02d}"
    result = "MADE" if e["made"] else "MISS" if e["made"] is False else "?"
    return f"Q{e['quarter'] or '?'} {clock} {result}"


def export_reel(game_id: str) -> Path | None:
    gd = game_dir(game_id)
    rows = [e for e in reviewed_shots(game_id, statuses=("confirmed",)) if clip_path(gd, e["id"]).exists()]
    if not rows:
        return None
    out = gd / "export"
    out.mkdir(exist_ok=True)
    return build_reel([(clip_path(gd, e["id"]), _label(e)) for e in rows], out / f"{game_id}.reel.mp4")
```

`film/export/evaluate.py`:
```python
from __future__ import annotations

from pathlib import Path

from film.common.paths import game_dir
from film.interpret.events import load_events
from film.review.decisions import load_decisions


def load_reference(path: Path) -> list[float]:
    """One shot time per line: seconds, M:SS of video time, or t=<seconds>. Blank lines and # comments ignored."""
    times = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("t="):
            line = line[2:]
        if ":" in line:
            minutes, seconds = line.split(":", 1)
            times.append(float(minutes) * 60 + float(seconds))
        else:
            times.append(float(line))
    return times


def match_times(machine: list[float], reference: list[float], tol_s: float = 2.0) -> tuple[int, int, int]:
    """Greedy one-to-one matching in time order. Returns (matched, unmatched machine, unmatched reference)."""
    machine, reference = sorted(machine), sorted(reference)
    matched, i, j = 0, 0, 0
    while i < len(machine) and j < len(reference):
        gap = machine[i] - reference[j]
        if abs(gap) <= tol_s:
            matched += 1
            i += 1
            j += 1
        elif gap < 0:
            i += 1
        else:
            j += 1
    return matched, len(machine) - matched, len(reference) - matched


def evaluate(game_id: str, reference_times: list[float] | None = None, tol_s: float = 2.0) -> dict:
    gd = game_dir(game_id)
    events = load_events(gd)
    decisions = load_decisions(gd)
    shots = [e for e in events if e.type == "shot"]
    by_status = {s: [e for e in shots if decisions.get(e.id, {}).get("status") == s] for s in ("confirmed", "rejected", "unsure")}
    confirmed, rejected, unsure = (len(by_status[s]) for s in ("confirmed", "rejected", "unsure"))
    compared = [e for e in by_status["confirmed"] if e.made is not None]
    agree = sum(1 for e in compared if decisions[e.id].get("made", e.made) == e.made)
    result = {
        "events": len(events), "shots": len(shots), "confirmed": confirmed, "rejected": rejected, "unsure": unsure,
        "undecided": len(shots) - confirmed - rejected - unsure,
        "precision": confirmed / (confirmed + rejected) if confirmed + rejected else None,
        "made_compared": len(compared), "made_agreement": agree / len(compared) if compared else None,
    }
    if reference_times is not None:
        machine = [e.t for e in by_status["confirmed"] + by_status["unsure"]]
        matched, _, _ = match_times(machine, reference_times, tol_s)
        result.update({"reference": len(reference_times), "matched": matched,
                       "recall": matched / len(reference_times) if reference_times else None,
                       "precision_vs_reference": matched / len(machine) if machine else None})
    return result
```

`film/commands/export.py`:
```python
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("export", help="write the reviewed shot list (CSV and JSON) and, optionally, the reel")
    parser.add_argument("game_id")
    parser.add_argument("--reel", action="store_true", help="also build export/<game_id>.reel.mp4 from confirmed shots")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.export.shots import export_reel, export_shots

    csv_path, json_path = export_shots(args.game_id)
    print(f"wrote {csv_path}\nwrote {json_path}")
    if args.reel:
        reel = export_reel(args.game_id)
        print(f"wrote {reel}" if reel else "no confirmed shots with clips; reel not built")
    return 0
```

`film/commands/eval.py`:
```python
from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("eval", help="score the machine against the coach's decisions and an optional list of true shot times")
    parser.add_argument("game_id")
    parser.add_argument("--reference", type=Path, default=None, help="text file, one shot time per line (seconds or M:SS of video time)")
    parser.add_argument("--tol", type=float, default=2.0, help="seconds of slack when matching times")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.export.evaluate import evaluate, load_reference

    reference = load_reference(args.reference) if args.reference else None
    result = evaluate(args.game_id, reference_times=reference, tol_s=args.tol)
    width = max(len(k) for k in result)
    for key, value in result.items():
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"{key:{width}s}  {shown}")
    return 0
```

In `film/cli.py`, add `"film.commands.export",` and `"film.commands.eval",` to `COMMANDS` after `"film.commands.review",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_export_shots.py tests/test_evaluate.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add film tests
git commit -m "Export the reviewed shot list and reel, and score the machine against the coach

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 21: `film run`, the README, and the first full game

**Files:**
- Create: `film/commands/run.py`, `README.md`
- Modify: `film/cli.py` (add `"film.commands.run"`)
- Test: `tests/test_run_command.py`

**Interfaces:**
- Produces: `film run <source> [--date] [--profile] [--fps] [--device] [--limit-s]` chaining ingest, perceive, interpret and clips, then printing the review command.

- [ ] **Step 1: Write the failing test**

`tests/test_run_command.py`:
```python
import types

from film.cli import main


def test_run_chains_the_stages(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_GAMES_DIR", str(tmp_path / "games"))
    calls = []
    import film.export.clips as clips
    import film.interpret.run as interpret_mod
    import film.perceive.run as perceive_mod

    monkeypatch.setattr(perceive_mod, "perceive", lambda game_id, **kw: calls.append(("perceive", game_id, kw.get("limit_s"))))
    monkeypatch.setattr(interpret_mod, "interpret", lambda game_id, **kw: calls.append(("interpret", game_id)) or [])
    monkeypatch.setattr(clips, "clips_for_game", lambda game_id, **kw: calls.append(("clips", game_id)) or 0)
    assert main(["run", str(synthetic_video), "--date", "2026-01-01", "--profile", "test", "--limit-s", "2"]) == 0
    assert calls == [("perceive", "2026-01-01-synthetic", 2.0), ("interpret", "2026-01-01-synthetic"), ("clips", "2026-01-01-synthetic")]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_run_command.py -v`
Expected: FAIL (`run` is not a valid choice).

- [ ] **Step 3: Write the command and the README**

`film/commands/run.py`:
```python
from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("run", help="ingest, perceive, interpret and cut clips for one game file")
    parser.add_argument("source", type=Path)
    parser.add_argument("--date", help="game date YYYY-MM-DD (default: file modification date)")
    parser.add_argument("--profile", help="camera profile override")
    parser.add_argument("--fps", type=float, default=10.0, help="working frame rate (5 on the laptop)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-s", type=float, default=None, help="only process this many seconds (for a quick check)")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.export import clips
    from film.ingest.run import ingest
    from film.interpret import run as interpret_mod
    from film.perceive import run as perceive_mod

    m = ingest(args.source, date=args.date, profile=args.profile, fps=args.fps)
    print(f"ingest   {m.game_id}  profile={m.profile}  {m.duration_s:.0f}s")
    if m.profile == "unknown":
        print("warning: camera profile unknown, so no scoreboard will be read; pass --profile to set it")
    perceive_mod.perceive(m.game_id, device=args.device, limit_s=args.limit_s)
    interpret_mod.interpret(m.game_id, device=args.device)
    clips.clips_for_game(m.game_id)
    print(f"\nnext:  uv run film review        (then pick {m.game_id})")
    return 0
```

In `film/cli.py`, add `"film.commands.run",` to `COMMANDS` after `"film.commands.eval",`. The final list, in order:
```python
COMMANDS = [
    "film.commands.doctor",
    "film.commands.ingest",
    "film.commands.calibrate",
    "film.commands.scoreboard",
    "film.commands.perceive",
    "film.commands.interpret",
    "film.commands.prelabel",
    "film.commands.train",
    "film.commands.clips",
    "film.commands.review",
    "film.commands.export",
    "film.commands.eval",
    "film.commands.run",
]
```

`README.md`:
```markdown
# Basketball film pipeline

Turns a downloaded game film into shot data for the shot tracker. Rung 1 (this repo today): every shot's
time, made or missed, quarter and clock, reviewed on a local page, exported as a shot list and a clip reel.
The design is in `docs/superpowers/specs/2026-09-30-film-pipeline-design.md`.

## Setup (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # no Homebrew needed
uv sync --extra dev
uv run film doctor                                   # expect torch ... mps=True on a Mac
```

## A game, end to end

```bash
uv run film run ~/Downloads/"JV vs Marcos - Tactical.mp4" --date 2025-02-05
uv run film review                                   # opens http://127.0.0.1:8765
uv run film export 2025-02-05-jv-vs-marcos-tactical --reel
uv run film eval 2025-02-05-jv-vs-marcos-tactical
```

`film run` is `ingest` + `perceive` + `interpret` + `clips`. Each can be re-run alone. Add `--limit-s 120`
for a two-minute smoke test, `--fps 5` on the 8 GB laptop.

## Making it better: label, then train

Until you train, the pipeline runs on open-vocabulary fallbacks (YOLO-World) and the scoreboard alone.
Each of these makes it faster and more accurate:

1. `uv run film prelabel rim <game_id>` then the **Label rim** tab, on two or three games, at least 150 frames. Then `uv run film train rim` (overnight).
2. `uv run film prelabel ball <game_id>` on a game with events, then **Label ball**, at least 300 crops. Then `uv run film train ball`.
3. `uv run film prelabel rimclf <game_id>` then **Label clips** (made / missed / not a shot). Then `uv run film train rimclf`.

New versions land in `models/<name>/vN/` and are picked up automatically. Re-run `film perceive` and
`film interpret` on a game to use them.

## Where things live

| Path | What |
|---|---|
| `games/<game_id>/` | manifest, tracks, motion, scoreboard, events, clips, review decisions, exports |
| `labels/<dataset>/pending/` | what the Label tabs show next |
| `datasets/<dataset>/` | accepted labels in YOLO or ImageFolder layout |
| `models/<name>/vN/` | trained weights and their manifest |
| `fixtures/frames/` | real frames used by the tests |

Set `FILM_GAMES_DIR`, `FILM_MODELS_DIR`, `FILM_DATASETS_DIR`, `FILM_LABELS_DIR` to move any of these,
for example onto an external drive.

## Tests

```bash
uv run pytest
```
No test needs weights, the internet, or a GPU.
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: everything passes.

- [ ] **Step 5: Run the first full game with the fallback models**

```bash
uv run film run ~/Downloads/"JV vs Marcos - Tactical.mp4" --date 2025-02-05
```
On the laptop this takes a while (YOLO-World is slow; expect an hour or more). Then `uv run film review`, work through the shot queue for the first quarter, and run `uv run film eval 2025-02-05-jv-vs-marcos-tactical`. Record the precision and made-agreement numbers in the commit message of Step 6; they are the baseline the trained models have to beat.

- [ ] **Step 6: Commit**

```bash
git add film tests README.md
git commit -m "film run, the README, and the first full game on fallback models

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Human step (Garrett), the rung-1 finish line:** label and train per the README, re-run `film perceive` and `film interpret` on the same game, review again, and compare `film eval` to the baseline. Shots-only is "ready" (spec section 10) when the machine rarely invents a shot, rarely misses one you find in the film, and agrees with you on made-or-missed nearly every time.

---

## Notes for the executor

- The spec's section 4 file table listed a `frames/` directory. This plan does not write frames; every stage decodes on the fly with PyAV (Task 3). The spec has been updated to match.
- The spec placed the model registry under `film/train/`; it lives in `film/common/registry.py` so `perceive` does not import from `train`.
- No Homebrew: PyAV decodes and probes, and the ffmpeg binary bundled with `imageio-ffmpeg` cuts clips. `film doctor` shows which ffmpeg is in use.
- YOLO-World's first `set_classes` call installs a CLIP package from GitHub and downloads weights; that is the only step needing the internet, and it happens once.
- Review setup from spec section 8 (confirming the profile and the home colour, marking quarters by hand) is not in this plan: rung 1 is Hudl-only and reads quarters from the scoreboard. It arrives with the other camera profiles.
- Three human steps gate model quality but never block the code: labeling rims, labeling balls, and labeling clips. Everything runs before any of them on the fallback models.
