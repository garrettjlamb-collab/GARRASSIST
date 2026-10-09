from __future__ import annotations

import platform
import sys


def register(sub) -> None:
    parser = sub.add_parser("doctor", help="check the environment")
    parser.set_defaults(func=run)


def _films(root) -> None:
    """Where each game's film is: on the Mac, in iCloud only, or on the Mac because a job pulled it."""
    from film.common.cloud import film_state
    from film.common.manifest import Manifest

    if not root.exists():
        return
    for gd in sorted(p for p in root.iterdir() if (p / "manifest.json").exists()):
        try:
            state = film_state(Manifest.load(gd).source)
        except (ValueError, TypeError, OSError):
            state = "unknown"
        print(f"film {state:8s} {gd.name}")


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
    _films(games_dir())
    return 0
