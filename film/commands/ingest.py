from __future__ import annotations

from pathlib import Path

from film.ingest.profile import PROFILES


def register(sub) -> None:
    parser = sub.add_parser("ingest", help="probe a game file, pick its camera profile, write the manifest")
    parser.add_argument("source", type=Path)
    parser.add_argument("--date", help="game date YYYY-MM-DD (default: file modification date)")
    parser.add_argument("--profile", choices=PROFILES, help="override the camera profile")
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
