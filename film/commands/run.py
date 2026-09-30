from __future__ import annotations

from pathlib import Path

from film.ingest.profile import PROFILES


def register(sub) -> None:
    parser = sub.add_parser("run", help="ingest, perceive, interpret and cut clips for one game file")
    parser.add_argument("source", type=Path)
    parser.add_argument("--date", help="game date YYYY-MM-DD (default: file modification date)")
    parser.add_argument("--profile", choices=PROFILES, help="override the camera profile")
    parser.add_argument("--fps", type=float, default=10.0, help="working frame rate (5 on the laptop)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-s", type=float, default=None, help="only process this many seconds (for a quick check)")
    parser.set_defaults(func=run)


def run(args) -> int:
    import sys

    from film.export import clips
    from film.ingest.run import ingest
    from film.interpret import run as interpret_mod
    from film.perceive import run as perceive_mod

    m = ingest(args.source, date=args.date, profile=args.profile, fps=args.fps)
    print(f"ingest   {m.game_id}  profile={m.profile}  {m.duration_s:.0f}s")
    if m.profile == "unknown" and args.profile is None:
        from film.commands.perceive import unknown_profile_message

        print(unknown_profile_message(), file=sys.stderr)
        return 2
    perceive_mod.perceive(m.game_id, device=args.device, limit_s=args.limit_s)
    interpret_mod.interpret(m.game_id, device=args.device)
    clips.clips_for_game(m.game_id)
    print(f"\nnext:  uv run film review        (then pick {m.game_id})")
    return 0
