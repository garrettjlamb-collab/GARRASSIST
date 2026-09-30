from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("perceive", help="one pass over the film: rims, ball, motion energy, scoreboard")
    parser.add_argument("game_id")
    parser.add_argument("--device", default=None, help="mps, cuda or cpu (default: auto)")
    parser.add_argument("--limit-s", type=float, default=None, help="stop after this many seconds of video")
    parser.set_defaults(func=run)


def run(args) -> int:
    import sys

    from film.common.manifest import Manifest
    from film.common.paths import game_dir
    from film.perceive.run import perceive

    if Manifest.load(game_dir(args.game_id)).profile == "unknown":
        print(unknown_profile_message(), file=sys.stderr)
        return 2
    perceive(args.game_id, device=args.device, limit_s=args.limit_s)
    return 0


def unknown_profile_message() -> str:
    from film.ingest.profile import PROFILES

    return ("error: the camera profile is unknown, so perceive would run for about half an hour without a "
            "scoreboard or rim templates. Re-run ingest (or `film run`) with --profile, one of: "
            + ", ".join(PROFILES))
