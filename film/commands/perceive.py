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
