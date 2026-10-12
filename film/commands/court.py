# film/commands/court.py
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("court", help="fit the court mapping through a game and write court.parquet")
    parser.add_argument("game_id")
    parser.add_argument("--step-s", type=float, default=None, help="seconds between fits (hudl 1.0, else 0.5)")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.perceive.court_run import court_pass

    court_pass(args.game_id, device=args.device, step_s=args.step_s)
    return 0
