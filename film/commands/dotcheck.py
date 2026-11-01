# film/commands/dotcheck.py
from __future__ import annotations

import json


def register(sub) -> None:
    parser = sub.add_parser("dotcheck", help="the coach's own dots on a few shots, placed blind, to measure the machine's in feet")
    parser.add_argument("action", choices=["pick", "score"])
    parser.add_argument("--n", type=int, default=30, help="pick: how many shots")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.review.dotcheck import pick, score

    if args.action == "pick":
        return 0 if pick(args.n) else 1
    print(json.dumps(score(), indent=1))
    return 0
