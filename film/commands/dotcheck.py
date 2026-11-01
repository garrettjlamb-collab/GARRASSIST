# film/commands/dotcheck.py
from __future__ import annotations

import json


def register(sub) -> None:
    parser = sub.add_parser("dotcheck", help="the coach's own dots on a few shots, placed blind, to measure the machine's in feet")
    parser.add_argument("action", choices=["pick", "score", "doubt"])
    parser.add_argument("--n", type=int, default=30, help="pick: how many shots")
    parser.add_argument("--shots", type=int, nargs="*", default=[], help="doubt: shot numbers on the page he says may be the wrong shot")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.review.dotcheck import mark_doubtful, pick, score

    if args.action == "pick":
        return 0 if pick(args.n) else 1
    if args.action == "doubt":
        for a in mark_doubtful(args.shots):
            print("doubtful:", a["game_id"], a["event_id"])
        return 0
    print(json.dumps(score(), indent=1))
    return 0
