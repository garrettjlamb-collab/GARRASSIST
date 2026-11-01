# film/commands/dotreview.py
from __future__ import annotations

import json


def register(sub) -> None:
    parser = sub.add_parser("dotreview", help="how the coach's review of the machine's dots stands, and what its hidden cards measure")
    parser.add_argument("action", choices=["score", "trust"])
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.review.dotreview import next_shot, score

    if args.action == "trust":
        from film.perceive.shooter_trust import write_trust

        write_trust()
    print(json.dumps({"queue": {k: v for k, v in next_shot().items() if k in ("remaining", "doubtful", "checks", "no_dot", "bulk", "done_so_far")}, **score()}, indent=1))
    return 0
