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
