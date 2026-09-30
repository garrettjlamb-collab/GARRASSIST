from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("clips", help="cut a short clip for every event, for the review page")
    parser.add_argument("game_id")
    parser.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.export.clips import clips_for_game

    clips_for_game(args.game_id, force=args.force)
    return 0
