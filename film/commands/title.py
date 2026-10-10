from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("title", help="name a game for people (the review page shows it instead of the id)")
    parser.add_argument("game_id")
    parser.add_argument("title", nargs="?", help="the new title; omitted, prints the current one")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.common.manifest import Manifest
    from film.common.paths import game_dir

    gd = game_dir(args.game_id)
    m = Manifest.load(gd)
    if args.title:
        m.title = args.title
        m.save(gd)
    print(f"{m.game_id}: {m.title}")
    return 0
