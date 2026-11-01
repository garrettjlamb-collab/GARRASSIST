# film/commands/locate.py
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("locate", help="place a dot on the half court for each confirmed shot, from the ball's flight back from the rim")
    parser.add_argument("game_id")
    parser.add_argument("--again", action="store_true", help="only run the rules again over what the detectors already saw")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.common.cloud import game_film, local_source
    from film.common.device import pick_device
    from film.export.location_sheet import location_sheet
    from film.perceive.shotspots import collect, place

    with local_source(game_film(args.game_id)):
        if not args.again:
            collect(args.game_id, device=pick_device(args.device))
        place(args.game_id)
        location_sheet(args.game_id)
    return 0
