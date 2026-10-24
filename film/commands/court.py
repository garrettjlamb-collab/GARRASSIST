# film/commands/court.py
from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("court", help="fit the court mapping through a game and write court.parquet")
    parser.add_argument("game_id")
    parser.add_argument("--step-s", type=float, default=None, help="seconds between fits (hudl 1.0, else 0.5)")
    parser.add_argument("--refs-from", default=None,
                        help="place the court from another game's reference pictures (the same fixed camera, e.g. Hudl at MDN)")
    parser.add_argument("--model", action="store_true",
                        help="use the trained court finder even when the game has reference pictures")
    parser.add_argument("--train-frames", type=int, default=0,
                        help="also save this many reference-placed frames as court-finder training labels")
    parser.add_argument("--sheet", action="store_true", help="only draw the contact sheet from the existing court.parquet")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def _print_check(game_id: str) -> None:
    from film.perceive.court_refs import check_references

    for r in check_references(game_id):
        at = f"{int(r['t'] // 60)}:{int(r['t'] % 60):02d}"
        if not r["lined_up"] or not r["shared"]:
            print(f"  reference at {at}: {r['points']} points; sees a part of the gym the others do not, so it is not checked")
        else:
            print(f"  reference at {at}: {r['points']} points; against the others {r['shared']} shared, "
                  f"median {r['median_px']} px, worst {r['worst']} {r['max_px']} px")


def run(args) -> int:
    from film.common.cloud import game_film, local_source
    from film.export.court_sheet import court_sheet
    from film.perceive.court_refs import reference_ids, refs_pass
    from film.perceive.court_run import court_pass

    with local_source(game_film(args.game_id)):
        if args.sheet:
            return 0 if court_sheet(args.game_id) else 1
        if not args.model and (args.refs_from or reference_ids(args.game_id)):
            refs_pass(args.game_id, refs_from=args.refs_from, step_s=args.step_s, train_frames=args.train_frames)
            _print_check(args.refs_from or args.game_id)
        else:
            court_pass(args.game_id, device=args.device, step_s=args.step_s)
        court_sheet(args.game_id)
    return 0
