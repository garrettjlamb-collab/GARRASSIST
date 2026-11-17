from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("statbook", help="put the courtside stat book's plays on the film; clips and reels by player or play")
    parser.add_argument("game_id")
    parser.add_argument("statbook", nargs="?", help="the stat book's JSON (its export, or a raw game); omitted, reuses the last match")
    parser.add_argument("--player", help="only this player's plays (number or name)")
    parser.add_argument("--type", action="append", dest="types",
                        help="only these plays: shot reb ast stl blk to foul (repeatable)")
    parser.add_argument("--team", choices=["us", "them"])
    parser.add_argument("--clips", action="store_true", help="cut a labeled clip for each selected play")
    parser.add_argument("--reel", metavar="NAME", help="join the selected plays into one labeled reel")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.export.statbook import cut_clips, load_result, match_game, reel, select

    if args.statbook:
        result = match_game(args.game_id, args.statbook)
        for p in result["periods"]:
            print(f"period {p['period']}: {p['shots_matched']}/{p['shots_logged']} shots on film, "
                  f"{p['fts_matched']}/{p['fts_logged']} free throws, {p['cards']} cards "
                  f"({p['strong_cards_unmatched']} sure cards unmatched), logged {p['mode']}, +/-{p['sigma_s']}s")
        hows = {}
        for x in result["plays"]:
            hows[x["how"]] = hows.get(x["how"], 0) + 1
        print("plays placed: " + ", ".join(f"{k} {v}" for k, v in sorted(hows.items())))
    else:
        result = load_result(args.game_id)
    chosen = select(result["plays"], args.player, args.types, args.team)
    if args.reel:
        print(f"wrote {reel(args.game_id, chosen, args.reel)} ({len(chosen)} plays)")
    elif args.clips:
        cut_clips(args.game_id, chosen)
    return 0
