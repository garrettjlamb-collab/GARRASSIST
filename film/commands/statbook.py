from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("statbook", help="put the courtside stat book's plays on the film; clips and reels by player or play")
    parser.add_argument("game_id")
    parser.add_argument("statbook", nargs="?", help="the stat book's JSON (its export, or a raw game); omitted, reuses the last match")
    parser.add_argument("--player", help="only this player's plays (number or name)")
    parser.add_argument("--type", action="append", dest="types",
                        help="only these plays: shot reb ast stl blk to foul (repeatable)")
    parser.add_argument("--team", choices=["us", "them"])
    parser.add_argument("--made", choices=["yes", "no"], help="only made (or missed) shots")
    parser.add_argument("--pts", type=int, choices=[1, 2, 3], help="only shots worth this: 3 for threes, 1 for free throws")
    parser.add_argument("--lineup", help='only plays with exactly these five on the floor, e.g. "0 3 4 13 15"')
    parser.add_argument("--with", dest="with_players", help='only plays with all of these on the floor, e.g. "3 4"')
    parser.add_argument("--period", type=int)
    parser.add_argument("--list", action="store_true", help="print the selected plays")
    parser.add_argument("--clips", action="store_true", help="cut a labeled clip for each selected play")
    parser.add_argument("--reel", metavar="NAME", help="join the selected plays into one labeled reel")
    parser.add_argument("--all-reels", metavar="FOLDER", help="write every standard reel (players, opponent, lineups) into FOLDER")
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
    if args.all_reels:
        from film.export.statbook import standard_reels

        made = standard_reels(args.game_id, result, args.all_reels)
        print(f"{len(made)} reels in {args.all_reels}")
        return 0
    chosen = select(result["plays"], args.player, args.types, args.team,
                    made=None if args.made is None else args.made == "yes", pts=args.pts,
                    lineup=args.lineup, with_players=args.with_players, period=args.period)
    if args.list:
        for p in chosen:
            print(f"  P{p['period']} {p.get('clock_s') or '':>5} film {p['film_t']:7.1f}  {p['label']}")
        print(f"{len(chosen)} plays")
    if args.reel:
        print(f"wrote {reel(args.game_id, chosen, args.reel)} ({len(chosen)} plays)")
    elif args.clips:
        cut_clips(args.game_id, chosen)
    return 0
