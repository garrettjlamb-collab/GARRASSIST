from __future__ import annotations

from pathlib import Path

from film.ingest.profile import PROFILES

DEFAULT_PROFILE = "hudl-mdn-2026"


def register(sub) -> None:
    parser = sub.add_parser("game", help="one game end to end: film run, the stat book on the film, and every reel")
    parser.add_argument("source", type=Path, help="the game film")
    parser.add_argument("--statbook", type=Path, required=True,
                        help="the stat book's JSON for this game (its export, or the game pulled from its store)")
    parser.add_argument("--date", help="game date YYYY-MM-DD (default: the stat book's, else the file's)")
    parser.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE,
                        help=f"camera profile (default {DEFAULT_PROFILE}: Hudl at Marcos de Niza, 2026-27; handheld for a "
                             "sideline film with no scoreboard, placed by film camstops)")
    parser.add_argument("--out", type=Path, help='where the reels go (default: "<film name> reels" beside the film)')
    parser.add_argument("--no-reels", action="store_true", help="stop after the match; cut reels later with film statbook")
    parser.add_argument("--device", default=None)
    parser.set_defaults(func=run)


def run(args) -> int:
    import argparse
    import json
    import shutil

    from film.commands.run import _run
    from film.common.cloud import local_source
    from film.export.statbook import match_game, standard_reels
    from film.ingest.run import ingest
    from film.interpret.statbook import load_statbook

    book = load_statbook(args.statbook)                       # refuse a bad stat book before half an hour of film
    date = args.date or book.game.get("date")
    run_args = argparse.Namespace(source=args.source, date=date, profile=args.profile, fps=10.0,
                                  device=args.device, limit_s=None, force=False)
    with local_source(args.source):
        code = _run(run_args)
        if code:
            return code
        game_id = ingest(args.source, date=date, profile=args.profile).game_id
        from film.perceive.scoreboard import has_scoreboard

        if not has_scoreboard(args.profile):
            # no board to find quarters by: line the live log up with each stretch the camera recorded
            from film.commands.camstops import run as camstops

            camstops(argparse.Namespace(game_id=game_id, statbook=args.statbook, off=False))
        result = match_game(game_id, args.statbook)
        for p in result["periods"]:
            print(f"period {p['period']}: {p['shots_matched']}/{p['shots_logged']} shots on film, logged {p['mode']}")
        folder = args.out or args.source.parent / f"{args.source.stem} reels"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path("games") / game_id / "export" / f"{game_id}.plays.csv", folder / "every play with film time.csv")
        if not args.no_reels:
            made = standard_reels(game_id, result, folder)
            print(f"\n{len(made)} reels in {folder}")
    (Path("games") / game_id / "statbook.source.json").write_text(json.dumps({"statbook": str(args.statbook)}))
    return 0
