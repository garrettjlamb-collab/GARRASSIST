from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("export", help="write the reviewed shot list (CSV and JSON) and, optionally, the reel")
    parser.add_argument("game_id")
    parser.add_argument("--reel", action="store_true", help="also build export/<game_id>.reel.mp4 from confirmed shots")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.export.shots import export_reel, export_shots

    csv_path, json_path = export_shots(args.game_id)
    print(f"wrote {csv_path}\nwrote {json_path}")
    if args.reel:
        reel = export_reel(args.game_id)
        print(f"wrote {reel}" if reel else "no confirmed shots with clips; reel not built")
    return 0
