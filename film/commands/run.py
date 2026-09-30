from __future__ import annotations

from pathlib import Path

from film.ingest.profile import PROFILES


def register(sub) -> None:
    parser = sub.add_parser("run", help="ingest, perceive, interpret and cut clips for one game file")
    parser.add_argument("source", type=Path)
    parser.add_argument("--date", help="game date YYYY-MM-DD (default: file modification date)")
    parser.add_argument("--profile", choices=PROFILES, help="override the camera profile")
    parser.add_argument("--fps", type=float, default=10.0, help="working frame rate (5 on the laptop)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-s", type=float, default=None, help="only process this many seconds (for a quick check)")
    parser.add_argument("--force", action="store_true", help="re-run perceive and interpret even if they are done")
    parser.set_defaults(func=run)


def run(args) -> int:
    import sys

    from film.common.paths import game_dir
    from film.export import clips
    from film.interpret.events import load_events
    from film.review.decisions import load_decisions
    from film.ingest.run import ingest
    from film.interpret import run as interpret_mod
    from film.perceive import run as perceive_mod

    m = ingest(args.source, date=args.date, profile=args.profile, fps=args.fps)
    print(f"ingest   {m.game_id}  profile={m.profile}  {m.duration_s:.0f}s")
    if m.profile == "unknown" and args.profile is None:
        from film.commands.perceive import unknown_profile_message

        print(unknown_profile_message(), file=sys.stderr)
        return 2
    def done(stage: str) -> bool:
        return m.stages.get(stage, {}).get("status") == "done" and not args.force

    if done("perceive"):
        print("perceive already done, skipped (--force to re-run)")
    else:
        perceive_mod.perceive(m.game_id, device=args.device, limit_s=args.limit_s)
    if done("perceive") and done("interpret"):          # new tracks always get a fresh interpret
        print("interpret already done, skipped (--force to re-run)")
    else:
        interpret_mod.interpret(m.game_id, device=args.device)
    gd = game_dir(m.game_id)
    if (gd / "events.json").exists():
        ids = {e.id for e in load_events(gd)}
        pruned = prune_clips(gd, ids)
        print(f"clips    pruned {pruned} stale clip files")
        orphaned = [d for d in load_decisions(gd) if d not in ids]
        if orphaned:
            print(f"review   {len(orphaned)} review decision{'s' if len(orphaned) != 1 else ''} no longer "
                  f"match{'es' if len(orphaned) == 1 else ''} an event (kept in review.json)")
    clips.clips_for_game(m.game_id)
    print(f"\nnext:  uv run film review        (then pick {m.game_id})")
    return 0


def prune_clips(gd: Path, keep_ids: set[str]) -> int:
    """Delete clips/<id>.mp4 and <id>.part.mp4 for event ids no longer in events.json."""
    folder = Path(gd) / "clips"
    if not folder.exists():
        return 0
    n = 0
    for path in folder.glob("*.mp4"):
        event_id = path.name.removesuffix(".mp4").removesuffix(".part")
        if event_id not in keep_ids:
            path.unlink()
            n += 1
    return n
