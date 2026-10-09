from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    parser = sub.add_parser("scoreboard", help="read the burned-in scoreboard once a second and write scoreboard.jsonl")
    parser.add_argument("game_id")
    parser.add_argument("--limit-s", type=float, default=None, help="stop after this many seconds of video")
    parser.set_defaults(func=run)


def run(args) -> int:
    from film.common.cloud import local_source
    from film.common.manifest import Manifest
    from film.common.paths import game_dir
    from film.perceive.scoreboard import build_timeline, clean_timeline, has_scoreboard, load_profile, save_timeline

    gd = game_dir(args.game_id)
    m = Manifest.load(gd)
    if not has_scoreboard(m.profile):
        print(f"profile {m.profile} has no scoreboard reader; nothing to do")
        return 1
    with local_source(Path(m.source)) as film:
        raw = build_timeline(film, load_profile(m.profile), limit_s=args.limit_s)
    kept = clean_timeline(raw)
    out = save_timeline(kept, gd / "scoreboard.jsonl")
    m.mark("scoreboard", "done", raw=len(raw), kept=len(kept))
    m.save(gd)
    print(f"reads {len(raw)} kept {len(kept)} -> {out}")
    for r in kept[:3] + kept[-3:]:
        print(f"  t={r.t:7.1f} P{r.period} {r.clock_s} {r.home}-{r.away}")
    return 0
